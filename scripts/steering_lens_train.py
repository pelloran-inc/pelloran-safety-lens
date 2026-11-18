#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Steering Lens Trainer (prototype, scientific).

Purpose:
- Learn a linear projection W: R^D -> R^d that:
  (1) preserves baseline geometry (structure preservation)
  (2) suppresses hard negatives (A should not retrieve B)

A minimal research trainer, as used for the paper.

Security:
- No secrets. Uses precomputed embeddings input.

Environment note:
- If LD_LIBRARY_PATH points at another libtorch build, importing the Python
  torch wheel can fail. Run with:
    env -u LD_LIBRARY_PATH python3 scripts/steering_lens_train.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np


def _require_safe_env() -> None:
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "libtorch" in ld:
        raise SystemExit(
            "LD_LIBRARY_PATH points at a separate libtorch build, which can clash with PyTorch's own libraries. "
            "Re-run with: env -u LD_LIBRARY_PATH python3 scripts/steering_lens_train.py ..."
        )


def _normalize_rows_np(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def load_liability_triplets(dataset_path: Path) -> List[Tuple[int, int, int]]:
    """
    For the Liability Trap JSON, we use the deterministic corpus ordering:
      for each case i: append A, then B, then C.
    Returns list of (a_idx, b_idx, c_idx) in that corpus ordering.
    """
    ds = json.load(dataset_path.open("r", encoding="utf-8"))
    cases = ds["test_cases"]
    triples: List[Tuple[int, int, int]] = []
    idx = 0
    for _tc in cases:
        a = idx
        b = idx + 1
        c = idx + 2
        triples.append((a, b, c))
        idx += 3
    return triples


def read_embeddings_json(path: Path) -> np.ndarray:
    embs = json.load(path.open("r", encoding="utf-8"))
    x = np.asarray(embs, dtype=np.float32)
    if x.ndim != 2:
        raise SystemExit(f"Expected 2D embeddings array in {path}, got shape {x.shape}")
    return x


def aqwt_read_matrix(path: Path) -> Tuple[int, int, int, float, float, np.ndarray]:
    """
    Read AQWT v2 weights into a matrix W with shape (D, d) as used by x @ W.
    File stores weights as row-major (compressed_dim × original_dim).
    """
    data = path.read_bytes()
    if len(data) < 18 or data[:4] != b"AQWT":
        raise SystemExit(f"Invalid AQWT file: {path}")
    ver = data[4]
    if ver != 2:
        raise SystemExit(f"Unsupported AQWT version {ver} in {path}")
    orig = int(struct.unpack_from("<H", data, 5)[0])
    comp = int(struct.unpack_from("<H", data, 7)[0])
    model_type = int(struct.unpack_from("<B", data, 9)[0])
    spearman = float(struct.unpack_from("<f", data, 10)[0])
    rot = float(struct.unpack_from("<f", data, 14)[0])
    n = orig * comp
    weights = np.frombuffer(data, dtype="<f4", count=n, offset=18).astype(np.float32, copy=True)
    # Stored as (comp, orig). Convert to (orig, comp) for x @ W.
    w = weights.reshape(comp, orig).T.copy()
    return orig, comp, model_type, spearman, rot, w


def aqwt_write_matrix(
    path: Path,
    w: np.ndarray,
    model_type: int,
    spearman: float = 0.0,
    rotation_scale: float = 1.75,
) -> None:
    """
    Write AQWT v2 weights from matrix W with shape (D, d) as used by x @ W.
    Stored as row-major (d × D).
    """
    if w.ndim != 2:
        raise SystemExit(f"W must be 2D, got {w.shape}")
    orig = int(w.shape[0])
    comp = int(w.shape[1])
    weights = w.T.reshape(-1).astype(np.float32, copy=False)

    buf = bytearray()
    buf += b"AQWT"
    buf += struct.pack("<B", 2)
    buf += struct.pack("<H", orig)
    buf += struct.pack("<H", comp)
    buf += struct.pack("<B", int(model_type))
    buf += struct.pack("<f", float(spearman))
    buf += struct.pack("<f", float(rotation_scale))
    buf += struct.pack("<%sf" % (orig * comp), *[float(x) for x in weights])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)


@dataclass
class HyperParams:
    compressed_dim: int = 52
    alpha_preserve: float = 1.0
    beta_hardneg: float = 3.0
    margin: float = 0.05
    l2: float = 1e-4
    pairs_per_epoch: int = 8000
    epochs: int = 200
    lr: float = 5e-3
    seed: int = 42
    init_aqwt: Optional[Path] = None


@dataclass
class TrainResult:
    weights: np.ndarray
    metrics: Dict[str, float]
    final_loss: float
    epochs: int
    model_type: int
    rotation_scale: float


def evaluate_triplets(
    Z: "torch.Tensor", triples: "torch.Tensor", margin: float, baseline_sim: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute basic metrics on the embedded triplets:
    - triplet_accuracy: share of cases where sim(A,C) > sim(A,B)+margin
    - hinge_loss: mean hinge used during training
    - preserve_mse (optional): MSE vs baseline cosine similarities if provided
    """
    import torch  # local import

    A = triples[:, 0]
    B = triples[:, 1]
    C = triples[:, 2]

    sim_ab = (Z[A] * Z[B]).sum(dim=1)
    sim_ac = (Z[A] * Z[C]).sum(dim=1)
    hinge = torch.relu(sim_ab - sim_ac + margin)
    acc = (sim_ac - sim_ab > margin).float().mean().item()

    out: Dict[str, float] = {
        "triplet_accuracy": float(acc),
        "hinge_loss": float(hinge.mean().item()),
    }

    if baseline_sim is not None:
        n = baseline_sim.shape[0]
        if n <= 512:
            target = baseline_sim[np.triu_indices(n, k=1)]
            sim_full = (Z @ Z.T)
            sim = sim_full.cpu().numpy()[np.triu_indices(n, k=1)]
        else:
            # approximate sampling for large n to avoid O(n^2) memory
            k = min(30000, n * 5)
            rng = np.random.default_rng(0)
            i = rng.integers(0, n, size=k)
            j = rng.integers(0, n, size=k)
            same = i == j
            j[same] = (j[same] + 1) % n
            swap = j < i
            i[swap], j[swap] = j[swap], i[swap]
            target = baseline_sim[i, j]
            sim = (Z[i] * Z[j]).sum(dim=1).detach().cpu().numpy()
        mse = float(np.mean((sim - target) ** 2))
        out["preserve_mse"] = mse

    return out


def train_once(
    x0: np.ndarray,
    triples: List[Tuple[int, int, int]],
    hparams: HyperParams,
    baseline_sim: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> TrainResult:
    """
    Single training run returning weights + metrics (no file I/O).
    """
    _require_safe_env()
    import torch  # local import after env guard

    x0 = _normalize_rows_np(x0.astype(np.float32, copy=False))
    n, d0 = x0.shape
    triple_idx = torch.tensor(triples, dtype=torch.long)  # (T,3)
    A = triple_idx[:, 0]
    B = triple_idx[:, 1]
    C = triple_idx[:, 2]

    if n != len(triples) * 3:
        raise SystemExit(f"Expected {len(triples)*3} embeddings, got {n}")

    if baseline_sim is None:
        baseline_sim = (x0 @ x0.T).astype(np.float32)

    torch.manual_seed(int(hparams.seed))
    X = torch.tensor(x0, dtype=torch.float32)

    d = int(hparams.compressed_dim)
    if hparams.init_aqwt:
        orig, comp, model_type, _sp, rot, w0 = aqwt_read_matrix(Path(hparams.init_aqwt))
        if orig != d0 or comp != d:
            raise SystemExit(f"Init AQWT dims mismatch: got {orig}->{comp}, expected {d0}->{d}")
        W = torch.tensor(w0, dtype=torch.float32, requires_grad=True)
        out_model_type = model_type
        out_rot = rot
    else:
        W = torch.randn(d0, d, dtype=torch.float32) * 0.02
        W.requires_grad_(True)
        out_model_type = 3 if d0 in (1536, 3072) else 1
        out_rot = 1.75

    opt = torch.optim.Adam([W], lr=float(hparams.lr))

    rng = np.random.default_rng(int(hparams.seed))
    all_pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int32)

    alpha = float(hparams.alpha_preserve)
    beta = float(hparams.beta_hardneg)
    margin = float(hparams.margin)
    l2 = float(hparams.l2)

    def normalize_rows_t(t: torch.Tensor) -> torch.Tensor:
        nrm = torch.linalg.norm(t, dim=1, keepdim=True).clamp_min(1e-12)
        return t / nrm

    best_loss = float("inf")
    best_w = None

    for epoch in range(int(hparams.epochs)):
        opt.zero_grad()

        Z = normalize_rows_t(X @ W)  # (N,d)

        idx = rng.choice(all_pairs.shape[0], size=min(int(hparams.pairs_per_epoch), all_pairs.shape[0]), replace=False)
        ij = all_pairs[idx]
        i = torch.tensor(ij[:, 0], dtype=torch.long)
        j = torch.tensor(ij[:, 1], dtype=torch.long)
        target = torch.tensor(baseline_sim[ij[:, 0], ij[:, 1]], dtype=torch.float32)
        sim = (Z[i] * Z[j]).sum(dim=1)
        loss_pres = torch.mean((sim - target) ** 2)

        sim_ab = (Z[A] * Z[B]).sum(dim=1)
        sim_ac = (Z[A] * Z[C]).sum(dim=1)
        loss_hn = torch.relu(sim_ab - sim_ac + margin).mean()

        loss_reg = (W * W).mean()

        loss = alpha * loss_pres + beta * loss_hn + l2 * loss_reg
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = float(loss.item())
            best_w = W.detach().clone()

        if verbose and ((epoch + 1) % 25 == 0 or epoch == 0):
            print(
                f"epoch {epoch+1:4d} | loss={loss.item():.6f} "
                f"(pres={loss_pres.item():.6f}, hn={loss_hn.item():.6f})"
            )

    if best_w is None:
        best_w = W.detach().clone()

    Z_final = (X @ best_w)
    Z_final = Z_final / torch.linalg.norm(Z_final, dim=1, keepdim=True).clamp_min(1e-12)
    metrics = evaluate_triplets(Z_final, triple_idx, margin, baseline_sim=baseline_sim)
    metrics["final_loss"] = best_loss

    w_out = best_w.detach().cpu().numpy().astype(np.float32)
    return TrainResult(
        weights=w_out,
        metrics=metrics,
        final_loss=best_loss,
        epochs=int(hparams.epochs),
        model_type=int(out_model_type),
        rotation_scale=float(out_rot),
    )


def main() -> None:
    _require_safe_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="Embeddings JSON aligned to Liability corpus (150xD).")
    ap.add_argument("--dataset", required=True, help="Liability Trap dataset JSON.")
    ap.add_argument("--out-aqwt", required=True, help="Output .aqwt path for the trained steering lens.")

    ap.add_argument("--compressed-dim", type=int, default=52)
    ap.add_argument("--init-aqwt", default=None, help="Optional init weights (.aqwt) to finetune from.")

    ap.add_argument("--alpha-preserve", type=float, default=1.0, help="Weight for structure preservation loss.")
    ap.add_argument("--beta-hardneg", type=float, default=3.0, help="Weight for hard negative suppression loss.")
    ap.add_argument("--margin", type=float, default=0.05, help="Hinge margin for (A,B) vs (A,C).")
    ap.add_argument("--l2", type=float, default=1e-4)

    ap.add_argument("--pairs-per-epoch", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    emb_path = Path(args.embeddings)
    ds_path = Path(args.dataset)
    out_path = Path(args.out_aqwt)

    x0 = read_embeddings_json(emb_path)
    triples = load_liability_triplets(ds_path)

    hparams = HyperParams(
        compressed_dim=int(args.compressed_dim),
        alpha_preserve=float(args.alpha_preserve),
        beta_hardneg=float(args.beta_hardneg),
        margin=float(args.margin),
        l2=float(args.l2),
        pairs_per_epoch=int(args.pairs_per_epoch),
        epochs=int(args.epochs),
        lr=float(args.lr),
        seed=int(args.seed),
        init_aqwt=Path(args.init_aqwt) if args.init_aqwt else None,
    )

    result = train_once(x0, triples, hparams, baseline_sim=None, verbose=True)
    aqwt_write_matrix(
        out_path,
        result.weights,
        model_type=int(result.model_type),
        spearman=0.0,
        rotation_scale=float(result.rotation_scale),
    )
    print(json.dumps({"metrics": result.metrics, "epochs": result.epochs}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

