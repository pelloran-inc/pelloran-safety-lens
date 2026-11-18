#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Build trap-agnostic baseline projections at matched d.

Two baselines, written in the SAME AQWT v2 format the shipped lenses use, so
they are measured by the UNMODIFIED scripts/miss_diagnostic.py (identical to
the paper's Cell C path):

  RP(d)   random Gaussian W ~ N(0,1), shape (1024, d), numpy default_rng(seed=42).
          Domain-independent (one matrix per d).
  PCA(d)  uncentered PCA: top-d right singular vectors of the domain's 5k pool
          DISTRACTOR rows only (never the A/B/C triples -> no label leakage),
          applied as x @ W, matching how the lens is applied (pure linear, no
          centering/bias in miss_diagnostic). See DEVIATION_108.75 for why
          uncentered (preserves the unmodified-path guarantee of the PRE_REG).

Writes <out>/rp_d{D}.aqwt and <out>/pca_<domain>_d{D}.aqwt.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

SEED_RP = 42


def read_aqtc_v1_distractors(path: Path):
    data = path.read_bytes()
    magic, ver, dim, n_cases, n_distr = struct.unpack_from("<IIIII", data, 0)
    if magic != 0x43545141:
        raise SystemExit(f"Bad AQTC magic {magic:#x} in {path}")
    if ver != 1:
        raise SystemExit(f"Unsupported AQTC ver {ver}")
    off = 20 + 4 * 3 * n_cases * dim  # skip the triplets
    distr = np.frombuffer(data, dtype="<f4", count=n_distr * dim, offset=off).reshape(
        n_distr, dim
    )
    return dim, n_distr, np.array(distr, dtype=np.float64)  # f64 for a stable SVD


def write_aqwt_v2(path: Path, W: np.ndarray):
    """W has shape (orig=1024, comp=d). Stored so read_aqwt_v2 returns W exactly."""
    orig, comp = W.shape
    hdr = b"AQWT" + bytes([2]) + struct.pack("<HH", orig, comp) + b"\x00" * 9  # 18 bytes
    body = np.ascontiguousarray(W.T, dtype="<f4").tobytes()  # stored as (comp, orig) C-order
    path.write_bytes(hdr + body)
    assert path.stat().st_size == 18 + 4 * orig * comp, "AQWT size mismatch"


def rp_matrix(d: int) -> np.ndarray:
    rng = np.random.default_rng(SEED_RP)
    return rng.standard_normal((1024, d)).astype(np.float32)


def pca_matrix(distractors: np.ndarray, d: int) -> np.ndarray:
    # uncentered PCA: top-d eigenvectors of D^T D (1024x1024, symmetric PSD)
    gram = distractors.T @ distractors  # (1024, 1024)
    evals, evecs = np.linalg.eigh(gram)  # ascending
    W = evecs[:, -d:][:, ::-1]  # top-d, descending eigenvalue
    return np.ascontiguousarray(W, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aqtc-dir", type=Path, required=True,
                    help="dir holding <domain>_5k.aqtc for pca fit")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[11, 32])
    ap.add_argument("--domains", nargs="+",
                    default=["legal", "medical", "patent", "financial"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for d in args.dims:
        W = rp_matrix(d)
        p = args.out / f"rp_d{d}.aqwt"
        write_aqwt_v2(p, W)
        print(f"[baseline] RP   d={d}: {p} std={W.std():.4f} ({p.stat().st_size} B)",
              file=sys.stderr)

    for dom in args.domains:
        aqtc = args.aqtc_dir / f"{dom}_5k.aqtc"
        if not aqtc.exists():
            print(f"[baseline] SKIP pca {dom}: {aqtc} missing", file=sys.stderr)
            continue
        dim, n_distr, distr = read_aqtc_v1_distractors(aqtc)
        for d in args.dims:
            W = pca_matrix(distr, d)
            p = args.out / f"pca_{dom}_d{d}.aqwt"
            write_aqwt_v2(p, W)
            print(f"[baseline] PCA  {dom} d={d}: {p} n_distr={n_distr} "
                  f"std={W.std():.4f} ({p.stat().st_size} B)", file=sys.stderr)


if __name__ == "__main__":
    main()
