#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Miss diagnostic: measure a safety lens on a trap corpus by exact search.

For a given lens (.aqwt) applied to a given AQTC binary (E5 embeddings of
300 cases + N distractors) and the source dataset JSON, emit for each
case:

  case_id, trap_type, rank_b (in top-K), rank_c, top_outrankers, top-K
  scores + pool indices.

Lists every HN leak (rank_b <= K) and every HitC miss (rank_c > K).
With --raw instead of --lens it measures raw cosine on the E5 vectors
themselves (Cell A, no lens), on the same pool and by the same ranking.
The ranking is exact: every pool row is scored against the query and
the scores are sorted. No approximate index is involved.

Usage:
    python3 scripts/miss_diagnostic.py \\
        --lens lenses/medical_d2c_d32.aqwt \\
        --dataset data/traps/medical_300.json \\
        --aqtc medical_5k.aqtc \\
        --out medical_d2c_d32_on_medical_5k.json \\
        --k 10

    python3 scripts/miss_diagnostic.py --raw \\
        --dataset data/traps/medical_300.json --aqtc medical_5k.aqtc \\
        --out raw_cosine_on_medical_5k.json
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import List

import numpy as np


def read_aqwt_v2(path: Path):
    data = path.read_bytes()
    if len(data) < 18 or data[:4] != b"AQWT":
        raise SystemExit(f"Invalid AQWT file: {path}")
    ver = data[4]
    if ver != 2:
        raise SystemExit(f"Unsupported AQWT version {ver} in {path}")
    orig = int(struct.unpack_from("<H", data, 5)[0])
    comp = int(struct.unpack_from("<H", data, 7)[0])
    n = orig * comp
    weights = np.frombuffer(data, dtype="<f4", count=n, offset=18).astype(
        np.float32, copy=True
    )
    return orig, comp, weights.reshape(comp, orig).T  # (D, d)


def read_aqtc_v1(path: Path):
    """Read the AQTC v1 binary. Layout:
       u32 magic 'AQTC' (0x43545141) — 4 bytes
       u32 version                    — 4 bytes
       u32 embedding_dim              — 4 bytes
       u32 n_cases                    — 4 bytes
       u32 n_distractors              — 4 bytes
       f32[3 * n_cases * dim] triplets — interleaved A/B/C per case
       f32[n_distr * dim] distractors
    """
    data = path.read_bytes()
    magic, ver, dim, n_cases, n_distr = struct.unpack_from("<IIIII", data, 0)
    if magic != 0x43545141:
        raise SystemExit(f"Bad magic {magic:#x} in {path}")
    if ver != 1:
        raise SystemExit(f"Unsupported AQTC ver {ver}")
    off = 20
    tri_bytes = 4 * 3 * n_cases * dim
    triplets = np.frombuffer(data, dtype="<f4", count=3 * n_cases * dim, offset=off).reshape(
        3 * n_cases, dim
    )
    off += tri_bytes
    distractors = np.frombuffer(data, dtype="<f4", count=n_distr * dim, offset=off).reshape(
        n_distr, dim
    )
    return dim, n_cases, n_distr, triplets, distractors


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def main():
    ap = argparse.ArgumentParser()
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--lens", type=Path)
    which.add_argument("--raw", action="store_true", help="no lens: raw cosine on the E5 vectors (Cell A)")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--aqtc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    if args.raw:
        w = None
        print("[miss_diag] no lens: raw cosine (Cell A)", file=sys.stderr)
    else:
        print(f"[miss_diag] loading lens {args.lens}", file=sys.stderr)
        orig, comp, w = read_aqwt_v2(args.lens)
        print(f"[miss_diag]   lens: {orig}D -> {comp}D", file=sys.stderr)

    print(f"[miss_diag] loading dataset {args.dataset}", file=sys.stderr)
    ds = json.loads(args.dataset.read_text())
    cases = ds["test_cases"]

    print(f"[miss_diag] loading AQTC {args.aqtc}", file=sys.stderr)
    dim, n_cases, n_distr, triplets, distractors = read_aqtc_v1(args.aqtc)
    print(
        f"[miss_diag]   dim={dim} n_cases={n_cases} n_distr={n_distr}",
        file=sys.stderr,
    )
    assert w is None or dim == orig, f"AQTC dim {dim} != lens orig {orig}"
    assert n_cases == len(cases), f"AQTC n_cases {n_cases} != dataset {len(cases)}"

    # Project through lens (raw x @ W)
    print(f"[miss_diag] projecting triplets + distractors...", file=sys.stderr)
    if w is None:
        tri_proj = l2_normalize(triplets.astype(np.float32))
        dis_proj = l2_normalize(distractors.astype(np.float32))
    else:
        tri_proj = l2_normalize((triplets @ w).astype(np.float32))
        dis_proj = l2_normalize((distractors @ w).astype(np.float32))

    # Build the pool: interleaved A/B/C per case + distractors (matches AQTC layout)
    # But for ranking: for each case i, the pool from case i's perspective is
    # "everyone except case i's A" = pool_idx: 3*j for j != i (A_j),
    # 3*j+1 (B_j), 3*j+2 (C_j), + distractors.
    # The test's ranking uses the SAME pool for every case: all 3n case-rows +
    # all distractors. So B_i and C_i are within the pool as their pool indices.
    pool_proj = np.concatenate([tri_proj, dis_proj], axis=0)
    n_pool = pool_proj.shape[0]
    print(f"[miss_diag]   pool={n_pool}", file=sys.stderr)

    misses = []
    hn_leaks = []
    hitc_misses = []

    for i, tc in enumerate(cases):
        a_idx = 3 * i
        b_idx = 3 * i + 1
        c_idx = 3 * i + 2
        a_proj = tri_proj[a_idx]

        # Score everything (cosine = dot on L2-normalized)
        scores = pool_proj @ a_proj

        # Exclude A itself (rank on pool minus a_idx).
        # A common convention: rank by descending, find where B and C fall.
        # Rank is 1-indexed and A_i is skipped.
        order = np.argsort(-scores)  # descending
        # Filter out A_i from order
        order_filtered = order[order != a_idx]

        # Where is B in the ranked order?
        rank_b_pos = int(np.where(order_filtered == b_idx)[0][0]) + 1  # 1-indexed
        rank_c_pos = int(np.where(order_filtered == c_idx)[0][0]) + 1

        top_k_indices = order_filtered[: args.k].tolist()
        top_k_scores = scores[top_k_indices].tolist()

        # Are B or C in top-K?
        b_in_topk = rank_b_pos <= args.k
        c_in_topk = rank_c_pos <= args.k

        entry = {
            "case_index": i,
            "case_id": tc.get("id", f"case_{i}"),
            "trap_type": tc.get("trap_type", "unknown"),
            "clause_a": tc["clause_a"],
            "clause_b": tc["clause_b"],
            "clause_c": tc["clause_c"],
            "rank_b": rank_b_pos,
            "rank_c": rank_c_pos,
            "score_b": float(scores[b_idx]),
            "score_c": float(scores[c_idx]),
            "top_k": [
                {
                    "pool_idx": int(idx),
                    "score": float(sc),
                    "kind": (
                        "case_A" if idx < 3 * n_cases and idx % 3 == 0
                        else "case_B" if idx < 3 * n_cases and idx % 3 == 1
                        else "case_C" if idx < 3 * n_cases
                        else "distractor"
                    ),
                    "case_ref": (idx // 3) if idx < 3 * n_cases else None,
                }
                for idx, sc in zip(top_k_indices, top_k_scores)
            ],
        }

        if b_in_topk:
            hn_leaks.append(entry)
        if not c_in_topk:
            hitc_misses.append(entry)

    out = {
        "meta": {
            "lens": "none (raw cosine)" if args.raw else str(args.lens),
            "dataset": str(args.dataset),
            "aqtc": str(args.aqtc),
            "k": args.k,
            "n_cases": len(cases),
            "n_pool": n_pool,
            "n_distr": n_distr,
        },
        "summary": {
            "hn_leaks_count": len(hn_leaks),
            "hitc_misses_count": len(hitc_misses),
            "hn_at_k": len(hn_leaks) / len(cases),
            "hitc_at_k": 1.0 - len(hitc_misses) / len(cases),
        },
        "hn_leak_cases": hn_leaks,
        "hitc_miss_cases": hitc_misses,
    }

    args.out.write_text(json.dumps(out, indent=2))
    print(
        f"[miss_diag] wrote {args.out}: HN leaks={len(hn_leaks)}, "
        f"HitC misses={len(hitc_misses)} at k={args.k}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
