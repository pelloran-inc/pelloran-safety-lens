#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Query-only rewording diagnostic.

Measures a reworded user query against an UNCHANGED corpus: the search pool
is the ORIGINAL A/B/C triples plus the original distractors, exactly as
shipped; only the query vector is the reworded A'. For each case i the query
is A'_i and the pool is every original row except the case's own original A_i
(the usual self-exclusion). This is the clean form of the query-only test:
nothing in the pool is reworded, unlike a pool rebuilt from A' rows.

  --orig-aqtc   : the UNCHANGED pool (original A/B/C + distractors)
  --query-aqtc  : an AQTC whose A rows hold the reworded queries A'
  --lens        : the lens .aqwt (projection applied as x @ W, then L2-norm)
"""
import argparse, json, struct, sys
from pathlib import Path
import numpy as np


def read_aqwt_v2(path: Path):
    d = path.read_bytes()
    assert d[:4] == b"AQWT" and d[4] == 2, f"bad AQWT {path}"
    orig = struct.unpack_from("<H", d, 5)[0]; comp = struct.unpack_from("<H", d, 7)[0]
    w = np.frombuffer(d, dtype="<f4", count=orig * comp, offset=18).reshape(comp, orig).T
    return orig, comp, np.array(w, dtype=np.float32)


def read_aqtc_v1(path: Path):
    d = path.read_bytes()
    magic, ver, dim, nc, nd = struct.unpack_from("<IIIII", d, 0)
    assert magic == 0x43545141 and ver == 1, f"bad AQTC {path}"
    off = 20
    tri = np.frombuffer(d, dtype="<f4", count=3 * nc * dim, offset=off).reshape(3 * nc, dim)
    off += 4 * 3 * nc * dim
    dis = np.frombuffer(d, dtype="<f4", count=nd * dim, offset=off).reshape(nd, dim)
    return dim, nc, nd, np.array(tri, np.float32), np.array(dis, np.float32)


def l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return x / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--orig-aqtc", type=Path, required=True, help="UNCHANGED pool: original A/B/C + distractors")
    ap.add_argument("--query-aqtc", type=Path, required=True, help="AQTC whose A rows are the reworded queries A'")
    ap.add_argument("--dataset", type=Path, required=True, help="dataset JSON for the cases (count must match)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    _, _, W = read_aqwt_v2(args.lens)
    dim, nc, nd, tri, dis = read_aqtc_v1(args.orig_aqtc)        # UNCHANGED pool
    qdim, qnc, qnd, qtri, _ = read_aqtc_v1(args.query_aqtc)     # A' in rows 0,3,6,...
    assert dim == qdim and nc == qnc, "orig/query dim or case-count mismatch"
    cases = json.loads(args.dataset.read_text())["test_cases"]
    assert len(cases) == nc, f"dataset {len(cases)} != aqtc {nc}"

    pool = l2(np.concatenate([tri, dis], axis=0) @ W)           # original pool, projected
    queries = l2(qtri[0::3] @ W)                                # A' projected (reworded queries only)
    n_pool = pool.shape[0]

    hn = hitc = 0
    hn_cases, hitc_miss = [], []
    for i in range(nc):
        a, b, c = 3 * i, 3 * i + 1, 3 * i + 2                   # indices into the ORIGINAL pool
        scores = pool @ queries[i]
        order = np.argsort(-scores)
        order = order[order != a]                              # exclude the case's own original A
        rank_b = int(np.where(order == b)[0][0]) + 1
        rank_c = int(np.where(order == c)[0][0]) + 1
        if rank_b <= args.k: hn += 1; hn_cases.append(cases[i].get("id"))
        if rank_c > args.k: hitc_miss.append(cases[i].get("id"))
        else: hitc += 1

    out = {
        "meta": {"lens": str(args.lens), "orig_aqtc": str(args.orig_aqtc),
                 "query_aqtc": str(args.query_aqtc), "dataset": str(args.dataset),
                 "k": args.k, "n_cases": nc, "n_pool": n_pool,
                 "note": "query-only: reworded query A' vs UNCHANGED original pool; pool contains NO reworded rows"},
        "summary": {"hn_leaks_count": hn, "hitc_misses_count": nc - hitc,
                    "hn_at_k": hn / nc, "hitc_at_k": hitc / nc},
        "hn_leak_ids": hn_cases, "hitc_miss_ids": hitc_miss,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[query_only] HN@{args.k}={hn/nc:.4f} ({hn}/{nc})  HitC@{args.k}={hitc/nc:.4f} ({hitc}/{nc})  "
          f"n_pool={n_pool} (unchanged original pool)", file=sys.stderr)


if __name__ == "__main__":
    main()
