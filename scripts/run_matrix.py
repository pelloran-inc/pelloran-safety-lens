#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Cross-domain contamination check: every lens on every trap corpus.

Runs scripts/miss_diagnostic.py for each lens in lenses/ on each AQTC
binary in --aqtc-dir named <corpus>_<size>.aqtc (for example
patent_5k.aqtc), with the trap set data/traps/<corpus>_<n>.json, and
writes <lens>_on_<corpus>_<size>.json into --out-dir.

A lens trained on one domain is expected to fail on the others
(HN@10 well above 0.30, paper §3.6). If it does not, it has learned a
generic filter rather than the structure of its own domain.

Usage:
    python3 scripts/run_matrix.py --aqtc-dir DIR --out-dir OUT [--size 5k]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MISS = REPO / "scripts" / "miss_diagnostic.py"


def trap_set(corpus: str) -> Path:
    found = [p for p in (REPO / "data" / "traps").glob(f"{corpus}_*.json")
             if not p.stem.startswith(f"{corpus}_base")]
    if len(found) != 1:
        raise SystemExit(f"expected one trap set for {corpus}, found {[p.name for p in found]}")
    return found[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aqtc-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--size", default="5k")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpora = sorted(p.name[: -len(f"_{args.size}.aqtc")]
                     for p in args.aqtc_dir.glob(f"*_{args.size}.aqtc"))
    if not corpora:
        print(f"no *_{args.size}.aqtc in {args.aqtc_dir}", file=sys.stderr)
        return 2
    for lens in sorted((REPO / "lenses").glob("*.aqwt")):
        for corpus in corpora:
            out = args.out_dir / f"{lens.stem}_on_{corpus}_{args.size}.json"
            subprocess.run(
                [sys.executable, str(MISS), "--lens", str(lens), "--dataset", str(trap_set(corpus)),
                 "--aqtc", str(args.aqtc_dir / f"{corpus}_{args.size}.aqtc"),
                 "--out", str(out), "--k", str(args.k)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            s = json.loads(out.read_text())["summary"]
            print(f"{lens.stem:>18} on {corpus:<10} HN@{args.k}={s['hn_at_k']:.4f}"
                  f"  HitC@{args.k}={s['hitc_at_k']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
