#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Reproduce the stored results.

For every results/**/*.json, run scripts/miss_diagnostic.py with the lens
(or --raw, for a raw-cosine result) and the trap set named in its meta block, on the AQTC binary of the same
name in --aqtc-dir, and compare n_pool, HN@10 and HitC@10 with the stored
values. They must be equal, not close: the measurement is exact search.

A result whose binary is not in --aqtc-dir is skipped and listed.

Exit codes:
    0    every checked result matches, and at least one was checked
    1    at least one mismatch
    2    nothing could be checked

Usage:
    python3 scripts/check_results.py --aqtc-dir DIR [--only SUBSTRING]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MISS = REPO / "scripts" / "miss_diagnostic.py"


RAW = "none (raw cosine)"   # the lens of a raw-cosine result (miss_diagnostic.py --raw)


def measure(lens: str, dataset: Path, aqtc: Path, k: int) -> dict:
    which = ["--raw"] if lens == RAW else ["--lens", str(REPO / lens)]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.json"
        subprocess.run(
            [sys.executable, str(MISS), *which, "--dataset", str(dataset),
             "--aqtc", str(aqtc), "--out", str(out), "--k", str(k)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        r = json.loads(out.read_text())
    return {"n_pool": r["meta"]["n_pool"], "hn_at_k": r["summary"]["hn_at_k"],
            "hitc_at_k": r["summary"]["hitc_at_k"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aqtc-dir", type=Path, required=True)
    ap.add_argument("--only", default=None, help="check only results whose path contains this")
    args = ap.parse_args()

    checked = mismatched = 0
    skipped = []
    for res in sorted((REPO / "results").rglob("*.json")):
        rel = res.relative_to(REPO)
        if args.only and args.only not in str(rel):
            continue
        r = json.loads(res.read_text())
        if "meta" not in r:   # expected-value manifests (baselines/, robustness/) are not result files;
            continue          # they are checked by scripts/reproduce_extras.sh instead
        meta = r["meta"]
        aqtc = args.aqtc_dir / meta["aqtc"]
        if not aqtc.exists():
            skipped.append(f"{rel} (needs {meta['aqtc']})")
            continue
        got = measure(meta["lens"], REPO / meta["dataset"], aqtc, meta["k"])
        want = {"n_pool": meta["n_pool"], "hn_at_k": r["summary"]["hn_at_k"],
                "hitc_at_k": r["summary"]["hitc_at_k"]}
        ok = got == want
        checked += 1
        mismatched += 0 if ok else 1
        print(f"{'OK  ' if ok else 'DIFF'} {rel}  HN@{meta['k']}={got['hn_at_k']:.4f}"
              f"  HitC@{meta['k']}={got['hitc_at_k']:.4f}  n_pool={got['n_pool']}"
              + ("" if ok else f"  (stored: HN={want['hn_at_k']:.4f} HitC={want['hitc_at_k']:.4f}"
                               f" n_pool={want['n_pool']})"))
    for s in skipped:
        print(f"SKIP {s}")
    print(f"checked {checked}, mismatched {mismatched}, skipped {len(skipped)}")
    if mismatched:
        return 1
    return 0 if checked else 2


if __name__ == "__main__":
    sys.exit(main())
