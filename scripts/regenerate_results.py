#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Re-measure and REWRITE every stored result against a fresh set of pools.

This is the write-side companion to `check_results.py`. `check_results.py`
re-measures and *compares*; this script re-measures and *overwrites* the
stored result JSONs, so the package can be re-baselined after the pools are
rebuilt from pinned sources (new library versions change the E5 embeddings,
and therefore the exact-search HitC@10 at the margin; HN@10 stays 0).

For every `results/**/*.json` it runs `miss_diagnostic.py` with the lens
(or --raw) and trap set named in the result's own meta block, on the AQTC
binary of the same name in --aqtc-dir, and writes the new measurement back
to the result file. The relative `lens`/`dataset`/`aqtc` fields in meta are
preserved verbatim (only the measured numbers change).

Usage:
    python3 scripts/regenerate_results.py --aqtc-dir DIR [--only SUBSTRING]
Then recompute the hashes:
    (cd DIR && sha256sum *.aqtc > /path/to/results/POOLS_SHA256SUMS)   # pool hashes
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MISS = REPO / "scripts" / "miss_diagnostic.py"
RAW = "none (raw cosine)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aqtc-dir", type=Path, required=True)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    done = skipped = 0
    for res in sorted((REPO / "results").rglob("*.json")):
        rel = res.relative_to(REPO)
        if args.only and args.only not in str(rel):
            continue
        meta = json.loads(res.read_text())["meta"]
        aqtc = args.aqtc_dir / meta["aqtc"]
        if not aqtc.exists():
            print(f"SKIP {rel} (needs {meta['aqtc']})")
            skipped += 1
            continue
        which = ["--raw"] if meta["lens"] == RAW else ["--lens", str(REPO / meta["lens"])]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.json"
            subprocess.run(
                [sys.executable, str(MISS), *which,
                 "--dataset", str(REPO / meta["dataset"]),
                 "--aqtc", str(aqtc), "--out", str(out), "--k", str(meta["k"])],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            new = json.loads(out.read_text())
        # Preserve the stored relative-path meta format; keep the new numbers.
        new["meta"]["lens"] = meta["lens"]
        new["meta"]["dataset"] = meta["dataset"]
        new["meta"]["aqtc"] = meta["aqtc"]
        res.write_text(json.dumps(new, indent=2))
        s = new["summary"]
        print(f"OK   {rel}  HN@{meta['k']}={s['hn_at_k']:.4f}  HitC@{meta['k']}={s['hitc_at_k']:.4f}"
              f"  n_pool={new['meta']['n_pool']}")
        done += 1
    print(f"regenerated {done}, skipped {skipped}")
    return 0 if done else 2


if __name__ == "__main__":
    sys.exit(main())
