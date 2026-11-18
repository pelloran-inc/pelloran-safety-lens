#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Convert a 'dataset pack' (manifest.json / texts.jsonl) into the plain
JSON array of clause strings that `embed_and_pack.py --distractor-pool`
expects.

The legal distractor builder (`build_legal_distractor_pool.py`) writes a
pack: `manifest.json` ([{id, text, source, meta}, ...]) and `texts.jsonl`
({"id","text","source",...} per line). `embed_and_pack.py`, like the
patent/medical/financial pools, consumes a flat JSON list of strings. This
script is the intermediate step between the two, so the legal pool can be
built end to end without a manual conversion.

Usage:
  # from the legal builder's out-dir (uses texts.jsonl, else manifest.json)
  python3 scripts/manifest_to_distractor_pool.py \
      --in tmp/legal_distractors/ledgar100k --out legal_pool_100k.json
"""
import argparse
import json
import sys
from pathlib import Path


def load_texts(src: Path, text_field: str):
    """Yield the text of each row from a pack dir, a texts.jsonl, or a
    manifest.json. Accepts a directory or a file."""
    if src.is_dir():
        jsonl = src / "texts.jsonl"
        manifest = src / "manifest.json"
        if jsonl.exists():
            src = jsonl
        elif manifest.exists():
            src = manifest
        else:
            sys.exit(f"no texts.jsonl or manifest.json in {src}")
    if src.suffix == ".jsonl":
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)[text_field]
    else:  # a JSON array of row objects, or already a list of strings
        data = json.loads(src.read_text(encoding="utf-8"))
        for row in data:
            yield row if isinstance(row, str) else row[text_field]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, nargs="+", required=True,
                    help="one or more pack dirs, texts.jsonl, or manifest.json (merged + deduped)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output JSON array of clause strings")
    ap.add_argument("--text-field", default="text")
    args = ap.parse_args()

    # Deterministic, deduplicated, same format as the other pools. Merges all
    # inputs into one set.
    merged = set()
    for src in args.inp:
        merged.update(t for t in load_texts(src, args.text_field) if t and t.strip())
    clauses = sorted(merged)
    args.out.write_text(json.dumps(clauses, indent=None, separators=(",", ":")))
    print(f"[pack] wrote {args.out} ({len(clauses)} unique clauses, "
          f"{args.out.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
