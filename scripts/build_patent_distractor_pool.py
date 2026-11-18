#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Build the patent distractor pool from the Hugging Face
`big_patent` dataset.

Fetches patent abstracts across the nine BIGPATENT categories (a-i,
covering different USPTO arts), sentence-splits them, filters for
matched shape (8-30 words), deduplicates, and saves as a JSON array
of clause strings.

Target: ≥ 200k unique short clauses to support the ≥ 100k-pool
scale-out measurement with dedup headroom.
"""
import argparse
import json
import re
import sys
from pathlib import Path


# Simple sentence splitter (avoids nltk / spaCy dependency; good enough
# for patent abstracts which use standard punctuation).
_SENT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str):
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RX.split(text) if s.strip()]


def keep_sentence(s: str, wmin: int, wmax: int) -> bool:
    words = s.split()
    n = len(words)
    if n < wmin or n > wmax:
        return False
    # Filter out sentences that are mostly numbers / references
    n_alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    if n_alpha / n < 0.6:
        return False
    # Filter out obvious boilerplate
    lo = s.lower()
    for bad in ("cross-reference", "the present application claims",
                "priority claim", "background of the invention",
                "field of the invention", "[00", "hereinafter"):
        if bad in lo:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=200000)
    ap.add_argument("--wmin", type=int, default=8)
    ap.add_argument("--wmax", type=int, default=30)
    ap.add_argument("--categories", default="a,b,c,d,e,f,g,h,i",
                    help="BIGPATENT categories to stream from")
    ap.add_argument("--per-cat-cap", type=int, default=25000,
                    help="Max sentences per category (for diversity)")
    args = ap.parse_args()
    from datasets import load_dataset  # imported here, so that --help works without it
    from _pins import dataset_rev

    categories = args.categories.split(",")
    print(f"[distr] streaming BIGPATENT categories: {categories}", file=sys.stderr)

    all_sents = set()
    per_cat_counts = {}

    for cat in categories:
        cat = cat.strip()
        try:
            ds = load_dataset("big_patent", cat, split="train", streaming=True,
                              revision=dataset_rev("big_patent"))
        except Exception as e:
            print(f"[distr] WARN: category {cat} failed to load: {e}",
                  file=sys.stderr)
            continue
        n_kept_this_cat = 0
        for row in ds:
            abstract = row.get("abstract", "") or ""
            for sent in split_sentences(abstract):
                if keep_sentence(sent, args.wmin, args.wmax):
                    all_sents.add(sent)
                    n_kept_this_cat += 1
                    if n_kept_this_cat >= args.per_cat_cap:
                        break
            if n_kept_this_cat >= args.per_cat_cap:
                break
            if len(all_sents) >= args.target:
                break
        per_cat_counts[cat] = n_kept_this_cat
        print(f"[distr] category {cat}: {n_kept_this_cat} kept (total unique: {len(all_sents)})",
              file=sys.stderr)
        if len(all_sents) >= args.target:
            break

    print(f"[distr] total unique short clauses: {len(all_sents)}",
          file=sys.stderr)
    print(f"[distr] per-category counts: {per_cat_counts}", file=sys.stderr)

    # Save as JSON array of strings (matches medical_short_distractor_pool.json)
    out = sorted(all_sents)  # deterministic ordering
    args.out.write_text(json.dumps(out, indent=None, separators=(",", ":")))
    print(f"[distr] wrote {args.out} ({args.out.stat().st_size} bytes)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
