#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Build the financial distractor pool from five Hugging Face
financial NLP datasets, combined and sentence-split.

Sources (parquet-based):
  nickmuchi/financial-classification         — financial news snippets
  zeroshot/twitter-financial-news-sentiment  — financial-news tweets
  zeroshot/twitter-financial-news-topic      — financial-news tweets by topic
  ChanceFocus/flare-fpb                      — Financial PhraseBank via FLARE
  kdave/Indian_Financial_News                — financial news articles

Filters to 8-30 word matched-shape short clauses. Deduplicates.
Target: as many unique clauses as possible. The output is sorted.
"""
import argparse
import json
import re
import sys
from pathlib import Path


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
    n_alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    if n / n_alpha > 2.0 if n_alpha else True:
        return False
    lo = s.lower()
    # Filter out obvious URLs, tweet handles etc.
    if "http://" in lo or "https://" in lo or lo.startswith("$"):
        return False
    return True


def stream_text_field(name, split, text_field, cap):
    from datasets import load_dataset  # imported here, so that --help works without it
    from _pins import dataset_rev
    try:
        d = load_dataset(name, split=split, streaming=True, revision=dataset_rev(name))
    except Exception as e:
        print(f"[distr] FAIL loading {name}: {str(e)[:100]}", file=sys.stderr)
        return
    n = 0
    for row in d:
        t = row.get(text_field, "") if isinstance(row, dict) else None
        if not t:
            continue
        yield t
        n += 1
        if n >= cap:
            break
    print(f"[distr]   pulled {n} rows from {name} ({split})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wmin", type=int, default=8)
    ap.add_argument("--wmax", type=int, default=30)
    ap.add_argument("--per-source-cap", type=int, default=50000)
    args = ap.parse_args()

    sources = [
        # (dataset_name, split, text_field)
        ("nickmuchi/financial-classification", "train", "text"),
        ("zeroshot/twitter-financial-news-sentiment", "train", "text"),
        ("zeroshot/twitter-financial-news-topic", "train", "text"),
        ("ChanceFocus/flare-fpb", "train", "text"),
        # Bigger source: Indian financial news, long articles sentence-split for volume
        ("kdave/Indian_Financial_News", "train", "Content"),
    ]

    all_sents = set()
    for name, split, text_field in sources:
        print(f"[distr] streaming {name} split={split}", file=sys.stderr)
        for text in stream_text_field(name, split, text_field, args.per_source_cap):
            for sent in split_sentences(text):
                if keep_sentence(sent, args.wmin, args.wmax):
                    all_sents.add(sent)
        print(f"[distr]   total unique so far: {len(all_sents)}", file=sys.stderr)

    print(f"[distr] final unique count: {len(all_sents)}", file=sys.stderr)
    out = sorted(all_sents)
    args.out.write_text(json.dumps(out, indent=None, separators=(",", ":")))
    print(f"[distr] wrote {args.out} ({args.out.stat().st_size} bytes)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
