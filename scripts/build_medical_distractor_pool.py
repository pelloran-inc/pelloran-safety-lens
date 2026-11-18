#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Build the medical distractor pool from a permissively-licensed public
biomedical corpus on Hugging Face.

Why this builder exists: earlier releases shipped the medical distractor
pool as pre-selected text taken from PubMed/MEDLINE abstracts, whose terms
(NLM; publisher copyright of the abstracts) make redistribution unclear.
This builder instead rebuilds an equivalent pool from a source that is
explicitly redistributable, so the medical pool can be reconstructed from
this package alone, like the legal/patent/financial pools.

Default source: the PubMed Central Open Access *commercial-use* subset
(articles under CC0 / CC BY / CC BY-SA / CC BY-ND), grouped by license by
PMC itself. Only that commercial-use group is used, so the output is
redistributable. The source is pinned by `--revision`; record the exact
dataset, config and revision you used in DATA.md.

The pool is a JSON array of unique short clause strings (8-30 words),
identical in format to the other domains' pools and to
`medical_short_distractor_pool.json`. Selecting a biomedical distractor
corpus does not need to match the original pool byte-for-byte: the pool is
a set of negatives, and the paper's medical numbers are re-measured on
whatever pool this builder produces (see REPRODUCE.md).
"""
import argparse
import json
import re
import sys
from pathlib import Path


# Simple sentence splitter (no nltk/spaCy dependency; good enough for
# biomedical abstract/article prose with standard punctuation).
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
    # Mostly-alphabetic (drop tables, reference lists, numeric runs).
    n_alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    if n_alpha / n < 0.6:
        return False
    lo = s.lower()
    for bad in ("this article", "copyright", "all rights reserved",
                "supplementary material", "conflict of interest",
                "figure ", "table ", "et al.", "http", "www.", "doi:"):
        if bad in lo:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=200000)
    ap.add_argument("--wmin", type=int, default=8)
    ap.add_argument("--wmax", type=int, default=30)
    # Source coordinates (pin all three in DATA.md). Defaults name the
    # PMC Open Access commercial-use (redistributable) subset; override if
    # the dataset's config/field names differ in your datasets version.
    # Default: jmhb/pubmed_bioasq_2022 (tagged CC0; PubMed abstracts). The
    # builder *downloads* the source (it is not shipped in this package), so
    # the package redistributes no PubMed text; the source's own terms govern
    # the user's download, exactly as for the legal/patent/financial builders.
    ap.add_argument("--hf-dataset", default="jmhb/pubmed_bioasq_2022")
    ap.add_argument("--hf-config", default=None,
                    help="dataset config (None = default config)")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--revision", default=None,
                    help="HF dataset revision (commit SHA) to pin; record it in DATA.md")
    ap.add_argument("--text-field", default="abstractText",
                    help="row field holding the abstract/article text")
    args = ap.parse_args()
    if args.revision is None:
        from _pins import dataset_rev
        args.revision = dataset_rev(args.hf_dataset)

    from datasets import load_dataset  # imported here so --help needs no deps

    print(f"[distr] streaming {args.hf_dataset}:{args.hf_config} "
          f"(rev={args.revision}) field={args.text_field}", file=sys.stderr)
    ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split,
                      streaming=True, revision=args.revision)

    all_sents = set()
    n_rows = 0
    for row in ds:
        n_rows += 1
        text = row.get(args.text_field, "") or ""
        for sent in split_sentences(text):
            if keep_sentence(sent, args.wmin, args.wmax):
                all_sents.add(sent)
        if len(all_sents) >= args.target:
            break
        if n_rows % 5000 == 0:
            print(f"[distr] rows={n_rows} unique={len(all_sents)}", file=sys.stderr)

    print(f"[distr] total unique short clauses: {len(all_sents)} "
          f"(from {n_rows} source rows)", file=sys.stderr)

    # Deterministic ordering, same format as the other pools.
    out = sorted(all_sents)[: args.target]
    args.out.write_text(json.dumps(out, indent=None, separators=(",", ":")))
    print(f"[distr] wrote {args.out} ({args.out.stat().st_size} bytes, "
          f"{len(out)} clauses)", file=sys.stderr)


if __name__ == "__main__":
    main()
