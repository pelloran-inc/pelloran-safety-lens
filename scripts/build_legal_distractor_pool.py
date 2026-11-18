#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Legal distractor pool builder (deterministic) for Steering Lens evaluation.

Goal:
- Produce a realistic distractor corpus (5k / 50k / 100k) of legal/contract-like clauses.
- Output in the same "dataset pack" style as other quality tools (manifest + jsonl).

Sources (HF datasets, no secrets):
- lex_glue/ledgar (contract provisions)
- lex_glue/unfair_tos (terms of service clauses)
- urialon/converted_contract_nli (contract NLI converted; extract the clause text)
- DerivedFunction/sec-filings-market-risk-and-derivatives (EDGAR-style snippets)

Outputs (out_dir):
- manifest.json: [{id, text, source, meta}]
- texts.jsonl:  {"id","text","source",...}
- source_info.json: dataset identifiers + selection settings (seed, n, split)

Run:
  python3 scripts/build_legal_distractor_pool.py --out-dir tmp/legal_distractors/ledgar50k --source lexglue --n 50000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import re

from _pins import dataset_rev


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _clean_text(t: str) -> str:
    # Keep deterministic, minimal normalization: collapse whitespace.
    return " ".join(str(t).replace("\r", "\n").split())


@dataclass(frozen=True)
class Item:
    text: str
    source: str
    meta: Dict[str, str]


def iter_lexglue() -> Iterator[Item]:
    from datasets import load_dataset

    # Two strong legal-ish sources: contract clauses + ToS clauses
    for cfg, label_key in [("ledgar", "label"), ("unfair_tos", "labels")]:
        ds = load_dataset("lex_glue", cfg, split="train", revision=dataset_rev("lex_glue"))
        for i in range(len(ds)):
            t = _clean_text(ds[i]["text"])
            if len(t) < 40:
                continue
            meta = {"hf_dataset": "lex_glue", "config": cfg}
            # keep label if present (diagnostic only)
            try:
                meta["label"] = str(ds[i].get(label_key))
            except Exception:
                pass
            yield Item(text=t, source=f"lex_glue/{cfg}", meta=meta)


def _extract_contract_nli_clause(inp: str) -> str:
    """
    urialon/converted_contract_nli rows look like:
      "Q: <question>\nText: <clause>"
    We'll extract the clause part when possible.
    """
    s = str(inp)
    # common markers
    for marker in ["Text:", "\nText:", "\nText"]:
        if marker in s:
            parts = s.split(marker, 1)
            if len(parts) == 2:
                tail = parts[1].strip()
                if tail:
                    return _clean_text(tail)
    return _clean_text(s)


def _split_sentences(text: str) -> List[str]:
    """
    Deterministic, lightweight sentence splitter (no ML, no locale dependency).
    Good enough for building chunk distractors.
    """
    t = _clean_text(text)
    if not t:
        return []
    # Split on common sentence boundaries while keeping determinism.
    parts = re.split(r"(?<=[.!?;:])\s+", t)
    out = [p.strip() for p in parts if p and p.strip()]
    return out


def _window_chunks(sentences: List[str], window_size: int) -> List[str]:
    window_size = max(1, int(window_size))
    if not sentences:
        return []
    if window_size == 1:
        return sentences
    out: List[str] = []
    for i in range(0, len(sentences)):
        chunk = " ".join(sentences[i : i + window_size]).strip()
        if chunk:
            out.append(chunk)
    return out


def iter_contract_nli_chunks(window_size: int, splits: List[str]) -> Iterator[Item]:
    """
    Build a larger contract-style distractor pool by chunking ContractNLI clauses into sentence windows.
    This is needed because the raw dataset has ~10k rows total, which is smaller than 50k.
    """
    from datasets import load_dataset

    for split in splits:
        ds = load_dataset("urialon/converted_contract_nli", split=split)
        for i in range(len(ds)):
            clause = _extract_contract_nli_clause(ds[i]["input"])
            if len(clause) < 40:
                continue
            sents = _split_sentences(clause)
            chunks = _window_chunks(sents, window_size=window_size)
            for c in chunks:
                if len(c) < 40:
                    continue
                yield Item(
                    text=c,
                    source="urialon/converted_contract_nli",
                    meta={
                        "hf_dataset": "urialon/converted_contract_nli",
                        "split": str(split),
                        "chunk_mode": "sentence_window",
                        "window_size": str(int(window_size)),
                    },
                )


def iter_contract_nli() -> Iterator[Item]:
    from datasets import load_dataset

    # Keep for backwards-compat (single clause per row, train only).
    ds = load_dataset("urialon/converted_contract_nli", split="train")
    for i in range(len(ds)):
        clause = _extract_contract_nli_clause(ds[i]["input"])
        if len(clause) < 40:
            continue
        yield Item(text=clause, source="urialon/converted_contract_nli", meta={"hf_dataset": "urialon/converted_contract_nli", "split": "train"})


def iter_edgar() -> Iterator[Item]:
    from datasets import load_dataset

    # EDGAR 10-K filings (SEC EDGAR; U.S.-government public-domain documents).
    # Replaces DerivedFunction/sec-filings-market-risk-and-derivatives, which
    # was removed from the Hub. We draw sentence-length clauses from the
    # legal/contractual sections (business, risk factors, legal proceedings,
    # MD&A) and shape-filter them to 8-30 words, matching the other pools.
    try:
        ds = load_dataset("eloukas/edgar-corpus", "full", split="train",
                          streaming=True, trust_remote_code=True,
                          revision=dataset_rev("eloukas/edgar-corpus"))
    except Exception:
        ds = load_dataset("eloukas/edgar-corpus", split="train",
                          streaming=True, trust_remote_code=True,
                          revision=dataset_rev("eloukas/edgar-corpus"))
    sections = ["section_1A", "section_3", "section_7", "section_1"]
    for row in ds:
        for sec in sections:
            text = row.get(sec) or ""
            for sent in _split_sentences(text):
                wc = len(sent.split())
                if 8 <= wc <= 30 and len(sent) >= 40:
                    yield Item(
                        text=sent,
                        source="eloukas/edgar-corpus",
                        meta={"hf_dataset": "eloukas/edgar-corpus", "section": sec},
                    )


def take_deterministic(it: Iterable[Item], n: int) -> List[Item]:
    out: List[Item] = []
    for x in it:
        out.append(x)
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", required=True, choices=["lexglue", "contractnli", "edgar"])
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--chunk-mode", default="none", choices=["none", "sentence_window"], help="Optional chunking mode (primarily for contractnli).")
    ap.add_argument("--window-size", type=int, default=3, help="Sentence window size for chunking (if enabled).")
    ap.add_argument("--contractnli-splits", default="train,validation,test", help="Comma-separated splits for contractnli (if chunking is enabled).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(args.n)

    if args.source == "lexglue":
        items = take_deterministic(iter_lexglue(), n)
    elif args.source == "contractnli":
        if str(args.chunk_mode) == "sentence_window":
            splits = [s.strip() for s in str(args.contractnli_splits).split(",") if s.strip()]
            items = take_deterministic(iter_contract_nli_chunks(window_size=int(args.window_size), splits=splits), n)
        else:
            items = take_deterministic(iter_contract_nli(), n)
    elif args.source == "edgar":
        items = take_deterministic(iter_edgar(), n)
    else:
        raise SystemExit("unknown source")

    manifest = []
    jsonl_path = out_dir / "texts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as w:
        for idx, it in enumerate(items):
            # stable id based on content hash, avoid duplicates across runs
            hid = sha256_text(it.text)[:16]
            rid = f"{args.source}:{idx:07d}:{hid}"
            row = {"id": rid, "text": it.text, "source": it.source, "meta": it.meta}
            manifest.append(row)
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "source_info.json").write_text(
        json.dumps(
            {
                "source": args.source,
                "n": len(items),
                "notes": "Deterministic first-N extraction; minimal whitespace normalization only.",
                "chunk_mode": str(args.chunk_mode),
                "window_size": int(args.window_size),
                "contractnli_splits": str(args.contractnli_splits),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[ok] wrote {out_dir}/manifest.json and texts.jsonl (n={len(items)})", file=sys.stderr)


if __name__ == "__main__":
    main()

