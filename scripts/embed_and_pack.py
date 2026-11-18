#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""E5-Large embedding of a trap set and its distractor pool, and AQTC
binary emission.

Consumes:
  - a trap set JSON (data/traps/<domain>_*.json; N A/B/C triples)
  - a distractor pool JSON (a list of clause strings)

Emits, named after --name (e.g. "financial"):
  - <name>_abc.e5large.json   (N*3 A/B/C embeddings)
  - <name>_5k.aqtc / _50k.aqtc / _100k.aqtc  (AQTC v1 binaries)
  - <name>_distractor_emb.npy (the distractor embeddings)

E5 convention (the same for every domain):
  - No prefix ("passage:" / "query:" stripping)
  - L2-normalize embeddings (normalize_embeddings=True)
  - 1024-D output
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def write_aqtc(path: Path, triplets_flat: np.ndarray, distractors: np.ndarray,
               emb_dim: int, n_cases: int):
    """AQTC v1: 20-byte header + interleaved A/B/C + distractors.

    Header (u32 LE): magic 'AQTC' (0x43545141), version 1, embedding_dim,
    n_cases, n_distractors.
    """
    n_distr = distractors.shape[0]
    with path.open("wb") as f:
        f.write(struct.pack("<IIIII", 0x43545141, 1, emb_dim, n_cases, n_distr))
        f.write(triplets_flat.astype("<f4").tobytes())
        f.write(distractors.astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", type=Path, required=True)
    ap.add_argument("--distractor-pool", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", required=True,
                    help="domain name for the output files, e.g. financial")
    ap.add_argument("--model", default="intfloat/e5-large-v2")
    ap.add_argument("--model-revision", default=None,
                    help="HF model revision (commit SHA) to pin for byte-exact "
                         "embeddings; record the value you used in DATA.md")
    ap.add_argument("--pool-sizes", type=int, nargs="+",
                    default=[5000, 50000, 100000])
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    if args.model_revision is None:
        from _pins import model_rev
        args.model_revision = model_rev(args.model)
    from sentence_transformers import SentenceTransformer  # imported here, so that --help works without it

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[embed_medical] loading triples {args.triples}", file=sys.stderr)
    ds = json.loads(args.triples.read_text())
    triples = ds["test_cases"]
    n_cases = len(triples)
    print(f"[embed_medical]   {n_cases} A/B/C triples", file=sys.stderr)

    # Interleave A/B/C texts for E5 batch encoding
    abc_texts = []
    for t in triples:
        abc_texts.append(t["clause_a"])
        abc_texts.append(t["clause_b"])
        abc_texts.append(t["clause_c"])
    print(f"[embed_medical]   {len(abc_texts)} A/B/C texts total", file=sys.stderr)

    print(f"[embed_medical] loading distractor pool {args.distractor_pool}",
          file=sys.stderr)
    distr_texts = json.loads(args.distractor_pool.read_text())
    print(f"[embed_medical]   {len(distr_texts)} distractor paragraphs",
          file=sys.stderr)

    max_needed = max(args.pool_sizes)
    if len(distr_texts) < max_needed:
        print(f"[embed_medical] WARN: pool has {len(distr_texts)} paragraphs, "
              f"need {max_needed} for largest AQTC", file=sys.stderr)
    distr_texts = distr_texts[:max_needed]

    print(f"[embed_medical] loading E5 model {args.model} (rev={args.model_revision})",
          file=sys.stderr)
    m = SentenceTransformer(args.model, device="cpu", revision=args.model_revision)

    print(f"[embed_medical] embedding {len(abc_texts)} A/B/C texts ...",
          file=sys.stderr)
    abc_emb = m.encode(
        abc_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    print(f"[embed_medical]   abc_emb shape: {abc_emb.shape}", file=sys.stderr)

    d = abc_emb.shape[1]

    # Emit E5-JSON (for the trainer that reads embeddings JSON)
    abc_json_path = args.out_dir / f"{args.name}_abc.e5large.json"
    abc_json_path.write_text(json.dumps(abc_emb.tolist()))
    print(f"[embed_medical] wrote {abc_json_path}", file=sys.stderr)

    print(f"[embed_medical] embedding {len(distr_texts)} distractors ...",
          file=sys.stderr)
    distr_emb = m.encode(
        distr_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    print(f"[embed_medical]   distr_emb shape: {distr_emb.shape}", file=sys.stderr)

    # Build AQTC binaries at each pool size
    triplets_flat = abc_emb.reshape(-1)  # (n_cases * 3 * d,) contiguous

    for pool_size in args.pool_sizes:
        pool_size = min(pool_size, distr_emb.shape[0])
        out_aqtc = args.out_dir / f"{args.name}_{pool_size // 1000}k.aqtc"
        write_aqtc(
            out_aqtc,
            triplets_flat,
            distr_emb[:pool_size],
            emb_dim=d,
            n_cases=n_cases,
        )
        actual_pool = 3 * n_cases + pool_size
        print(f"[embed_medical]   wrote {out_aqtc} "
              f"(n_cases={n_cases}, n_distr={pool_size}, pool={actual_pool})",
              file=sys.stderr)

    # Also emit the raw distractor embeddings for dedup script consumption
    distr_npy_path = args.out_dir / f"{args.name}_distractor_emb.npy"
    np.save(distr_npy_path, distr_emb)
    print(f"[embed_medical] wrote {distr_npy_path}", file=sys.stderr)

    print("[embed_medical] done", file=sys.stderr)


if __name__ == "__main__":
    main()
