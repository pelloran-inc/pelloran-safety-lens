#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Embed A/B/C triples with E5 (identical convention to embed_and_pack:
no prefix, L2-normalised, e5-large-v2 pinned revision) and splice them onto
the SAME distractor block as a source AQTC. Guarantees byte-identical
distractors, so the only variable is the triples' wording.

Writes a new AQTC v1 (dim=1024, n_cases=len(triples), n_distr from src) and
the triple embeddings as <out>.abc.npy (for the F1/F3 faithfulness gates).
"""
import argparse, json, struct, sys
from pathlib import Path
import numpy as np


def read_aqtc_distractors(path: Path):
    data = path.read_bytes()
    magic, ver, dim, n_cases, n_distr = struct.unpack_from("<IIIII", data, 0)
    assert magic == 0x43545141 and ver == 1, f"bad AQTC {path}"
    off = 20 + 4 * 3 * n_cases * dim
    distr = np.frombuffer(data, dtype="<f4", count=n_distr * dim, offset=off).reshape(n_distr, dim)
    return dim, n_distr, np.array(distr, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", type=Path, required=True)
    ap.add_argument("--src-aqtc", type=Path, required=True, help="AQTC to reuse the distractor block from")
    ap.add_argument("--out-aqtc", type=Path, required=True)
    ap.add_argument("--model", default="intfloat/e5-large-v2")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    if args.model_revision is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from _pins import model_rev
            args.model_revision = model_rev(args.model)
        except Exception:
            args.model_revision = None

    dim, n_distr, distr = read_aqtc_distractors(args.src_aqtc)
    triples = json.loads(args.triples.read_text())["test_cases"]
    texts = []
    for t in triples:
        texts += [t["clause_a"], t["clause_b"], t["clause_c"]]
    print(f"[splice] {len(triples)} triples ({len(texts)} texts); distractors "
          f"{n_distr}x{dim} from {args.src_aqtc.name}; E5 rev={args.model_revision}", file=sys.stderr)

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(args.model, device="cpu", revision=args.model_revision)
    abc = m.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    assert abc.shape[1] == dim, f"dim mismatch {abc.shape[1]} != {dim}"

    with args.out_aqtc.open("wb") as f:
        f.write(struct.pack("<IIIII", 0x43545141, 1, dim, len(triples), n_distr))
        f.write(abc.reshape(-1).astype("<f4").tobytes())
        f.write(distr.astype("<f4").tobytes())
    np.save(str(args.out_aqtc) + ".abc.npy", abc)
    print(f"[splice] wrote {args.out_aqtc} (n_cases={len(triples)}, n_distr={n_distr}) + .abc.npy", file=sys.stderr)


if __name__ == "__main__":
    main()
