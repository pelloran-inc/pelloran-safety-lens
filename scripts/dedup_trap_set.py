#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""De-duplicate a trap set and rebuild its AQTC binary.

Reads the 384-case dataset JSON + its aligned AQTC binary. Groups rows
by SHA-256 of (clause_a, clause_b, clause_c) triple. Keeps first
appearance of each unique triple. Emits:

  - duplicate_audit.json — per-key row-idx groupings
  - pack_v09_plus_dedup.json — 299 unique test_cases
  - pack_v09_plus_dedup_embeddings.e5large.json — 897-row f32 embeddings
    (each unique case × 3 for A/B/C, first-appearance order)
  - pack_v09_plus_dedup.aqtc — AQTC v1 binary: 299 × 3 f32 A/B/C rows
    followed by the original 50 000 distractor rows verbatim.

Design invariants:
  - Total row-count sanity: n_unique + n_duplicate == 384.
  - First-appearance order: row 0 of the dedup'd set is the first row
    in the input whose triple appeared for the first time.
  - Byte-identical distractor blob: the 50 000 distractor rows are
    memcpy'd from the source AQTC binary — no re-embedding, no
    reordering. Pool ordering property (rows 0..3n-1 are case A/B/C,
    3n..3n+d-1 are distractors) is preserved.

Usage:
    python3 dedup_trap_set.py \\
        --in-dataset conditional_scope_traps_v2_plus_v09.json \\
        --in-aqtc trap_corpus.aqtc \\
        --in-embeddings conditional_scope_traps_v2_plus_v09_abc.e5large.json \\
        --out-dataset pack_v09_plus_dedup.json \\
        --out-aqtc pack_v09_plus_dedup.aqtc \\
        --out-embeddings pack_v09_plus_dedup_embeddings.e5large.json \\
        --out-audit duplicate_audit.json
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np


def canonical_key(case):
    """SHA-256 of the (clause_a, clause_b, clause_c) triple."""
    payload = "\n".join([case["clause_a"], case["clause_b"], case["clause_c"]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_aqtc_header(path):
    """Read only the AQTC v1 header + return (emb_dim, n_cases, n_distr,
    header_size)."""
    data = path.read_bytes()
    magic, version, emb_dim, n_cases, n_distr = struct.unpack_from("<IIIII", data, 0)
    if magic != 0x43545141:
        raise SystemExit(f"Bad AQTC magic {magic:#x}")
    if version != 1:
        raise SystemExit(f"Unsupported AQTC version {version}")
    return emb_dim, n_cases, n_distr, data


def write_aqtc(path, emb_dim, cases_rows_flat, distractor_bytes):
    """Write AQTC v1 header + case rows + distractor bytes.

    cases_rows_flat: 1-D float32 array of length n_cases * 3 * emb_dim
    distractor_bytes: raw f32 bytes of length n_distr * emb_dim * 4
    """
    n_cases = cases_rows_flat.size // (3 * emb_dim)
    n_distr = len(distractor_bytes) // (emb_dim * 4)
    with path.open("wb") as f:
        f.write(struct.pack("<IIIII", 0x43545141, 1, emb_dim, n_cases, n_distr))
        f.write(cases_rows_flat.tobytes())
        f.write(distractor_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dataset", type=Path, required=True)
    ap.add_argument("--in-aqtc", type=Path, required=True)
    ap.add_argument("--in-embeddings", type=Path, required=True)
    ap.add_argument("--out-dataset", type=Path, required=True)
    ap.add_argument("--out-aqtc", type=Path, required=True)
    ap.add_argument("--out-embeddings", type=Path, required=True)
    ap.add_argument("--out-audit", type=Path, required=True)
    args = ap.parse_args()

    # Load source dataset
    ds = json.loads(args.in_dataset.read_text())
    cases = ds["test_cases"]
    n_input = len(cases)
    print(f"[dedup] input dataset: {n_input} cases", file=sys.stderr)

    # Group rows by canonical key
    key_to_rows = {}
    first_row = {}
    for i, c in enumerate(cases):
        k = canonical_key(c)
        if k not in key_to_rows:
            first_row[k] = i
            key_to_rows[k] = [i]
        else:
            key_to_rows[k].append(i)

    n_unique = len(key_to_rows)
    n_duplicate = n_input - n_unique
    assert n_unique + n_duplicate == n_input, "dedup arithmetic broken"
    print(f"[dedup] unique triples: {n_unique}", file=sys.stderr)
    print(f"[dedup] duplicate rows removed: {n_duplicate}", file=sys.stderr)

    duplicated_keys = [k for k, rows in key_to_rows.items() if len(rows) > 1]
    print(f"[dedup] distinct triples that were duplicated: {len(duplicated_keys)}",
          file=sys.stderr)

    # Emit audit
    audit = {
        "input_row_count": n_input,
        "unique_triples": n_unique,
        "duplicate_rows_removed": n_duplicate,
        "duplicated_distinct_triples": len(duplicated_keys),
        "groups": [
            {
                "canonical_key": k,
                "first_row_idx": first_row[k],
                "n_instances": len(key_to_rows[k]),
                "all_row_idxs": key_to_rows[k],
                "case_id": cases[first_row[k]]["id"],
                "trap_type": cases[first_row[k]]["trap_type"],
            }
            for k in key_to_rows
        ],
    }
    args.out_audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    # Preserve first-appearance ordering
    ordered_keys = sorted(key_to_rows.keys(), key=lambda k: first_row[k])
    kept_row_idxs = [first_row[k] for k in ordered_keys]

    # Emit dedup'd dataset JSON
    new_cases = []
    for kept_row_idx in kept_row_idxs:
        c = dict(cases[kept_row_idx])
        c["_dedup_original_row_idx"] = kept_row_idx
        c["_dedup_all_instances_row_idxs"] = key_to_rows[canonical_key(cases[kept_row_idx])]
        new_cases.append(c)
    out_ds = {
        "test_cases": new_cases,
        "_dedup": True,
        "_source_dataset_sha256": hashlib.sha256(args.in_dataset.read_bytes()).hexdigest(),
        "_n_input_cases": n_input,
        "_n_unique_triples": n_unique,
        "_n_duplicate_rows_removed": n_duplicate,
    }
    args.out_dataset.write_text(json.dumps(out_ds, ensure_ascii=False, indent=2))
    print(f"[dedup] wrote dedup'd dataset: {args.out_dataset}", file=sys.stderr)

    # Load source embeddings JSON + emit dedup'd version
    src_emb = np.asarray(json.loads(args.in_embeddings.read_text()), dtype=np.float32)
    if src_emb.shape[0] != 3 * n_input:
        raise SystemExit(
            f"embeddings shape {src_emb.shape} does not match 3 × {n_input}"
        )
    d = src_emb.shape[1]

    dedup_emb_rows = []
    for kept_row_idx in kept_row_idxs:
        base = 3 * kept_row_idx
        dedup_emb_rows.extend([src_emb[base], src_emb[base + 1], src_emb[base + 2]])
    dedup_emb = np.stack(dedup_emb_rows, axis=0)
    assert dedup_emb.shape == (3 * n_unique, d)
    args.out_embeddings.write_text(json.dumps(dedup_emb.tolist()))
    print(f"[dedup] wrote dedup'd embeddings: {args.out_embeddings} "
          f"({dedup_emb.shape})", file=sys.stderr)

    # Rebuild AQTC binary — first read source header + distractor bytes
    src_emb_dim, src_n_cases, src_n_distr, src_aqtc_data = read_aqtc_header(args.in_aqtc)
    assert src_emb_dim == d, f"AQTC dim {src_emb_dim} != embeddings dim {d}"
    assert src_n_cases == n_input, f"AQTC n_cases {src_n_cases} != dataset {n_input}"
    header_size = 20
    triplet_bytes_size = src_n_cases * 3 * src_emb_dim * 4
    distractor_bytes = src_aqtc_data[header_size + triplet_bytes_size:]
    expected_distr_size = src_n_distr * src_emb_dim * 4
    assert len(distractor_bytes) == expected_distr_size, (
        f"distractor byte count {len(distractor_bytes)} != "
        f"expected {expected_distr_size}"
    )

    # Reorder the case A/B/C rows AS f32 from source
    src_case_f32 = np.frombuffer(
        src_aqtc_data,
        dtype="<f4",
        count=src_n_cases * 3 * src_emb_dim,
        offset=header_size,
    ).reshape(src_n_cases * 3, src_emb_dim)

    dedup_case_rows = np.empty((n_unique * 3, src_emb_dim), dtype="<f4")
    for out_case_i, kept_row_idx in enumerate(kept_row_idxs):
        src_base = 3 * kept_row_idx
        out_base = 3 * out_case_i
        dedup_case_rows[out_base] = src_case_f32[src_base]
        dedup_case_rows[out_base + 1] = src_case_f32[src_base + 1]
        dedup_case_rows[out_base + 2] = src_case_f32[src_base + 2]

    # Consistency check: dedup_case_rows[i] should match dedup_emb[i] to
    # within numerical noise (they came from different sources but should be
    # the same embeddings).
    max_diff = float(np.abs(dedup_case_rows - dedup_emb).max())
    print(f"[dedup] AQTC-vs-JSON embedding max abs diff: {max_diff:.2e}",
          file=sys.stderr)
    assert max_diff < 1e-3, (
        f"AQTC and JSON embeddings disagree by max {max_diff} > 1e-3"
    )

    write_aqtc(args.out_aqtc, src_emb_dim, dedup_case_rows.flatten(), distractor_bytes)
    print(f"[dedup] wrote AQTC binary: {args.out_aqtc} "
          f"(n_cases={n_unique}, n_distr={src_n_distr}, "
          f"pool_size={n_unique * 3 + src_n_distr})", file=sys.stderr)


if __name__ == "__main__":
    main()
