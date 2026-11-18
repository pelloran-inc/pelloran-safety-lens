#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Preflight duplicate check for steering-lens training pipelines.

Runs before any lens trainer. Exits non-zero if the input embeddings JSON has ≥ 5 % of
its case rows byte-identical to another row. Duplicate rows silently
inflate the reported HitC@10 metric: in an early legal trap set, 22 %
duplicates overstated HitC@10 by 16 percentage points (paper §3.4). Catch them here,
before wasting a training run + shipping cycle.

Domain-agnostic — operates purely on the (A, B, C)-triplet embedding
bytes. No dataset-schema knowledge, no field-name assumptions, no
synonym dictionaries. Works on any triplet dataset (medical trap,
patent trap, compliance trap, …) that follows the interleaved
A/B/C-per-case convention.

Bypass:
    --allow-duplicates    (CLI flag)
    PELLORAN_SAFETY_LENS_ALLOW_DUPLICATES=1   (env var)

Exit codes:
    0    Dataset is clean OR bypass explicitly requested
    1    Dataset has ≥ threshold duplicates and no bypass
    2    Input error (bad file, malformed JSON, etc.)

Usage (as pipeline gate):
    python3 preflight_dedup_check.py \\
        --embeddings my_embeddings.e5large.json \\
        --dataset my_dataset.json \\
        --emit-audit dedup_audit.json \\
        && python3 steering_lens_train.py --embeddings … --dataset …

"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

DEFAULT_DUPLICATE_THRESHOLD_PCT = 5.0
ALLOW_DUPLICATES_ENV = "PELLORAN_SAFETY_LENS_ALLOW_DUPLICATES"

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_INPUT_ERROR = 2


def sha256_row_group(rows):
    """SHA-256 of a list of f32 lists, concatenated as raw
    little-endian bytes."""
    h = hashlib.sha256()
    for row in rows:
        for x in row:
            h.update(struct_pack_f32_le(x))
    return h.digest()


def struct_pack_f32_le(x):
    import struct
    return struct.pack("<f", float(x))


def audit_duplicates(embeddings):
    """Given a list-of-list f32 embeddings (3n rows), return an audit
    dict describing byte-identical duplicate cases.

    Returns:
        {
            "n_input_cases": int,
            "n_unique_cases": int,
            "n_duplicate_rows": int,
            "duplicate_pct": float,
            "distinct_duplicated_groups": int,
            "top_group_size": int,
        }
    """
    n_rows = len(embeddings)
    if n_rows % 3 != 0:
        raise ValueError(
            f"embeddings row count {n_rows} not divisible by 3 "
            "(expected interleaved A/B/C per case)"
        )
    n_cases = n_rows // 3

    group_sizes = {}
    for i in range(n_cases):
        base = 3 * i
        key = sha256_row_group(embeddings[base:base + 3])
        group_sizes[key] = group_sizes.get(key, 0) + 1

    n_unique = len(group_sizes)
    n_duplicate_rows = sum(c - 1 for c in group_sizes.values())
    duplicate_pct = (100.0 * n_duplicate_rows / n_cases) if n_cases > 0 else 0.0
    distinct_duplicated_groups = sum(1 for c in group_sizes.values() if c > 1)
    top_group_size = max(group_sizes.values(), default=0)

    return {
        "n_input_cases": n_cases,
        "n_unique_cases": n_unique,
        "n_duplicate_rows": n_duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "distinct_duplicated_groups": distinct_duplicated_groups,
        "top_group_size": top_group_size,
    }


def bypass_active(cli_flag):
    if cli_flag:
        return True
    return os.environ.get(ALLOW_DUPLICATES_ENV) == "1"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--embeddings", type=Path, required=True,
                    help="Path to embeddings JSON (interleaved A/B/C per case)")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="Optional dataset JSON for row-count cross-check")
    ap.add_argument("--threshold-pct", type=float,
                    default=DEFAULT_DUPLICATE_THRESHOLD_PCT,
                    help=f"Refuse if duplicate_pct >= this (default: {DEFAULT_DUPLICATE_THRESHOLD_PCT})")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="Bypass the refusal (equivalent to PELLORAN_SAFETY_LENS_ALLOW_DUPLICATES=1)")
    ap.add_argument("--emit-audit", type=Path, default=None,
                    help="If set, write the audit JSON here regardless of pass/fail")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if not args.embeddings.exists():
        print(f"[preflight] ERROR: --embeddings not found: {args.embeddings}",
              file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        embeddings = json.loads(args.embeddings.read_text())
    except json.JSONDecodeError as e:
        print(f"[preflight] ERROR: --embeddings parse error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if not isinstance(embeddings, list):
        print(f"[preflight] ERROR: --embeddings must be a JSON array, "
              f"got {type(embeddings).__name__}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if args.dataset:
        try:
            ds = json.loads(args.dataset.read_text())
            n_ds_cases = len(ds.get("test_cases", []))
            if n_ds_cases and n_ds_cases * 3 != len(embeddings):
                print(f"[preflight] WARN: dataset has {n_ds_cases} cases but "
                      f"embeddings has {len(embeddings)} rows "
                      f"(expected {n_ds_cases * 3})", file=sys.stderr)
        except Exception as e:
            print(f"[preflight] WARN: --dataset load failed: {e}", file=sys.stderr)

    try:
        audit = audit_duplicates(embeddings)
    except ValueError as e:
        print(f"[preflight] ERROR: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    audit["threshold_pct"] = args.threshold_pct
    audit["bypass_requested"] = bypass_active(args.allow_duplicates)

    if args.emit_audit:
        args.emit_audit.write_text(json.dumps(audit, indent=2))
        if args.verbose:
            print(f"[preflight] wrote audit to {args.emit_audit}", file=sys.stderr)

    if args.verbose:
        print(f"[preflight] {audit['n_input_cases']} cases, "
              f"{audit['n_unique_cases']} unique, "
              f"{audit['n_duplicate_rows']} duplicate rows "
              f"({audit['duplicate_pct']:.2f}%), "
              f"{audit['distinct_duplicated_groups']} duplicated groups, "
              f"top group: {audit['top_group_size']}x",
              file=sys.stderr)

    over_threshold = audit["duplicate_pct"] >= args.threshold_pct

    if over_threshold and not audit["bypass_requested"]:
        print(f"[preflight] REFUSED: {audit['n_duplicate_rows']} of "
              f"{audit['n_input_cases']} case rows are byte-identical "
              f"duplicates ({audit['duplicate_pct']:.2f}%; threshold "
              f"{args.threshold_pct:.1f}%), across "
              f"{audit['distinct_duplicated_groups']} distinct duplicated "
              f"triples (largest group: {audit['top_group_size']}× "
              f"repetition). Duplicates inflate HitC@10 by counting each "
              f"duplicated miss pattern multiple times "
              f"(in an early legal trap set, 22% duplicates overstated "
              f"HitC@10 by 16 pp). To dedup: run "
              f"scripts/dedup_trap_set.py on your dataset. To bypass "
              f"(not recommended): pass --allow-duplicates or "
              f"set PELLORAN_SAFETY_LENS_ALLOW_DUPLICATES=1.",
              file=sys.stderr)
        return EXIT_REFUSED

    if over_threshold and audit["bypass_requested"]:
        print(f"[preflight] WARNING: {audit['n_duplicate_rows']} of "
              f"{audit['n_input_cases']} case rows are duplicates "
              f"({audit['duplicate_pct']:.2f}%). Bypass requested — "
              f"proceeding. Reported metrics will be inflated.",
              file=sys.stderr)

    if args.verbose or over_threshold:
        print(f"[preflight] OK: dedup check passed "
              f"({audit['duplicate_pct']:.2f}% duplicates < "
              f"{args.threshold_pct:.1f}% threshold)"
              if not over_threshold else "[preflight] OK (bypass): proceeding",
              file=sys.stderr)

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
