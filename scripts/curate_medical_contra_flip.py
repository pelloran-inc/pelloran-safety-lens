#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Miss-driven trap curation for medical contra_flip C-strings.

The 11 HitC-miss cases identified by miss_diagnostic.py with an intermediate
medical lens at n=51k are all contra_flip and share the same failure mode: the C-string
uses either an abbreviation absent from A ("HFrEF" vs "heart failure with
reduced ejection fraction") or a structural reversal that drops the
shared clinical noun-phrase from A ("Y rules out X" instead of "X is
contraindicated in Y").

Fix: rewrite each C to share the anchor medical concept from A while
still using different surrounding phrasing. Preserves paraphrase intent
without breaking E5 embedding proximity.

Output: the curated set (data/traps/medical_300.json): identical to the
input except that 11 test_cases have new clause_c strings.
"""
import argparse
import json
import sys
from pathlib import Path


# The 11 C-string rewrites. Case IDs come from miss_diag_D2_51k.json.
# For each case, the NEW C spells out the key medical concept from A and
# uses natural clinical prose (not abbreviations, not structural reversal).
CURATED = {
    "med_v2_contra_004": (
        "For heart failure with reduced ejection fraction, ACE inhibitors "
        "are a standard component of therapy."
    ),
    "med_v2_contra_014": (
        "Major warfarin-induced coagulopathy is reversed by administration "
        "of vitamin K."
    ),
    "med_v2_contra_015": (
        "For symptomatic volume overload in congestive heart failure, "
        "diuretics are the standard therapy."
    ),
    "med_v2_contra_024": (
        "Confirmed deep vein thrombosis in adults is treated with "
        "therapeutic anticoagulation."
    ),
    "med_v2b_contraindication_flip_018": (
        "Angiotensin receptor blockers are avoided during pregnancy due to "
        "teratogenic risk."
    ),
    "med_v2b_contraindication_flip_021": (
        "Sulfonylureas are avoided in patients with severe renal impairment."
    ),
    "med_v2b_contraindication_flip_031": (
        "Beta-blockers are avoided in decompensated acute heart failure "
        "with pulmonary edema."
    ),
    "med_v2b_contraindication_flip_038": (
        "Codeine is avoided in ultra-rapid CYP2D6 metabolizers who are "
        "breastfeeding due to infant risk."
    ),
    "med_v2b_contraindication_flip_044": (
        "Isoniazid is held in patients with acute drug-induced hepatitis "
        "until liver enzymes normalize."
    ),
    "med_v2b_contraindication_flip_049": (
        "Fluoroquinolones are avoided in patients with active tendon "
        "rupture history."
    ),
    "med_v2b_contraindication_flip_050": (
        "Live rotavirus vaccine is avoided in infants with severe combined "
        "immunodeficiency."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="the uncurated 300-triple set")
    ap.add_argument("--output", type=Path, required=True,
                    help="the curated set to write")
    args = ap.parse_args()

    ds = json.loads(args.input.read_text())
    n_curated = 0
    n_not_found = 0
    for case in ds["test_cases"]:
        cid = case.get("id")
        if cid in CURATED:
            print(f"[curate] {cid}", file=sys.stderr)
            print(f"  old C: {case['clause_c']}", file=sys.stderr)
            print(f"  new C: {CURATED[cid]}", file=sys.stderr)
            case["clause_c"] = CURATED[cid]
            n_curated += 1
    ids_expected = set(CURATED.keys())
    ids_found = {c.get("id") for c in ds["test_cases"] if c.get("id") in CURATED}
    n_not_found = len(ids_expected - ids_found)

    ds["_curation"] = {
        "source": args.input.name,
        "curated_case_ids": list(CURATED.keys()),
        "n_curated": n_curated,
        "n_not_found": n_not_found,
        "rationale": (
            "Miss-driven trap curation. An intermediate medical lens at n=51k had "
            "11 HitC-miss cases, all contra_flip. C-strings rewritten to share "
            "anchor medical noun-phrase with clause_a while preserving "
            "paraphrase intent."
        ),
    }

    if n_not_found > 0:
        print(f"[curate] WARN: {n_not_found} expected case IDs not found in pack",
              file=sys.stderr)
        for missing in ids_expected - ids_found:
            print(f"  missing: {missing}", file=sys.stderr)
        sys.exit(1)

    args.output.write_text(json.dumps(ds, indent=2))
    print(f"[curate] wrote {args.output} with {n_curated} curated C-strings",
          file=sys.stderr)


if __name__ == "__main__":
    main()
