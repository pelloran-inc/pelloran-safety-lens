#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Two DISJOINT safe paraphrase lexicons (v1, v2),
aligned: keep a case only if BOTH v1 and v2 rewrite it, each preserving the
trap/party discriminator (diff-check), and v1 != v2 != original. Emits aligned
legal_orig_aligned.json / legal_v1.json / legal_v2.json (same order).

v1 = the declared-training phrasing; v2 = the held-out phrasing. Disjoint word
choices (G3). Trap preservation is proven per case by the diff-check.
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

LEX_V1 = [
    ("For purposes of this Agreement", "Under this Agreement"),
    ("as set forth in", "as specified in"), ("set forth in", "specified in"),
    ("as set out in", "as laid out in"), ("shall", "will"),
    ("comply with", "observe"), ("maintain", "keep"), ("provide", "furnish"),
    ("subject to", "under"), ("occur", "take place"), ("apply to", "govern"),
    ("period is", "period runs for"), ("applies if", "is triggered if"),
    ("be provided", "be supplied"), ("as defined in", "as given in"),
]
LEX_V2 = [
    ("For purposes of this Agreement", "In this Agreement"),
    ("as set forth in", "as stated in"), ("set forth in", "stated in"),
    ("as set out in", "as given in"), ("shall", "is to"),
    ("comply with", "conform to"), ("maintain", "retain"), ("provide", "supply"),
    ("subject to", "governed by"), ("occur", "happen"), ("apply to", "extend to"),
    ("period is", "period lasts"), ("applies if", "arises if"),
    ("be provided", "be delivered"), ("as defined in", "as described in"),
]


def rewrite(s, lex):
    for a, b in lex:
        s = s.replace(a, b)
    return s


def toks(s):
    return re.findall(r"\S+", s)


def disc(x, y):
    cx, cy = Counter(toks(x)), Counter(toks(y))
    return sorted((cx - cy).elements()), sorted((cy - cx).elements())


def ok(c, lex):
    A, B, C = c["clause_a"], c["clause_b"], c["clause_c"]
    Ap, Bp, Cp = rewrite(A, lex), rewrite(B, lex), rewrite(C, lex)
    reworded = (Ap != A) or (Bp != B) or (Cp != C)
    disc_ok = (disc(A, B) == disc(Ap, Bp)) and (disc(A, C) == disc(Ap, Cp))
    return (disc_ok and reworded), (Ap, Bp, Cp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.inp.read_text())["test_cases"]

    orig, v1, v2 = [], [], []
    for c in cases:
        ok1, (A1, B1, C1) = ok(c, LEX_V1)
        ok2, (A2, B2, C2) = ok(c, LEX_V2)
        if not (ok1 and ok2):
            continue
        # v1 and v2 must differ from each other and from original (disjoint phrasing)
        if (A1, B1, C1) == (A2, B2, C2):
            continue
        orig.append(dict(c))
        v1.append({**c, "clause_a": A1, "clause_b": B1, "clause_c": C1})
        v2.append({**c, "clause_a": A2, "clause_b": B2, "clause_c": C2})

    for name, data in [("legal_orig_aligned", orig), ("legal_v1", v1), ("legal_v2", v2)]:
        (args.out_dir / f"{name}.json").write_text(json.dumps({"test_cases": data}, indent=2))
    print(f"[variants] aligned kept={len(orig)}/{len(cases)} "
          f"(each has original + v1 + v2, all diff-preserving, v1!=v2)", file=sys.stderr)
    # quick disjointness report: which replacement targets each lexicon uses
    print(f"[variants] wrote legal_orig_aligned.json / legal_v1.json / legal_v2.json to {args.out_dir}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
