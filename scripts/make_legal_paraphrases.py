#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Held-out paraphrases of the DECLARED legal cases.

Each declared case (A, B, C) is reworded by a fixed, fully-listed carrier
lexicon applied IDENTICALLY to A, B and C. The trap is preserved by a hard
guarantee: after rewording, the token-level symmetric difference between A'
and B' must equal that between A and B (and likewise A/C). Because the lexicon
only rewrites words common to the carrier, the trap-discriminating tokens
(numbers, Section/Schedule anchors, not/only/unless/except, party names,
may/must, per-occurrence/month, delivery/acceptance) are never touched; the
diff-check PROVES it per case and drops any case where it is violated.

This is a measurement of whether the EXISTING lens already covers phrasing
variety of the declared cases (concept-scoped-exact).
It does not retrain anything.
"""
import argparse, json, re, sys
from pathlib import Path

# Fixed carrier lexicon (longest first). Rewrites only neutral carrier phrases;
# never numbers, anchors, polarity words, party names, modal may/must, or the
# per-X / delivery-acceptance discriminators. The diff-check enforces this.
LEXICON = [
    ("For purposes of this Agreement", "Under this Agreement"),
    ("as set forth in", "as specified in"),
    ("set forth in", "specified in"),
    ("as set out in", "as laid out in"),
    ("shall", "will"),
    ("comply with", "observe"),
    ("maintain", "keep"),
    ("provide", "furnish"),
    ("subject to", "under"),
    ("occur", "take place"),
    ("apply to", "govern"),
    ("period is", "period runs for"),      # temporal_scope_window: keeps delivery/acceptance + warranty/maintenance
    ("applies if", "is triggered if"),     # numeric_threshold_boundary: keeps falls below (or equal to) + uptime/latency
    ("be provided", "be supplied"),        # time_window_condition: keeps 24/7 vs business-hours
]


def rewrite(s: str) -> str:
    for a, b in LEXICON:
        s = s.replace(a, b)
    return s


def toks(s: str):
    return re.findall(r"\S+", s)


def sym_disc(x: str, y: str):
    """Multiset symmetric difference of whitespace tokens (order-independent)."""
    from collections import Counter
    cx, cy = Counter(toks(x)), Counter(toks(y))
    only_x = list((cx - cy).elements())
    only_y = list((cy - cx).elements())
    return sorted(only_x), sorted(only_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-originals", type=Path, required=True,
                    help="matched trained-phrasing reference: the kept cases with ORIGINAL clauses, same order")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    cases = json.loads(args.inp.read_text())["test_cases"]
    kept, kept_orig, dropped, examples = [], [], [], []
    from collections import Counter
    kept_by_type, drop_by_type, drop_reason = Counter(), Counter(), Counter()
    seen_ex = set()

    for c in cases:
        A, B, C = c["clause_a"], c["clause_b"], c["clause_c"]
        Ap, Bp, Cp = rewrite(A), rewrite(B), rewrite(C)
        # trap-discriminator preservation (the hard guarantee)
        ab = sym_disc(A, B); abp = sym_disc(Ap, Bp)
        ac = sym_disc(A, C); acp = sym_disc(Ap, Cp)
        reworded = (Ap != A) or (Bp != B) or (Cp != C)
        disc_ok = (ab == abp) and (ac == acp)
        if disc_ok and reworded:
            kept.append({**c, "clause_a": Ap, "clause_b": Bp, "clause_c": Cp})
            kept_orig.append(dict(c))
            kept_by_type[c["trap_type"]] += 1
            if c["trap_type"] not in seen_ex:  # one verbatim example per trap_type, paired correctly
                seen_ex.add(c["trap_type"])
                examples.append({"id": c.get("id"), "trap_type": c["trap_type"],
                                 "A": A, "Ap": Ap, "B": B, "Bp": Bp, "C": C, "Cp": Cp})
        else:
            reason = "no_rewording" if (disc_ok and not reworded) else "disc_changed"
            dropped.append({"id": c.get("id"), "trap_type": c.get("trap_type"), "reason": reason,
                            "A": A, "Ap": Ap, "B": B, "Bp": Bp, "C": C, "Cp": Cp})
            drop_by_type[c["trap_type"]] += 1
            drop_reason[reason] += 1

    args.out.write_text(json.dumps({"test_cases": kept}, indent=2))
    args.out_originals.write_text(json.dumps({"test_cases": kept_orig}, indent=2))
    report = {
        "n_in": len(cases), "n_kept": len(kept), "n_dropped": len(dropped),
        "drop_rate": round(len(dropped) / len(cases), 4),
        "drop_reason": dict(drop_reason),
        "kept_by_type": dict(kept_by_type), "drop_by_type": dict(drop_by_type),
        "lexicon": LEXICON,
        "examples": examples,
        "dropped_disc_changed": [d for d in dropped if d["reason"] == "disc_changed"][:40],
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(f"[para] kept {len(kept)}/{len(cases)} (drop_rate={report['drop_rate']}); "
          f"wrote {args.out} + {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
