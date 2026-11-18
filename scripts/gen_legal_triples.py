#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Generate an extended Liability Trap dataset with multiple "trap types".

This file intentionally produces *synthetic* clauses that stress retrieval failure modes:
- negation flips (shall vs shall not)
- exceptions / unless (scope negation)
- modality flips (may vs must, may vs may not)
- numeric threshold flips (< vs >, at least vs at most)
- scope restriction flips (all vs only, including vs excluding)

Output schema matches the original dataset evaluator:
{
  "test_cases": [
    { "id": "...", "trap_type": "...", "clause_a": "...", "clause_b": "...", "clause_c": "..." },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Triplet:
    trap_type: str
    clause_a: str
    clause_b: str
    clause_c: str


def _liability_shall_not(actor_a: str, actor_b: str, obj: str) -> Triplet:
    a = f"The {actor_a} shall be liable for {obj}."
    b = f"The {actor_a} shall not be liable for {obj}."
    c = f"The {actor_b} shall be liable for {obj}."
    return Triplet("negation_shall_not", a, b, c)


def _unless_exception(actor_a: str, actor_b: str, obj: str, exception: str) -> Triplet:
    # A: liable except in condition; B: liable even in condition (flips exception scope)
    a = f"The {actor_a} shall be liable for {obj} unless {exception}."
    b = f"The {actor_a} shall be liable for {obj} even if {exception}."
    c = f"The {actor_b} shall be liable for {obj} unless {exception}."
    return Triplet("exception_unless", a, b, c)


def _modality_may_must(actor_a: str, actor_b: str, action: str) -> Triplet:
    # A: permission; B: obligation (policy flip)
    a = f"The {actor_a} may {action}."
    b = f"The {actor_a} must {action}."
    c = f"The {actor_b} may {action}."
    return Triplet("modality_may_vs_must", a, b, c)


def _numeric_threshold(actor_a: str, actor_b: str, amount: int, unit: str, action: str) -> Triplet:
    # A: requires action if > amount; B: requires action if < amount (threshold flip)
    a = f"If the {unit} is greater than {amount}, the {actor_a} shall {action}."
    b = f"If the {unit} is less than {amount}, the {actor_a} shall {action}."
    c = f"If the {unit} is greater than {amount}, the {actor_b} shall {action}."
    return Triplet("numeric_threshold_flip", a, b, c)


def _scope_only(actor_a: str, actor_b: str, obj_all: str, obj_only: str) -> Triplet:
    # A: includes more scope; B: restricts scope (can invert duties)
    a = f"The {actor_a} shall be responsible for {obj_all}."
    b = f"The {actor_a} shall be responsible only for {obj_only}."
    c = f"The {actor_b} shall be responsible for {obj_all}."
    return Triplet("scope_only_restriction", a, b, c)


def build_triplets(seed: int, per_type: int) -> List[Triplet]:
    rng = random.Random(seed)

    actors = [
        ("Lessee", "Lessor"),
        ("Buyer", "Seller"),
        ("Vendor", "Customer"),
        ("Tenant", "Landlord"),
        ("Service Provider", "Client"),
    ]

    liability_objs = [
        "structural repairs",
        "roof maintenance",
        "fire damage",
        "electrical repairs",
        "environmental remediation",
    ]

    exceptions = [
        "the damage is caused by force majeure",
        "the damage is caused by the other party's gross negligence",
        "the damage is caused by a third party",
        "the damage results from normal wear and tear",
        "the damage is covered by insurance",
    ]

    actions = [
        "terminate this agreement with 30 days' written notice",
        "access the premises during business hours",
        "subcontract the services to an approved third party",
        "store hazardous materials on the premises",
        "disclose confidential information to affiliates",
    ]

    numeric_units = [
        "payment amount",
        "late fee",
        "annual revenue",
        "penalty",
        "invoice total",
    ]

    numeric_actions = [
        "obtain prior written approval",
        "provide a compliance certificate",
        "notify the other party within 5 business days",
        "escalate the request to legal counsel",
        "trigger an audit",
    ]

    scope_all = [
        ("all repairs to the roof and walls", "repairs to the roof"),
        ("all customer data, including backups", "customer data excluding backups"),
        ("all fees and taxes", "fees"),
        ("all subcontractors and agents", "subcontractors"),
        ("all communications, including email", "communications excluding email"),
    ]

    out: List[Triplet] = []

    # For determinism + variety, we sample combinations with replacement.
    for _ in range(per_type):
        a, b = rng.choice(actors)
        obj = rng.choice(liability_objs)
        out.append(_liability_shall_not(a, b, obj))

    for _ in range(per_type):
        a, b = rng.choice(actors)
        obj = rng.choice(liability_objs)
        exc = rng.choice(exceptions)
        out.append(_unless_exception(a, b, obj, exc))

    for _ in range(per_type):
        a, b = rng.choice(actors)
        act = rng.choice(actions)
        out.append(_modality_may_must(a, b, act))

    for _ in range(per_type):
        a, b = rng.choice(actors)
        unit = rng.choice(numeric_units)
        amt = rng.choice([1000, 5000, 10000, 25000, 100000])
        act = rng.choice(numeric_actions)
        out.append(_numeric_threshold(a, b, amt, unit, act))

    for _ in range(per_type):
        a, b = rng.choice(actors)
        all_scope, only_scope = rng.choice(scope_all)
        out.append(_scope_only(a, b, all_scope, only_scope))

    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-type", type=int, default=50, help="How many cases per trap type")
    args = ap.parse_args()

    triplets = build_triplets(seed=int(args.seed), per_type=int(args.per_type))
    test_cases = []
    for i, t in enumerate(triplets):
        test_cases.append(
            {
                "id": f"lt_ext_{i:05d}",
                "trap_type": t.trap_type,
                "clause_a": t.clause_a,
                "clause_b": t.clause_b,
                "clause_c": t.clause_c,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"test_cases": test_cases}, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_path} (n_cases={len(test_cases)})")


if __name__ == "__main__":
    main()

