# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Integrity checks that need no corpus binary and no model.

Run: python3 -m unittest discover -s tests
"""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sum_mismatches(base: Path, sums_file: Path) -> list:
    """Files listed in a SHA256SUMS file whose hash differs or that are missing."""
    bad = []
    for line in sums_file.read_text().splitlines():
        want, name = line.split(maxsplit=1)
        p = base / name
        if not p.exists() or sha256(p) != want:
            bad.append(name)
    return bad


def summary_errors(result: dict) -> list:
    """A stored result must agree with its own case lists."""
    errs = []
    m, s = result["meta"], result["summary"]
    n = m["n_cases"]
    if s["hn_leaks_count"] != len(result["hn_leak_cases"]):
        errs.append("hn_leaks_count != len(hn_leak_cases)")
    if s["hitc_misses_count"] != len(result["hitc_miss_cases"]):
        errs.append("hitc_misses_count != len(hitc_miss_cases)")
    if s["hn_at_k"] != s["hn_leaks_count"] / n:
        errs.append("hn_at_k != hn_leaks_count / n_cases")
    if s["hitc_at_k"] != 1 - s["hitc_misses_count"] / n:
        errs.append("hitc_at_k != 1 - hitc_misses_count / n_cases")
    if m["n_pool"] != 3 * n + m["n_distr"]:
        errs.append("n_pool != 3 * n_cases + n_distr")
    return errs


class TestIntegrity(unittest.TestCase):
    def test_lenses_match_their_sums(self):
        self.assertEqual(sum_mismatches(REPO / "lenses", REPO / "lenses" / "SHA256SUMS"), [])

    def test_trap_sets_match_their_sums(self):
        self.assertEqual(sum_mismatches(REPO / "data", REPO / "data" / "SHA256SUMS"), [])

    def test_results_name_shipped_files(self):
        for res in (REPO / "results").rglob("*.json"):
            obj = json.loads(res.read_text())
            if "meta" not in obj:   # expected-value manifests (baselines/, robustness/) are not result files
                continue
            meta = obj["meta"]
            if meta["lens"] != "none (raw cosine)":   # a raw-cosine result has no lens
                self.assertTrue((REPO / meta["lens"]).exists(), f"{res.name}: {meta['lens']}")
            self.assertTrue((REPO / meta["dataset"]).exists(), f"{res.name}: {meta['dataset']}")

    def test_results_agree_with_themselves(self):
        for res in (REPO / "results").rglob("*.json"):
            obj = json.loads(res.read_text())
            if "meta" not in obj:   # skip expected-value manifests
                continue
            self.assertEqual(summary_errors(obj), [], res.name)

    def test_trap_sets_have_unique_triples(self):
        for p in (REPO / "data" / "traps").glob("*.json"):
            cases = json.loads(p.read_text())["test_cases"]
            keys = {(c["clause_a"], c["clause_b"], c["clause_c"]) for c in cases}
            self.assertEqual(len(keys), len(cases), p.name)


class TestTheChecksCatchFaults(unittest.TestCase):
    """Each check above must fail on a planted fault."""

    def test_flipped_byte_in_a_lens_is_caught(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shutil.copytree(REPO / "lenses", td / "lenses")
            victim = sorted((td / "lenses").glob("*.aqwt"))[0]
            b = bytearray(victim.read_bytes())
            b[-1] ^= 0x01
            victim.write_bytes(bytes(b))
            self.assertEqual(sum_mismatches(td / "lenses", td / "lenses" / "SHA256SUMS"), [victim.name])

    def test_inconsistent_result_is_caught(self):
        res = sorted((REPO / "results").rglob("*.json"))[0]
        r = json.loads(res.read_text())
        r["summary"]["hitc_at_k"] += 0.0001
        self.assertIn("hitc_at_k != 1 - hitc_misses_count / n_cases", summary_errors(r))


if __name__ == "__main__":
    unittest.main()
