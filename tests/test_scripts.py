# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Every script that parses options must at least answer --help.

A script whose module needs a library that is not installed (torch,
sentence-transformers, datasets) is skipped, not failed.

Run: python3 -m unittest discover -s tests
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def argparse_scripts():
    out = []
    for p in sorted((REPO / "scripts").glob("*.py")):
        if p.name.startswith("._"):   # skip macOS AppleDouble metadata files
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if "argparse" in text:
            out.append(p)
    return out


def help_result(script: Path):
    return subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)


class TestScriptsAnswerHelp(unittest.TestCase):
    def test_help(self):
        for script in argparse_scripts():
            with self.subTest(script=script.name):
                r = help_result(script)
                if r.returncode != 0 and "ModuleNotFoundError" in r.stderr:
                    self.skipTest(f"{script.name}: {r.stderr.strip().splitlines()[-1]}")
                self.assertEqual(r.returncode, 0, f"{script.name}: {r.stderr.strip()[-300:]}")


class TestTheHelpCheckCatchesFaults(unittest.TestCase):
    def test_duplicate_option_is_caught(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.py"
            bad.write_text("import argparse\nap = argparse.ArgumentParser()\n"
                           "ap.add_argument('--name')\nap.add_argument('--name')\nap.parse_args()\n")
            self.assertNotEqual(help_result(bad).returncode, 0)


if __name__ == "__main__":
    unittest.main()
