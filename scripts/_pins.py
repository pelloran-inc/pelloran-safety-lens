# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Pinned Hugging Face revisions, read from the shipped ``pins.json``.

The pool builders and ``embed_and_pack.py`` call ``dataset_rev(name)`` /
``model_rev(name)`` and pass the result as ``revision=`` to
``load_dataset`` / the model loader, so a rebuild uses the exact commits
that produced the shipped pools and results, never "latest". Returns
``None`` (HF default) only if ``pins.json`` is absent or lacks the entry.
"""
import json
from pathlib import Path

_CACHE = None


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        p = Path(__file__).resolve().parent.parent / "pins.json"
        try:
            _CACHE = json.loads(p.read_text())
        except Exception:
            _CACHE = {"model": {}, "datasets": {}}
    return _CACHE


def dataset_rev(name: str):
    return _load().get("datasets", {}).get(name)


def model_rev(name: str):
    return _load().get("model", {}).get(name)
