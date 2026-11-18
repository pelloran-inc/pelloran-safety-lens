# Releasing

**Every release ships its change-list.**
- [CHANGELOG.md](CHANGELOG.md) holds one entry per release.
- The GitHub release notes and any announcement are written from that entry.
- An announcement says nothing the entry does not.

## The steps, in order

1. **CHANGELOG.md first.**
   - Move the `[Unreleased]` items under a new heading, `## [X.Y.Z] - YYYY-MM-DD`.
   - Use the Keep a Changelog sections: Added, Changed, Deprecated, Removed, Fixed, Security.
   - Describe what a user sees.
2. **CITATION.cff:** set `version` to `X.Y.Z` and `date-released` to the release date; add the DOI once minted.
3. **The gates pass** (no model, no download):
   - `python3 -m unittest discover -s tests` ends with `OK`.
   - `(cd lenses && shasum -a 256 -c SHA256SUMS)` and `(cd data && shasum -a 256 -c SHA256SUMS)` are all `OK`.
   - `gitleaks dir .` reports no leaks.
4. **The reproduction holds** on a machine with the pinned environment (needs E5):
   - `PYTHON=/venv/bin/python bash scripts/reproduce.sh` → `checked 40, mismatched 0`.
   - `PYTHON=/venv/bin/python bash scripts/reproduce_extras.sh --aqtc-dir DIR` → `checked 26, mismatched 0`.
5. **Mint the DOI** (Zenodo), and surface it in `CITATION.cff`, the README badges + Citation section, and the paper.
6. **Tag** the release commit `vX.Y.Z` and create the GitHub release from that tag.
7. **The notes:** the GitHub release's notes are the CHANGELOG entry for `X.Y.Z`.

A release's version is its git tag, `vX.Y.Z`.
