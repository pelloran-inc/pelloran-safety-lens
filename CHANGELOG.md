# Changelog

All notable changes to pelloran-safety-lens are listed here, newest first. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). A release's version is its git tag; how a release is
made is in [RELEASING.md](RELEASING.md).

## [1.1.0] — 2026-11-18

## [1.1.0] - 2026-11-13

Reproducibility for the paper's v0.16 additions, all pre-registered and hashed.

### Added

- **Baselines (§Baselines).** `scripts/make_baseline_lenses.py` builds a random Gaussian projection (seed 42) and
  unsupervised PCA (fit on the distractors only) at `d = 11, 32`; `results/baselines/BASELINES_EXPECTED.json` holds
  the per-domain `HN@10`/`HitC@10`. At the shipped dimensions these trap-agnostic projections leave the trap in the
  top-10 for 53–97 % of the declared cases, while the trained lens is `0.0000`.
- **Rewording robustness (§Robustness).** `scripts/make_legal_paraphrases.py` and `scripts/make_legal_variants.py`
  rewrite the declared Legal cases with a fixed carrier lexicon under a per-case trap-preservation (diff) guarantee;
  `scripts/embed_triples_splice.py` embeds them (E5) onto the shipped distractor block.
  `lenses/legal_improved_d11.aqwt` is an improved, phrasing-robust Legal lens (fine-tuned from `legal_c02_d11`);
  `results/robustness/ROBUSTNESS_EXPECTED.json` holds the shipped-vs-improved numbers.
- **`scripts/reproduce_extras.sh`** — rebuilds both and checks them against the expected values (read-only, **exact**,
  tolerance 0; reports case counts + trap hits; `checked 24, mismatched 0`). Needs E5 for the robustness half.
- **Query-only rewording test** (`scripts/query_only_diagnostic.py`). Passes the reworded query `A'` separately and
  searches the **unchanged** original pool (original `A`/`B`/`C` + distractors — no reworded row in the corpus; the
  case's own original `A` is still excluded from its ranking). Safety holds (HN 0.0052) but utility drops
  (HitC 0.7938 shipped / 0.8763 improved, vs 0.9845 under joint rewording). Recorded in
  `results/robustness/ROBUSTNESS_EXPECTED.json`; checked by `reproduce_extras.sh`.

### Fixed

- `check_results.py` no longer raises `KeyError: 'meta'` on the new `results/**/*_EXPECTED.json` manifests — it skips
  files without a result schema (the integrity tests already did). The documented reproduction path is intact.
- The query-only test now freezes the pool. An earlier version rebuilt the pool from the reworded `A'` rows, so the
  other cases' `A'` competed as distractors and understated HitC (0.7216 / 0.7474); corrected to 0.7938 / 0.8763.

### Notes

- The improved lens weights are not bit-reproducible across training runs (CPU float non-determinism); its measured
  `HN`/`HitC` are stable, and the shipped weights are the ones measured.

## [1.0.0] - 2026-11-12

The first public release: the reproducibility package for the paper *Small Learned Projections Reduce
Semantic-Reversal Errors in Dense Retrieval: A Four-Domain Empirical Study*.

### Added

- **Trap sets** (`data/traps/`): legal 299, medical 300, patent 301 and financial 301 cases. Each case has a query A,
  a trap B that changes one meaning-carrying detail, and a paraphrase C. `data/SHA256SUMS` holds their hashes.
- **Five safety lenses** (`lenses/`), the ones the paper measures: Legal (d = 11 and d = 32), Medical, Patent and
  Financial (d = 32). `lenses/SHA256SUMS` holds their hashes.
- **The recipe's scripts** (`scripts/`): trap generators and curation, distractor-pool builders, embedding and packing,
  the duplicate check, the lens trainer, the exact-search measurement (`miss_diagnostic.py`, with `--raw` for raw
  cosine without a lens) and the cross-domain run (`run_matrix.py`).
- **Stored results** (`results/`): the pool sizes (5k, 50k, 100k), raw cosine without a lens, the cross-domain matrix
  and the dimension ablation. `scripts/check_results.py` re-measures every one of them on its pool binary.
  `results/POOLS_SHA256SUMS` holds the hashes of the twelve pool binaries they were measured on.
- **Integrity tests** (`tests/`), which need no model and no download, and a GitHub Actions workflow that runs them.
- **Licences:** GNU AGPL-3.0 (`LICENSE`), a commercial licence on request (`COMMERCIAL.md`), `NOTICE`, and the data
  sources' terms (`DATA.md`).

### Not included

- The distractor pools and their embeddings. The builder scripts download the public sources; see `DATA.md`.
- The proprietary retriever behind the paper's wall-clock figures. No quality number needs it.
