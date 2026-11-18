# Data

## Trap sets (`data/traps/`)

Written for this work: the clauses are hand-written or generated from templates by the scripts in `scripts/`, not copied from a corpus. Each case has an `id`, a `trap_type` and three clauses:
- `clause_a`, the query;
- `clause_b`, the trap: a small change to A that changes its meaning;
- `clause_c`, the same content as A in other words.

| File | Cases | Trap types (cases) |
|:---|:-:|:---|
| `legal_299.json` | 299 | 23 types of contract-clause traps. The largest: numeric unit (102), numeric threshold boundary (46), temporal scope window (45), anchor format (33), conditional trigger (33) |
| `medical_300.json` | 300 | dosage (75), contraindication (75), population (75), drug substitution (75) |
| `patent_301.json` | 301 | claim scope, comprising / consisting of (75); enablement (74); prior art (76); mechanism substitution (76) |
| `financial_301.json` | 301 | direction (75), polarity (75), period (76), party (75) |

In the legal set, `id` is not unique: 265 distinct ids cover 299 cases, and 12 ids are each shared by several cases with different clauses. Refer to a case by its position in `test_cases`, as the results do (`case_index`).

The legal set keeps two fields from its de-duplication: `_dedup_original_row_idx` and `_dedup_all_instances_row_idxs`. They record each case's row in the 384-row set it was de-duplicated from (`scripts/dedup_trap_set.py`).

## Distractor pools (not in this package)

The distractors are short clauses, about 8 to 30 words, sentence-split from public datasets of each domain. This package does not redistribute them; the builder scripts download them from their sources. Each source has its own terms of use, and anyone who builds a pool must follow them.

The licence column gives what each source's Hugging Face dataset card states, as checked on 2026-11-05. A card may repackage older data whose own terms also apply.

| Domain | Source | Licence on the card |
|:---|:---|:---|
| Legal | `lex_glue` (LEDGAR, UNFAIR-ToS) | CC BY 4.0 |
| Legal | `eloukas/edgar-corpus` (SEC EDGAR 10-K filings) | U.S.-government public domain |
| Medical | `jmhb/pubmed_bioasq_2022` (PubMed abstracts) | CC0 (public domain) |
| Patent | `big_patent` (nine BIGPATENT categories) | CC BY 4.0 |
| Financial | `nickmuchi/financial-classification` | none stated |
| Financial | `zeroshot/twitter-financial-news-sentiment` | MIT |
| Financial | `zeroshot/twitter-financial-news-topic` | MIT |
| Financial | `ChanceFocus/flare-fpb` (Financial PhraseBank) | MIT |
| Financial | `kdave/Indian_Financial_News` | none stated |

**Pinned for reproducibility.** The embeddings (and therefore the pools and the exact-search results) depend on the
model and library versions. The shipped results were produced with the E5 model `intfloat/e5-large-v2` at revision
`f169b11e22de13617baa190a028a32f3493550b6`, the medical dataset `jmhb/pubmed_bioasq_2022` at revision
`3082ea61ed0f87f25695746c1f1c0365d3d466e8`, and the `==`-pinned library versions in `pyproject.toml` (with
`pyarrow==15.0.2`). `pools/PINNED_REVISIONS.txt` records the revisions from the run that produced the shipped results.
The legal distractor pool merges `lex_glue` (LEDGAR/UNFAIR-ToS) and SEC EDGAR 10-K filings.

The embedding model `intfloat/e5-large-v2` is not part of this package either; its model card states the MIT licence.
