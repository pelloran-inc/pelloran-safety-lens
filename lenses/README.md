# Lenses

Each file is one safety lens: a matrix W of size 1024 × d in the `.aqwt` format that `scripts/miss_diagnostic.py` and `scripts/steering_lens_train.py` read. `SHA256SUMS` holds the SHA-256 of each file.

| File | d | β | margin | How it was trained | Trained on |
|:---|:-:|:-:|:-:|:---|:---|
| `legal_c02_d11.aqwt` | 11 | 12 | 0.08 | 200 epochs, warm-started from an earlier legal lens that was itself the end of a chain of warm starts. That chain is not part of this package, so this lens cannot be retrained from scratch with it. | the 384-row legal set that `data/traps/legal_299.json` was de-duplicated from |
| `legal_d32_v2.aqwt` | 32 | 15 | 0.10 | 200 epochs cold start, then a warm-started second round | `data/traps/legal_299.json` |
| `medical_d2c_d32.aqwt` | 32 | 15 | 0.10 | warm-started from a cold-start lens (d = 32, β = 10, margin = 0.05) | `data/traps/medical_300.json` |
| `patent_v3_d32.aqwt` | 32 | 15 | 0.10 | cold start, then two warm-started rounds | `data/traps/patent_301.json` |
| `financial_v4_d32.aqwt` | 32 | 17 | 0.12 | cold start (β = 15, margin = 0.10), a warm-started round with the same settings, then a warm-started round with β = 17, margin = 0.12 | `data/traps/financial_301.json` |

Common settings: α = 1.0, seed = 42, E5-Large-v2 embeddings of the trap set's clauses.

**Each lens is built for the trap types of its own domain.** It is trained on its domain's trap set, and `results/pool_sizes/` measures it on that set, among distractors of the same domain. `results/matrix/` measures each lens on the other domains' trap sets, where it is not meant to work.
