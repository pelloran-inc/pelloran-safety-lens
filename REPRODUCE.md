# Reproduce

Every number in the paper's results comes from a file in `results/`. This file gives the commands that check
them, the hashes they check against, and what each command must print. Run the commands from the repository
root. The times are for one Apple M3 Max.

## 1. Without a model and without a download (seconds)

```
python3 -m unittest discover -s tests
(cd lenses && shasum -a 256 -c SHA256SUMS)
(cd data && shasum -a 256 -c SHA256SUMS)
```

- The tests check the hashes of the five lenses and the four trap sets. They also check that every stored result
  names files that exist, that every result agrees with itself (its counts against its lists of cases), and that
  every script answers `--help`. They must end with `OK`.
- Every hash line must end with `OK`. On Linux, use `sha256sum -c` instead of `shasum -a 256 -c`.

The same checks run on GitHub Actions for every push (`.github/workflows/tests.yml`).

## 2. Build the pool binaries

A pool is one domain's trap set plus 5 000, 50 000 or about 100 000 distractor clauses, embedded with
`intfloat/e5-large-v2`. The pools are not shipped. Steps 2 and 3 of the README's pipeline build them:
- the builder scripts download the public sources (`DATA.md`);
- `embed_and_pack.py` writes `<domain>_<size>.aqtc` into a directory, below called `DIR`.

**All four pools rebuild from public sources.** The one-command path, in a pinned environment, is

```
PYTHON=/path/to/venv/bin/python bash scripts/reproduce.sh
```

which reads the pinned revisions from `pins.json`, builds the four distractor pools, embeds and packs the twelve `.aqtc`,
and checks the result against the shipped `results/POOLS_SHA256SUMS` and stored results (read-only; it overwrites nothing). To re-baseline deliberately, use `scripts/rebaseline.sh`. The legal pool combines
`build_legal_distractor_pool.py` (sources `lexglue` and `edgar`, the latter the public-domain SEC EDGAR corpus)
merged by `manifest_to_distractor_pool.py`; the medical pool is built by `build_medical_distractor_pool.py` from the
CC0 `jmhb/pubmed_bioasq_2022`. Pin the environment (`pip install -e .[train,embed,pools]`, plus `pyarrow==15.0.2`,
on `datasets==2.19.0`) so the embeddings, and therefore the pools and results, reproduce.

## 3. Check your pools against the measured ones

```
(cd DIR && shasum -a 256 -c /path/to/pelloran-safety-lens/results/POOLS_SHA256SUMS)
```

`results/POOLS_SHA256SUMS` holds the SHA-256 of the twelve pool binaries that the stored results were measured
on. A line that fails means your pool is not the same bytes. A source dataset that changed since is one
possible cause. The re-measurement in step 4 may then differ too.

## 4. Re-measure every stored result (45 s)

```
python3 scripts/check_results.py --aqtc-dir DIR
```

For every stored result whose pool is in `DIR`, it runs the measurement again and compares the pool size, HN@10
and HitC@10 with the stored values. They must be equal, not close, because the search is exact. Two of its lines:

```
OK   results/pool_sizes/legal_c02_d11_on_legal_5k.json  HN@10=0.0000  HitC@10=0.9866  n_pool=5897
OK   results/raw_cosine/raw_cosine_on_legal_5k.json  HN@10=0.8127  HitC@10=0.6722  n_pool=5897
```

With all twelve pools, it must end with `checked 40, mismatched 0, skipped 0` and exit code 0. A result whose pool
is missing is skipped and listed. `--only _5k` checks the 24 results on the 5k pools (5 s).

## Which result backs which part of the paper

| Folder | Files | In the paper |
|:---|:-:|:---|
| `results/pool_sizes/` | 12 | The lens on its declared cases at 5k, 50k and 100k: HN@10 and HitC@10 |
| `results/raw_cosine/` | 12 | The same table's raw-cosine column, and the numbers without a lens |
| `results/matrix/` | 12 | The 4 × 4 cross-domain transfer matrix |
| `results/ablation/` | 4 | The Legal d = 32 lens on all four domains |
| `results/baselines/` | 1 | §Baselines: random projection + PCA at d = 11, 32 (HN@10, HitC@10) |
| `results/robustness/` | 1 | §Robustness: shipped vs improved Legal lens on held-out rewordings |

The paper's Appendix A gives the first 12 characters of each lens, pool and result hash. The full hashes are in
`lenses/SHA256SUMS`, `data/SHA256SUMS` and `results/POOLS_SHA256SUMS`, and the result files are in `results/`.

## 5. The v0.16 additions — baselines and rewording robustness

The paper's §Baselines and §Robustness are reproduced by one read-only command, against the shipped expected values
(`results/baselines/`, `results/robustness/`):

```
PYTHON=/path/to/venv/bin/python bash scripts/reproduce_extras.sh --aqtc-dir DIR
```

`DIR` holds the `<domain>_5k.aqtc` pools (e.g. `repro_work/AQTC` from `reproduce.sh`, or any build that matches
`results/POOLS_SHA256SUMS`). It builds the random-projection and PCA baselines (`make_baseline_lenses.py`), rewrites
the declared Legal cases under a per-case trap-preservation guarantee (`make_legal_variants.py`), embeds them with E5
onto the shipped distractor block (`embed_triples_splice.py`), measures the shipped `legal_c02_d11` and the improved
`legal_improved_d11` lenses, and checks every value against the expected files. It must end with
`checked 22, mismatched 0`. The baseline half needs no model; the robustness half needs E5 (same pinned revision as
the pools). The improved lens is shipped because its weights are not bit-reproducible across training runs (CPU float
non-determinism); its measured HN/HitC are stable.
