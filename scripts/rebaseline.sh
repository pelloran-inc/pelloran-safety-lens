#!/usr/bin/env bash
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# REBASELINE (maintenance; OVERWRITES the published results). Use only when you
# intentionally re-baseline, e.g. after editing pins.json to new revisions or
# bumping the pinned library versions in pyproject.toml. It rebuilds the pools
# from pins.json, embeds and packs them, then OVERWRITES results/ and
# results/POOLS_SHA256SUMS. For a read-only reproduction check against the
# shipped release baseline, use reproduce.sh instead.
#
#   PYTHON=/path/to/pinned/venv/bin/python bash scripts/rebaseline.sh
#
set -euo pipefail
PYTHON="${PYTHON:-python3}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
export HF_DATASETS_TRUST_REMOTE_CODE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p pools AQTC
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# Pools (every dataset revision read from pins.json).
"$PYTHON" scripts/build_patent_distractor_pool.py    --out pools/patent_pool.json    --target 120000
"$PYTHON" scripts/build_financial_distractor_pool.py --out pools/financial_pool.json
"$PYTHON" scripts/build_medical_distractor_pool.py   --out pools/medical_pool.json   --target 120000
"$PYTHON" scripts/build_legal_distractor_pool.py --out-dir pools/legal_lexglue --source lexglue --n 120000
"$PYTHON" scripts/build_legal_distractor_pool.py --out-dir pools/legal_edgar   --source edgar   --n 120000
"$PYTHON" scripts/manifest_to_distractor_pool.py --in pools/legal_lexglue pools/legal_edgar --out pools/legal_pool.json

# Embed + pack (E5 model revision read from pins.json).
declare -A TRAP=( [legal]=legal_299 [medical]=medical_300 [patent]=patent_301 [financial]=financial_301 )
for d in legal medical patent financial; do
  log "embed+pack: $d"
  "$PYTHON" scripts/embed_and_pack.py --triples "data/traps/${TRAP[$d]}.json" \
      --distractor-pool "pools/${d}_pool.json" --name "$d" \
      --pool-sizes 5000 50000 100000 --out-dir AQTC
done

# OVERWRITE the published results and pool hashes (the maintenance step).
log "OVERWRITING results/ and results/POOLS_SHA256SUMS"
"$PYTHON" scripts/regenerate_results.py --aqtc-dir AQTC
( cd AQTC && sha256sum *.aqtc ) > results/POOLS_SHA256SUMS
log "REBASELINE_DONE (results overwritten; review and commit the change deliberately)"
