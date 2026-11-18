#!/usr/bin/env bash
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# REPRODUCE (read-only). Rebuild the pools from the PINNED sources (pins.json),
# embed and pack them, and verify that the result matches the SHIPPED release
# baseline: the pool hashes in results/POOLS_SHA256SUMS and the stored results.
# It writes only into a scratch work dir and OVERWRITES NOTHING in the package.
# This is reproduction, not re-baselining; for the latter see rebaseline.sh.
#
#   PYTHON=/path/to/pinned/venv/bin/python bash scripts/reproduce.sh
#
set -euo pipefail
PYTHON="${PYTHON:-python3}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${WORK:-$REPO/repro_work}"
export HF_DATASETS_TRUST_REMOTE_CODE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$WORK/pools" "$WORK/AQTC"
cd "$REPO"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# Pools. The builders read pins.json (scripts/_pins.py) for every dataset
# revision, so a rebuild uses the exact commits that produced the shipped pools.
log "pool: patent";    "$PYTHON" scripts/build_patent_distractor_pool.py    --out "$WORK/pools/patent_pool.json" --target 120000
log "pool: financial"; "$PYTHON" scripts/build_financial_distractor_pool.py --out "$WORK/pools/financial_pool.json"
log "pool: medical";   "$PYTHON" scripts/build_medical_distractor_pool.py   --out "$WORK/pools/medical_pool.json" --target 120000
log "pool: legal (lexglue + EDGAR)"
"$PYTHON" scripts/build_legal_distractor_pool.py --out-dir "$WORK/pools/legal_lexglue" --source lexglue --n 120000
"$PYTHON" scripts/build_legal_distractor_pool.py --out-dir "$WORK/pools/legal_edgar"   --source edgar   --n 120000
"$PYTHON" scripts/manifest_to_distractor_pool.py --in "$WORK/pools/legal_lexglue" "$WORK/pools/legal_edgar" --out "$WORK/pools/legal_pool.json"

# Embed + pack. embed_and_pack reads pins.json for the E5 model revision.
declare -A TRAP=( [legal]=legal_299 [medical]=medical_300 [patent]=patent_301 [financial]=financial_301 )
for d in legal medical patent financial; do
  log "embed+pack: $d"
  "$PYTHON" scripts/embed_and_pack.py --triples "data/traps/${TRAP[$d]}.json" \
      --distractor-pool "$WORK/pools/${d}_pool.json" --name "$d" \
      --pool-sizes 5000 50000 100000 --out-dir "$WORK/AQTC"
done

# Verify against the SHIPPED baseline (read-only).
log "pool-hash check vs shipped results/POOLS_SHA256SUMS"
( cd "$WORK/AQTC" && sha256sum -c "$REPO/results/POOLS_SHA256SUMS" )
log "result check vs shipped results/ (exact)"
"$PYTHON" scripts/check_results.py --aqtc-dir "$WORK/AQTC"
log "REPRODUCE_OK (pools and results match the shipped release baseline)"
