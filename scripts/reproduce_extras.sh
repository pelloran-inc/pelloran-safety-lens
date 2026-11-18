#!/usr/bin/env bash
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# REPRODUCE EXTRAS (read-only) — the v0.16 additions:
#   * Baselines: random projection + PCA at d=11,32 vs the lens.
#   * Robustness: the shipped and improved Legal lens on held-out
#     rewordings of the declared Legal cases, including the query-only test
#     (reword the query, keep the corpus) that isolates query from corpus.
# Rebuilds the artifacts from the package and checks them against the shipped
# expected values. EXACT match (tolerance 0); reports case counts, trap hits and
# correct hits for every cell. Overwrites nothing.
#
#   PYTHON=/path/to/pinned/venv/bin/python bash scripts/reproduce_extras.sh --aqtc-dir DIR
#
# DIR holds <domain>_5k.aqtc matching results/POOLS_SHA256SUMS. The robustness
# half needs E5 (same pinned revision as the pools).
set -euo pipefail
PY="${PYTHON:-python3}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
AQTC=""
while [ $# -gt 0 ]; do case "$1" in --aqtc-dir) AQTC="$2"; shift 2;; *) echo "unknown arg $1"; exit 2;; esac; done
[ -n "$AQTC" ] || { echo "usage: PYTHON=... bash scripts/reproduce_extras.sh --aqtc-dir DIR"; exit 2; }
WORK="${WORK:-$REPO/extras_work}"; mkdir -p "$WORK/lenses" "$WORK/results"
export HF_HUB_DISABLE_PROGRESS_BARS=1
MREV="$("$PY" -c "import json;print(json.load(open('pins.json'))['model']['intfloat/e5-large-v2'])")"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "=== 1. baselines: generate RP + PCA, measure 16 cells ==="
"$PY" scripts/make_baseline_lenses.py --aqtc-dir "$AQTC" --out "$WORK/lenses"
declare -A DS=( [legal]=legal_299 [medical]=medical_300 [patent]=patent_301 [financial]=financial_301 )
for dom in legal medical patent financial; do
  for proj in rp_d11 rp_d32 pca_${dom}_d11 pca_${dom}_d32; do
    "$PY" scripts/miss_diagnostic.py --lens "$WORK/lenses/${proj}.aqwt" --dataset "data/traps/${DS[$dom]}.json" \
        --aqtc "$AQTC/${dom}_5k.aqtc" --out "$WORK/results/${proj/${dom}_/}_${dom}.json" --k 10 2>/dev/null
  done
done

log "=== 2. robustness: aligned variants, splice-embed, measure joint + query-only (frozen pool) ==="
"$PY" scripts/make_legal_variants.py --in data/traps/legal_299.json --out-dir "$WORK"
for s in legal_orig_aligned legal_v1 legal_v2; do
  "$PY" scripts/embed_triples_splice.py --triples "$WORK/$s.json" --src-aqtc "$AQTC/legal_5k.aqtc" \
      --out-aqtc "$WORK/${s}_5k.aqtc" --model-revision "$MREV"
done
# joint rewording: whole triple (query + corpus) reworded
for s in legal_orig_aligned legal_v1 legal_v2; do
  for L in existing:legal_c02_d11 improved:legal_improved_d11; do
    "$PY" scripts/miss_diagnostic.py --lens "lenses/${L##*:}.aqwt" --dataset "$WORK/$s.json" \
        --aqtc "$WORK/${s}_5k.aqtc" --out "$WORK/results/res_${s}_${L%%:*}.json" --k 10 2>/dev/null
  done
done
# query-only: reworded query (v1 = seen by the improved lens; v2 = held-out) vs the UNCHANGED original pool
for QV in v1 v2; do
  suf=$([ "$QV" = v1 ] && echo "" || echo "_v2")
  for L in existing:legal_c02_d11 improved:legal_improved_d11; do
    "$PY" scripts/query_only_diagnostic.py --lens "lenses/${L##*:}.aqwt" \
        --orig-aqtc "$WORK/legal_orig_aligned_5k.aqtc" --query-aqtc "$WORK/legal_${QV}_5k.aqtc" \
        --dataset "$WORK/legal_orig_aligned.json" --out "$WORK/results/res_queryonly${suf}_${L%%:*}.json" --k 10 2>/dev/null
  done
done

log "=== 3. compare to shipped expected (EXACT, tolerance 0); report counts ==="
"$PY" - "$WORK" "$REPO" <<'PY'
import json,sys,os
WORK,REPO=sys.argv[1],sys.argv[2]
bad=0; n=0
def load(p):
    d=json.load(open(p)); s=d["summary"]; nc=d["meta"]["n_cases"]
    return (round(s["hn_at_k"],4), round(s["hitc_at_k"],4),
            nc, s["hn_leaks_count"], nc-s["hitc_misses_count"])
def chk(tag, f, ehn, ehc, enc):
    # compares HN@10, HitC@10 AND the case count against the expected values, all exact.
    global bad,n; n+=1
    if not os.path.exists(f): print(f"  MISS {tag}"); bad+=1; return
    hn,hc,nc,traphits,corr=load(f)
    ok = (hn==round(ehn,4)) and (hc==round(ehc,4)) and (nc==enc)
    if not ok: bad+=1
    cflag = "" if nc==enc else f"  CASES {nc}!=exp{enc}"
    print(f"  {'OK  ' if ok else 'DIFF'} {tag:50} HN@10={hn:.4f}(exp {ehn:.4f}) "
          f"HitC@10={hc:.4f}(exp {ehc:.4f}) [{nc}/{enc} cases, {traphits} trap-hits, {corr} correct]{cflag}")
DOMN={"legal":299,"medical":300,"patent":301,"financial":301}
# baselines: HN + HitC + case count, exact
B=json.load(open(REPO+"/results/baselines/BASELINES_EXPECTED.json"))
for proj in B["HN10"]:
    for dom in B["HN10"][proj]:
        chk(f"baseline {proj} {dom}", f"{WORK}/results/{proj}_{dom}.json",
            B["HN10"][proj][dom], B["HitC10"][proj][dom], DOMN[dom])
# robustness: HN + HitC + case count, exact; query-only uses the frozen-pool diagnostic
R=json.load(open(REPO+"/results/robustness/ROBUSTNESS_EXPECTED.json"))
M=[("res_legal_orig_aligned_existing.json","existing","trained_phrasings","existing_lens_legal_c02_d11"),
   ("res_legal_v1_existing.json","existing","held_out_paraphrases_v1","existing_lens_legal_c02_d11"),
   ("res_legal_v2_existing.json","existing","held_out_paraphrases_v2","existing_lens_legal_c02_d11"),
   ("res_queryonly_existing.json","existing","query_only_reworded_query_unchanged_corpus","existing_lens_legal_c02_d11"),
   ("res_queryonly_v2_existing.json","existing","query_only_v2_heldout_unchanged_corpus","existing_lens_legal_c02_d11"),
   ("res_legal_v1_improved.json","improved","declared_phrasings_v1","improved_lens_legal_improved_d11"),
   ("res_legal_orig_aligned_improved.json","improved","original_cases","improved_lens_legal_improved_d11"),
   ("res_legal_v2_improved.json","improved","held_out_v2","improved_lens_legal_improved_d11"),
   ("res_queryonly_improved.json","improved","query_only_reworded_query_unchanged_corpus","improved_lens_legal_improved_d11"),
   ("res_queryonly_v2_improved.json","improved","query_only_v2_heldout_unchanged_corpus","improved_lens_legal_improved_d11")]
for f,lens,key,grp in M:
    e=R[grp][key]
    chk(f"robustness {lens} {key}", f"{WORK}/results/{f}", e["HN10"], e["HitC10"], 194)
print(f"\nchecked {n}, mismatched {bad} (EXACT, tolerance 0; HN + HitC + case count)")
sys.exit(1 if bad else 0)
PY
echo "[$(date +%H:%M:%S)] REPRODUCE_EXTRAS done"
