#!/usr/bin/env bash
# =====================================================================
# 統合仕様 v5.2 パイプライン (スクリプト名の v50 は開発初期の名残)
# usage: ./run_v50.sh --derived-input <dir> --run-id <id> --outputs-root <dir> [--quick] [--fullfig]
#   --quick   : E7感度解析・データサイズ解析を省略 (主要解析E1-E5+回帰テストのみ)
#   --fullfig : 論文で使わない補助図 (概念図figC/計量figD/スケーリングfigR1) も生成
# パス引数は絶対パスで渡す (本スクリプトは冒頭で code/ へ移動する)。詳細は docs/USAGE_ja.md。
# =====================================================================
set -euo pipefail
ORIG_PWD="$PWD"
cd "$(dirname "$0")"

QUICK=0
FULLFIG=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --derived-input) DERIVED="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --outputs-root) OROOT="$2"; shift 2;;
    --quick) QUICK=1; shift;;
    --fullfig) FULLFIG="--fullfig"; shift;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done
: "${DERIVED:?--derived-input required}" "${RUN_ID:?--run-id required}" "${OROOT:?--outputs-root required}"

# 相対パスは呼び出し元ディレクトリ基準で解決する
abspath(){ case "$1" in /*) echo "$1";; *) echo "$ORIG_PWD/$1";; esac; }
export KM_DERIVED="$(cd "$(abspath "$DERIVED")" && pwd)"
OROOT="$(abspath "$OROOT")"
export KM_OUT="$OROOT/$RUN_ID"
export KM_DATA="${KM_DATA:-$KM_OUT/work}"   # 実行ごとに分離し、成果物と同じ場所に残す
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$KM_OUT" "$KM_DATA"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$KM_OUT/run_v50.log"; }

log "=== ext_v50 run: derived=$KM_DERIVED run_id=$RUN_ID quick=$QUICK ==="
log "--- Step 0: derived input -> corpus (§1.1) ---"
python3 00_derived_input_corpus.py

log "--- Step 1: TF-IDF L2/L1 (§1.3/§2) ---"
python3 02_tfidf_sklearn.py

log "--- Step 2: 2D map v3.0厳密定義 (§3) データ駆動アンカー+中央 ---"
python3 12_map_v30.py main

log "--- Step 3: 連続場 SPH/GPR(学習ゲート) (§4) ---"
python3 04_fields.py gpr
python3 04_fields.py grid
python3 04_fields.py grad

log "--- Step 4: 多計量測地線 (§5-6) ---"
python3 05_geodesics.py pairs
python3 05_geodesics.py graph
for p in near_intra mid_adjacent far_max_cos cross_max_2d high_uncertainty far_centroids; do
  for m in sph fr gpr1 gpr4 gpr9; do
    log "geodesic: $p / $m"
    python3 05_geodesics.py run "$p" "$m"
  done
done
python3 - <<'PY'
import json, os
O=os.environ["KM_OUT"]; D=os.environ["KM_DATA"]
ad=json.load(open(O+"/map_audit.json"))["anchor_docs"]
p=json.load(open(D+"/pairs.json"))
p.update({"G1_center_mm":[ad[8],ad[0]],"G2_center_pm":[ad[8],ad[1]],
          "G3_center_pp":[ad[8],ad[2]],"G4_center_mp":[ad[8],ad[3]]})
json.dump(p,open(D+"/pairs.json","w"),indent=1)
PY
for g in G1_center_mm G2_center_pm G3_center_pp G4_center_mp; do
  log "regression geodesic: $g"
  python3 05_geodesics.py run "$g" sph
done
python3 05_geodesics.py eval

log "--- Step 5: E1/E2 (§7) ---"
python3 06_eval.py e1
python3 06_eval.py e2 global
python3 06_eval.py e2 knn_adaptive
python3 10_extra.py

log "--- Step 5.5: E9 言語化L1/L2比較 + E10 既存埋め込み比較 ---"
python3 14_e9_verbalization.py far_max_cos
python3 15_e10_embeddings.py

log "--- Step 5.6: 計算キャッシュの成果物退避 (経路・GPR・クラスタ) ---"
mkdir -p "$KM_OUT/paths_cache"
cp "$KM_DATA"/path_*.npy "$KM_DATA"/geo_*.json "$KM_DATA"/pairs.json    "$KM_DATA"/cluster_labels.npy "$KM_DATA"/gpr_model.pkl "$KM_OUT/paths_cache/" 2>/dev/null || true

if [[ $QUICK -eq 0 ]]; then
  log "--- Step 6: E7 安定性 + データサイズ (§8-9) ---"
  python3 07_stability.py seedmaps 0 1 2 3 4 5 10 20
  python3 07_stability.py seedpost
  python3 12_map_v30.py anch4 0 4
  python3 12_map_v30.py anch8 0 8
  python3 07_stability.py anchors
  python3 07_stability.py regs 0 12
  python3 07_stability.py regpost
  python3 07_stability.py sph_h
  python3 07_stability.py gpr_kernels
  python3 07_stability.py delta
  python3 07_stability.py multistart
  N=$(python3 -c "import json,os;print(len(json.load(open(os.environ['KM_OUT']+'/corpus/docs_clean.json'))))")
  python3 07_stability.py datasize 20
  python3 07_stability.py datasize 50
  python3 07_stability.py datasize "$N"
else
  log "--- Step 6: (quick mode: E7/datasize スキップ — 報告書は該当節を未実施と記載) ---"
fi

log "--- Step 7: 図表 Figure A-I (§11) ---"
for grp in maps fields paths reports; do python3 08_figures.py "$grp" $FULLFIG || log "figures $grp: 一部スキップ"; done

log "--- Step 8: manifest(§9.5) + LLMエクスポート ---"
python3 09_manifest.py
python3 13_llm_export.py
if [[ $QUICK -eq 0 ]]; then
  python3 11_reports.py
  log "--- Step 9: 検証ゲート ---"
  python3 validate.py
else
  log "(quick mode: R0-R6生成とゲートの一部はフルデータ依存のためスキップ可)"
  python3 validate.py || log "validate: quickモードでは一部FAILは想定内(E7未実施)"
fi

log "ext_v50 DONE: $KM_OUT"
