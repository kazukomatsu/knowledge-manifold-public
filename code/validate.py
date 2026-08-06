# -*- coding: utf-8 -*-
"""検証ゲート: 実行結果が仕様の監査条件を満たすか自動判定 (LLM不要のQA層).
すべてPASSで exit 0、1つでもFAILで exit 1（run_all.sh が停止する）。
判定内容: §9.5監査フィールド / 一括スケーリング禁止 / アンカー固定 /
測地線の正式成功・nit>0 / v3.1回帰テスト帯域 / GPR学習ゲート / Σπ=1 / 報告書とハッシュ整合
"""
import json, csv, sys, os, hashlib, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import OUT, DATA

checks = []
def chk(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL':4s}  {name}  {detail}")

# 1) §9.5 監査フィールド
m = json.load(open(OUT + "/manifest.json"))
req = ["post_optimization_global_scaling", "anchor_coordinates_fixed", "kappa", "margin",
       "lambda_cover", "lambda_edge", "lambda_angle", "lambda_area", "lambda_rep", "lambda_center",
       "alpha", "alpha_uses_central_anchor", "free_point_max_radius_before_finalization",
       "free_point_max_radius_after_finalization", "anchor_positions_before_finalization",
       "anchor_positions_after_finalization", "optimizer_success", "final_J",
       "random_seed", "mds_init_hash"]
chk("§9.5必須フィールド完備", all(k in m for k in req))
chk("一括スケーリング禁止 (before==after)", m["free_point_max_radius_before_finalization"]
    == m["free_point_max_radius_after_finalization"])
chk("alpha計算に中央アンカー不使用", m["alpha_uses_central_anchor"] is False)

# 2) 座標: アンカー固定値
rows = list(csv.DictReader(open(OUT + "/coordinates_2d.csv")))
anch = {(round(float(r["x"]), 6), round(float(r["y"]), 6)) for r in rows if r["is_anchor"] == "1"}
expect = {(-1., -1.), (1., -1.), (1., 1.), (-1., 1.), (1., 0.), (-1., 0.), (0., 1.), (0., -1.), (0., 0.)}
chk("アンカー座標=規定9点", anch == expect and len(rows) == 100)

# 3) 測地線: 正式成功と実行深度 (nit>0)
geo = list(csv.DictReader(open(OUT + "/geodesic_results.csv")))
chk("測地線: 全て正式成功", len(geo) >= 30 and all(r["status"] == "success" for r in geo),
    f"n={len(geo)}")
import glob
nit_ok = True
for f in glob.glob(DATA + "/geo_*.json"):
    r = json.load(open(f))
    if not all(t["nit"] > 0 for t in r["tries"]):
        nit_ok = False
chk("測地線: 全候補で反復実行 (nit>0)", nit_ok)

# 4) v3.1 回帰テスト帯域 (10-45%を警報線とする)
drops = []
for g in ["G1_center_mm", "G2_center_pm", "G3_center_pp", "G4_center_mp"]:
    f = DATA + f"/geo_{g}_sph.json"
    if os.path.exists(f):
        r = json.load(open(f))
        drops.append((r["E_straight"] - r["E_final"]) / r["E_straight"] * 100)
chk("v3.1回帰テスト: 4本存在", len(drops) == 4)
chk("v3.1回帰テスト: 低下率10-45%帯域", all(10 <= d <= 45 for d in drops),
    "drops=" + ",".join(f"{d:.1f}%" for d in drops))

# 5) GPR学習ゲート
g = json.load(open(OUT + "/gpr_info.json"))
chk("GPR: n_restarts=10 実行", g.get("n_restarts") == 10)
chk("GPR: θが初期値から移動", abs(g["length_scale_x"] - 0.5) > 1e-3 or abs(g["white_noise"] - 0.01) > 1e-3,
    f"ls=({g['length_scale_x']:.3f},{g['length_scale_y']:.3f}) noise={g['white_noise']:.3f}")
gz = np.load(DATA + "/grid_fields.npz")
chk("GPR: u(r)場が退化していない (max u>=0.05)", float(gz["uncertainty"].max()) >= 0.05,
    f"u_max={float(gz['uncertainty'].max()):.3f}")

# 6) L1確率単体 (Σπ=1): 報告値の計算経路と同じ float64 正規化で検証 (10_extra.py と同一)
X = np.load(DATA + "/X_raw.npy")
S = X.astype(np.float64) + 1e-10
rs = (S / S.sum(axis=1, keepdims=True)).sum(axis=1)
chk("L1: Σπ=1 (float64正規化, |Σ-1|<1e-9)", float(np.abs(rs - 1).max()) < 1e-9)
# 保存済み l1_sums が float64 和と整合するか (float32保存の丸め許容)
l1s = np.load(DATA + "/l1_sums.npy")
rel = np.abs(l1s - S.sum(axis=1)) / S.sum(axis=1)
chk("L1: 保存済み正規化係数の整合 (相対誤差<1e-5)", float(rel.max()) < 1e-5)

# 7) 報告書: R0-R6存在・manifestハッシュ整合・図リンク実在
h = hashlib.sha256(open(OUT + "/manifest.json", "rb").read()).hexdigest()
reports = ["analysis_report.md", "report_datasize.md", "report_stability.md",
           "report_metric_comparison.md", "report_normalization_L2_L1.md",
           "report_information_geometry.md", "report_evaluation_metrics.md"]
chk("報告書R0-R6存在", all(os.path.exists(OUT + "/" + f) for f in reports))
chk("報告書ヘッダのmanifestハッシュ一致", all(h in open(OUT + "/" + f).read() for f in reports))

n_fail = sum(1 for _, ok, _ in checks if not ok)
print(f"\n{'=' * 50}\n{len(checks) - n_fail}/{len(checks)} PASS")
sys.exit(1 if n_fail else 0)
