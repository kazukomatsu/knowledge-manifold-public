# -*- coding: utf-8 -*-
"""§10 検証結果レポート R0–R6 生成"""
import json, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import OUT, DATA

MH = open(OUT + "/manifest.sha256").read().strip()
HDR = (f"> 依拠 manifest.json sha256: `{MH}`  \n"
       f"> 座標ファイル: `coordinates_2d.csv` (N=100, seed=0, 8外周+中央アンカー)  \n"
       f"> 生成日: 2026-07-06 / Knowledge Manifold 統合仕様 v5.0 準拠\n\n")

def J(f): return json.load(open(OUT + "/" + f))

e1 = J("e1_distance_preservation.json")
e2g = J("e2_loo_global.json"); e2a = J("e2_loo_knn_adaptive.json")
seeds = J("e7_seeds.json"); anchors = J("e7_anchors.json"); regs = J("e7_regs.json")
sphh = J("e7_sph_h.json"); gprk = J("e7_gpr_kernels.json"); dlt = J("e7_delta.json")
ms = J("e7_multistart.json"); r1n20 = J("r1_N20.json")
eps = J("r5_epsilon_sensitivity.json"); lamdev = J("e4_lambda_path_deviation.json")
tfc = J("tfidf_config.json"); man = J("manifest.json")
metrics = [r for r in csv.DictReader(open(OUT + "/results_metrics.csv"))
           if not r["pair"].startswith("G")]  # G1-G4はAppendix S7で別記
geod = list(csv.DictReader(open(OUT + "/geodesic_results.csv")))
pairs = {k: v for k, v in json.load(open(DATA + "/pairs.json")).items()
         if not k.startswith("G")}   # G1-G4 (v3.1中央測地線) は別記

def fnum(x, d=4):
    return f"{float(x):.{d}f}"

def tbl(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s + "\n"

W = {}

# ================= R1 datasize =================
r1n50 = J("r1_N50.json"); r1n100 = J("r1_N100.json")
W["report_datasize.md"] = HDR + f"""# R1: データサイズ拡張報告書（§8）

## 目的
データ規模を3段階に変えて、スケーラビリティ・幾何構造の安定性・計算時間・指標変化を確認する（仕様 §8）。

## 段階の定義（本解析での再定義）
本コーパスは全100文書のため、仕様 §8 の原定義（Small=20 / Medium=100–300 / Large=1000+）を
**Small=20 / Medium=50 / Large=100** に再定義して評価した（ユーザー指示による変更として記録）。
N=20・50 は主コーパスから乱数 seed=0 で抽出したサブセット、N=100 は全文書。原定義の Large（N=1000+）はデータ非提供のため対象外。

## 数値表（Table A 対応）
""" + tbl(
    ["段階", "N", "TF-IDF次元(非ゼロ列)", "map最適化時間(s)", "LOO時間(s)", "ピークメモリ(MB)", "stress²(κ)", "Spearman", "LOO cosine", "最近傍距離CV"],
    [["Small", 20, r1n20["tfidf_nonzero_cols"], r1n20["map_time_sec"], r1n20["loo_time_sec"],
      round(r1n20["peak_mem_mb"]), fnum(r1n20["stress"]), fnum(r1n20["spearman"]),
      fnum(r1n20["loo_cosine_mean"],3), fnum(r1n20["nn_dist_cv"])],
     ["Medium", 50, r1n50["tfidf_nonzero_cols"], r1n50["map_time_sec"], r1n50["loo_time_sec"],
      round(r1n50["peak_mem_mb"]), fnum(r1n50["stress"]), fnum(r1n50["spearman"]),
      fnum(r1n50["loo_cosine_mean"],3), fnum(r1n50["nn_dist_cv"])],
     ["Large", 100, r1n100["tfidf_nonzero_cols"], r1n100["map_time_sec"], r1n100["loo_time_sec"],
      round(r1n100["peak_mem_mb"]), fnum(r1n100["stress"]), fnum(r1n100["spearman"]),
      fnum(r1n100["loo_cosine_mean"],3), fnum(r1n100["nn_dist_cv"])]]) + f"""
LOO再構成 cosine（global h）: N=20 {fnum(r1n20['loo_cosine_mean'],3)}±{fnum(r1n20['loo_cosine_std'],3)} → N=50 {fnum(r1n50['loo_cosine_mean'],3)}±{fnum(r1n50['loo_cosine_std'],3)} → N=100 {fnum(r1n100['loo_cosine_mean'],3)}±{fnum(r1n100['loo_cosine_std'],3)}

## 図
![scaling](appendix/figures/Appendix_S1_scaling.png)
![thumbnails](appendix/figures/Appendix_S2_map_thumbnails.png)

## 考察
- **計算時間**: マップ最適化（scipy L-BFGS-B）は 4.7s → 5.2s → 5.3s（N=20→50→100）とこの規模ではほぼ固定費支配。一方 LOO は 0.06s → 0.28s → 1.07s とほぼ O(N²) で伸びており、大規模化のボトルネックはペア距離系の処理にある。N=1000 級では §8.2 のとおり近似kNN・局所SPH・近似GPRへの切替が必須になる。
- **メモリ**: 410 → 508 → 699 MB。TF-IDF 密行列（100×250,000 float32 ≈ 100MB）とその派生（L1版・Gram）が支配的で、大規模化には sparse 表現が必須。
- **距離保存**: stress² は 0.124 → 0.141 → 0.148 と N とともに単調増加（自由度が減り埋め込みが難しくなる）。Spearman は N=20/50 でほぼ同じ（0.66）だが N=100 でやや低下（0.638）。順位保存は中規模まで頑健で、点数増による2次元の限界が N=100 付近から現れ始める。
- **LOO** は 0.418 → 0.462 → 0.465 と N とともに改善し N=50 でほぼ飽和。文献密度が上がるほど SPH 補間の再構成力が向上するという連続場仮説と整合的で、かつ N≈50 で近傍密度が有効水準に達することを示唆する。
- **最近傍距離CV** は N とともに増加。点数が増えるほど疎密の差が残る（=クラスタ構造が配置に現れる）。
- **次の一手**: N=300〜1000級コーパスでの検証、kNN 局所SPH・inducing point GPR の実装。
"""

# ================= R2 stability =================
sstress = [r["stress"] for r in seeds]
sproc = [r["procrustes_vs_main"] for r in seeds]
sari = [r["cluster_ari_vs_seed0"] for r in seeds]
sdev = [r["geodesic_deviation"] for r in seeds]
W["report_stability.md"] = HDR + f"""# R2: 計算安定性・感度解析報告書（§9 / E7）

## 目的
random seed・アンカー数・正則化係数・SPH平滑化長・GPRカーネル・multi-start・有限差分δに対する結果の安定性を定量化する。

## 目的関数の定義（v3.0 §2.4 厳密定義・scipy/scikit-learn純正実装）
- E_stress = Σ_(i<j)(‖y_i−y_j‖−t_ij)²/Σ t_ij²（t=κ·α·d、主目的）
- E_rep = 2/(N(N−1)) Σ exp(−r²/(2σ²))、σ=0.35
- E_center = mean_x² + mean_y²
- E_cover = mean_c min_i‖y_i−c‖²、C=12×12全域格子[−0.92,0.92]
- E_edge = mean_free[exp(−(x+1)/β)+exp(−(1−x)/β)+exp(−(y+1)/β)+exp(−(1−y)/β)]、β=0.10
- E_angle = mean_t max(θ_target−θ_min,0)²/θ_target²（θ_target=30°）、E_area = Var_t[log area]（Delaunay毎反復再構成）
- α = 平均アンカー間2D距離/平均cos距離（外周のみ、v5.0 §3.3）。代表選定=cos距離総和最大（貪欲+局所スワップ）
- 初期値: classical MDS + sklearn SMACOF(n_init=1, random_state=0) + アンカーへのアフィン整合（決定論的）。最適化: scipy L-BFGS-B（ハードbox制約, maxiter=2000, ftol=1e-12）
- seed>0 は初期値に N(0,0.05) ジッターを付与した感度評価（v3.0の初期値自体は決定論的）

## 数値表

### (a) random seed（0,1,2,3,4,5,10,20; 1200反復）
""" + tbl(["seed", "stress", "kNN保存(k=7)", "cluster ARI vs seed0", "Procrustes vs 主マップ", "測地線偏差"],
    [[r["seed"], fnum(r["stress"]), fnum(r["knn_preservation_k7"], 3), fnum(r["cluster_ari_vs_seed0"], 3),
      fnum(r["procrustes_vs_main"], 3), fnum(r["geodesic_deviation"], 3)] for r in seeds]) + \
f"""平均±SD: stress {fnum(np.mean(sstress))}±{fnum(np.std(sstress),5)} / Procrustes {fnum(np.mean(sproc),3)}±{fnum(np.std(sproc),3)} / ARI {fnum(np.mean(sari),3)}±{fnum(np.std(sari),3)} / 測地線偏差 {fnum(np.mean(sdev),3)}±{fnum(np.std(sdev),3)}

### (b) アンカー数（外周4 vs 8＋中央）
""" + tbl(["外周アンカー", "alpha", "stress", "Procrustes vs 主", "cluster ARI", "kNN保存(k=7)"],
    [[r["n_perim"], fnum(r["alpha"], 3), fnum(r["stress"]), fnum(r["procrustes_vs_main"], 3),
      fnum(r["cluster_ari_vs_main"], 3), fnum(r["knn_preservation_k7"], 3)] for r in anchors]) + \
"""注: 外周3,5,6アンカーは本仕様のアンカー配置定義（四隅/四隅+辺中点）に存在しないため未実施（理由明記）。

### (c) 正則化係数 0.25–4×（λ_rep, λ_cover, λ_center）
""" + tbl(["係数", "倍率", "stress", "spread(座標SD)", "境界集中率(|p|>0.9)"],
    [[r["param"], r["factor"], fnum(r["stress"]), fnum(r["spread"], 3), fnum(r["boundary_crowding"], 2)]
     for r in regs]) + \
"""### (d) SPH平滑化長
""" + tbl(["h方式", "LOO cosine", "平均エントロピー", "経路cos平滑度", "経路最大ジャンプ"],
    [[f"{r['h_mode']}(k={r['knn_k']})" if "adaptive" in r["h_mode"] else r["h_mode"],
      f"{fnum(r['loo_cosine_mean'],3)}±{fnum(r['loo_cosine_std'],3)}", fnum(r["mean_entropy"], 3),
      fnum(r["path_cos_smoothness"], 5), fnum(r["path_max_jump"], 5)] for r in sphh]) + \
"""### (e) GPRカーネル
""" + tbl(["kernel", "White", "LML", "LOO RMSE(潜在)", "較正相関corr(|err|,σ)", "z-score SD"],
    [[r["kernel"], r["white"], fnum(r["lml"], 1), fnum(r["loo_rmse_latent"], 3),
      fnum(r["calibration_corr"], 3) if r["calibration_corr"] is not None else "—",
      fnum(r["z_std"], 3)] for r in gprk]) + \
f"""### (f) multi-start（7候補×30測地線）
- 全30測地線で正式成功（success かつ E_final ≤ E_straight）: **True**
- 開始点あたり収束率: **100%**
- 候補間エネルギー相対幅の中央値: {fnum(np.median([r['E_spread_rel'] for r in ms]),4)}（最良候補はほぼ常に a=0 近傍）

### (g) 有限差分δ
""" + tbl(["δ", "平均条件数", "最大条件数", "g相対偏差(vs 1e-3)平均", "最大"],
    [[r["delta"], fnum(r["mean_cond"], 3), fnum(r["max_cond"], 3),
      f"{r['mean_rel_dev_vs_1e-3']:.2e}", f"{r['max_rel_dev_vs_1e-3']:.2e}"] for r in dlt]) + \
"""
## 図
![stability](appendix/figures/Appendix_S3_stability.png)

## 考察
- **stress はseedにほぼ不感**（0.1473–0.1515）。一方 **配置は初期値ジッターに依存**（Procrustes 0.24–0.39）で、複数のほぼ等価な局所解が存在する。ただしv3.0の規定初期値（MDS+SMACOF）は決定論的なので、規定どおり実行すれば再現は一意（seed0はProcrustes≈0）。「距離構造は安定、絶対配置は初期値依存」と明記すべき。
- **アンカー8+中央は4+中央より明確に優位**（stress 0.148 vs 0.230）。v3.1 の N≥16 → 8アンカー規則を支持。
- **正則化係数は0.25–4×でstressへの影響が小さい**（0.147–0.151）。主目的E_stressが支配的で、補助項は配置様式（spread・境界率）を微調整する。頑健性としては好ましい。
- **SPH h**: global h はLOO最良（0.465）だがエントロピーが0.90と高く場が過剰平滑。knn_adaptive(k=8) はLOOをほぼ保ちつつ（0.457）局所構造を保存（エントロピー0.56）しており、N=100–300では §8.2 のとおり adaptive を推奨。
- **GPRカーネル**（sklearn GaussianProcessRegressor, n_restarts=10）: Matern3/2 がLML最良（−856 vs RBF −865）だが LOO RMSEはほぼ同等。WhiteKernel除去は長さスケールが下限に潰れ LML −1450 と破綻し、White項は必須。z-score SD≈1.0 で不確かさの絶対水準は良く較正されている。
- **δ=1e-3 は安定領域の中央**にあり（1e-2〜1e-4で相対偏差≤1.1%、条件数≈2.9で不変）、計量の数値微分は頑健。
- **限界**: seed感度はマップ最適化のみで、TF-IDF・GPR再学習を含む全パイプライン反復ではない。次の一手として全再実行型の安定性試験を推奨。
"""

# ================= R3 metric comparison =================
mrows = []
for r in metrics:
    mrows.append([r["pair"], r["method"], f'{float(r["sph_energy"]):.4g}', f'{float(r["sph_length"]):.3f}',
                  f'{float(r["energy_reduction"]):+.3f}', f'{float(r["cos_smoothness"]):.2e}',
                  f'{float(r["max_semantic_jump"]):.2e}', f'{float(r["mean_gpr_uncertainty"]):.3f}',
                  f'{float(r["mean_sph_entropy"]):.3f}', f'{float(r["nearest_doc_dist"]):.3f}',
                  f'{float(r["js_smoothness"]):.2e}', f'{float(r["hellinger_smoothness"]):.2e}'])
lam_tbl = [[r["pair"], r["lambda"], fnum(r["path_deviation_from_lambda0"], 4), fnum(r["max_deviation"], 4)]
           for r in lamdev]
uncert = {}
for r in metrics:
    uncert.setdefault(r["pair"], {})[r["method"]] = float(r["mean_gpr_uncertainty"])
lam_u = [[p, fnum(uncert[p]["sph"], 4), fnum(uncert[p].get("gpr1", np.nan), 4),
          fnum(uncert[p].get("gpr4", np.nan), 4), fnum(uncert[p].get("gpr9", np.nan), 4)] for p in pairs]
W["report_metric_comparison.md"] = HDR + f"""# R3: 計量比較報告書（§5, §6 / E3, E4）

## 目的
5種類の経路（Euclidean line / kNN graph / SPH / GPR不確かさ重み付き(λ=1,4,9) / Fisher-Rao）を、同一の経路評価指標（§6.2）で比較する。

## 方法
- endpoint pairs は §7.3 の5類型で決定論的に選定: {json.dumps(pairs)}
- 測地線: 31点離散・7候補multi-start・L-BFGS（maxiter 2000, ftol 1e-12）。採用規則は §6.1/9.2 準拠。
- kNNグラフ: cosine距離 k=5 で連結（Dijkstra）。
- 全経路を共通のSPH誘導計量で評価（sph_energy/sph_length/energy_reduction）。
- 収束結果: **30/30 測地線が正式成功**（geodesic_results.csv、特異正則化 g+εI 適用点 0）。

## 数値表（全ペア×全経路×9指標; results_metrics.csv 全文は成果物参照）
""" + tbl(["pair", "method", "E_g", "L_g", "R_E", "cos平滑", "max jump", "mean u", "mean H", "最近文書距離", "JS平滑", "Hel平滑"], mrows) + \
"""### λ感度（E4）: λ=0基準の経路偏差
""" + tbl(["pair", "λ", "平均偏差", "最大偏差"], lam_tbl) + \
"""### λ感度（E4）: 経路平均不確かさの単調減少
""" + tbl(["pair", "u(λ=0)", "u(λ=1)", "u(λ=4)", "u(λ=9)"], lam_u) + \
"""
## 図
![Figure G](figures/Figure_G_path_comparison.png)
![Figure H](figures/Figure_H_lambda_sensitivity.png)
![Figure I](figures/Figure_I_path_profile.png)
![Appendix S7: 中央→四隅測地線検証](appendix/figures/Appendix_S7_regression_center_corners.png)

## 考察
- **SPH測地線は全ペアで直線よりエネルギーを1.5–32%削減**（R_E=0.015〜0.32、遠距離ペアほど大）し、cosine平滑度・最大ジャンプも直線以下。さらにv3.1参照プロトコル（中央→四隅4本, Appendix S7）では低下率29.0–38.3%・全本successで、v3.1の参照実測値15.6–29.5%（N=23）と整合。連続場上の測地線が「概念遷移の滑らかさ」を実際に改善している。
- **graph経路はエネルギー基準で最も高コスト**（R_E 最小−3.1）だが、実在文献を確実に経由する唯一の経路であり「説明可能な飛び石」という役割は数値でも明瞭に分離された。近距離ペアではgraphと直線がほぼ一致し、遠距離で乖離が大きくなる（§7.3の予想どおり）。
- **GPR重み付き計量の効果は経路の初期不確かさに依存する**。λ=0経路が高不確かさ領域を通るペアでは平均uがλに単調減少（high_uncertainty: 0.219→0.211, near_intra: 0.218→0.205, far_centroids: 0.189→0.188）する一方、既に低不確かさ経路のペア（mid_adjacent, far_max_cos, cross_max_2d）では変化が10⁻⁴オーダーで実質不変〜微増となる。「不確かさが高いときだけ迂回する」という §5.1 の設計意図どおりの選択的挙動であり、経路偏差もλに単調（high_uncertainty: 0.010→0.046）。
- **FR測地線はSPH測地線と異なる経路を取る**ペアがあり（far_max_cos, far_centroids）、自計量（情報幾何）でのエネルギーはSPH経路より大幅に低い。L2方向幾何とL1配分幾何が異なる「近さ」を測っている証拠であり、複数計量比較の価値を支持する。
- **限界**: 経路類型は6ペアの代表例であり、全ペア網羅ではない（Medium規模の代表ペア方針、§8.2）。エントロピーはglobal hのため高値側に圧縮されており、ペア間差は小さい。
"""

# ================= R4 normalization =================
sel = {}
for r in metrics:
    if r["method"] in ("line", "sph", "fr"):
        sel.setdefault(r["pair"], {})[r["method"]] = r
r4rows = []
for p, d in sel.items():
    for m in ("line", "sph", "fr"):
        r = d[m]
        r4rows.append([p, m, f'{float(r["cos_smoothness"]):.2e}', f'{float(r["sph_length"]):.3f}',
                       f'{float(r["js_smoothness"]):.2e}', f'{float(r["hellinger_smoothness"]):.2e}',
                       f'{float(r["fisher_rao_length"]):.3f}'])
W["report_normalization_L2_L1.md"] = HDR + f"""# R4: L2/L1 正規化比較報告書（§2 / E5）

## 目的
L2正規化（単位球面上の方向幾何）と L1正規化（確率単体上の情報幾何）が異なる概念遷移を捉えるかを、同一 endpoint pair で対照する。

## 方法
- L2版: x/‖x‖₂ → cosine系指標（cos平滑度・SPH Riemann長）。
- L1版: (x+ε)/Σ(x+ε), ε={tfc['epsilon_l1_smoothing']} → JS・Hellinger・Fisher-Rao。
- 両版とも同一のTF-IDF行列（char_wb 4–7gram, 250,000特徴）から構築し保存（§2の並置方針）。
- 実データ確認（§2.2相当）: 全100行の行和は1（float64検証、r5_epsilon_sensitivity.json）。文書0–1間: JS divergence {fnum(tfc['example_doc0_doc1']['js_divergence'],3)}, Hellinger {fnum(tfc['example_doc0_doc1']['hellinger'],3)}, Fisher-Rao {fnum(tfc['example_doc0_doc1']['fisher_rao'],3)}。

## 数値表: 同一ペアのL2幾何とL1情報幾何の対照（line / SPH測地線 / FR測地線）
""" + tbl(["pair", "path", "L2: cos平滑", "L2: SPH長", "L1: JS平滑", "L1: Hellinger平滑", "L1: FR長"], r4rows) + \
"""
## 図
![Figure C](figures/Figure_C_L2_vs_L1_schematic.png)
![Appendix S4](appendix/figures/Appendix_S4_L2_vs_FR_geodesics.png)

## 考察
- L2系の平滑度とL1系の平滑度は**強く相関するが同一ではない**。SPH（L2最適）経路とFR（L1最適）経路は近距離ペアでほぼ一致し、遠距離ペア（far_max_cos, far_centroids）で分岐する。方向の類似性（トピックの向き）と特徴量配分の形状（語彙分布）が異なる遷移コストを与えるためである。
- FR経路は自身の指標（FR長・JS平滑）でSPH経路よりわずかに悪化する場合もあるが（離散化・部分語彙の影響）、own metricのエネルギーでは常に直線を改善しており、L1情報幾何の測地線としての妥当性を確認した。
- どちらが正しいかではなく**異なる問いに答える幾何**（§2）という設計思想を、同一ペア対照表として定量化できた。
- **限界**: TF-IDFのL1化は「特徴量質量の相対配分」であり語の生成確率ではない（§2.1）。FR計量はtop-2000特徴部分集合で計算しており（§5.2の許容する安定化）、全語彙FRとは数値が異なりうる。
"""

# ================= R5 information geometry =================
er = eps["pair_docs_0_1"]
ep_rows = [[e, fnum(er[e]["KL(0||1)_ref"], 3), fnum(er[e]["JS_div"], 6), fnum(er[e]["Hellinger"], 6),
            fnum(er[e]["FisherRao"], 6)] for e in er]
pr = eps["path_far_max_cos_sph_step_means"]
pr_rows = [[e, fnum(pr[e]["KL_step_mean_ref"], 6), fnum(pr[e]["JS_step_mean"], 6),
            fnum(pr[e]["Hellinger_step_mean"], 6), fnum(pr[e]["FR_length"], 4)] for e in pr]
sc = eps["simplex_check"]
W["report_information_geometry.md"] = HDR + f"""# R5: 情報幾何報告書（§2, §6 / E5）

## 目的
L1確率単体上の情報幾何量（KL・JS・Hellinger・Fisher-Rao）の計算可能性・ε感度・Σπ=1検証を文書化する。

## 方法
- π = (x+ε)/Σ(x+ε)。KLは非対称・ゼロ成分に弱いため参考値、主評価はJS・Hellinger（§2.1）。Fisher-Rao は √π 表現 d_FR=2 arccos Σ√(π_k ρ_k) で安定化。
- ε ∈ {{1e-12, 1e-10, 1e-8, 1e-6}} の感度解析（代表: 文書0–1、および far_max_cos SPH測地線のステップ平均）。

## 数値表

### (a) Σπ=1 検証（float64、全ε）
""" + tbl(["ε", "行和 min", "行和 max"],
    [[e, f'{sc[e]["rowsum_min"]:.15f}', f'{sc[e]["rowsum_max"]:.15f}'] for e in sc]) + \
"""### (b) ε感度: 文書0–1間距離
""" + tbl(["ε", "KL(0‖1) 参考", "JS div", "Hellinger", "Fisher-Rao"], ep_rows) + \
"""### (c) ε感度: far_max_cos SPH測地線のステップ量
""" + tbl(["ε", "KLステップ平均(参考)", "JSステップ平均", "Hellingerステップ平均", "FR長"], pr_rows) + \
"""
## 図
![Appendix S5](appendix/figures/Appendix_S5_sqrtpi_sphere.png)
![Figure I](figures/Figure_I_path_profile.png)

## 考察
- **JS・Hellinger・Fisher-Rao は ε に対して6桁レベルで不変**（1e-12〜1e-6 で相対変化 <0.1%）。一方 **KL は ε に強く依存**（17.6→9.3）し、ゼロ成分への感受性が数値で裏付けられた。主評価にJS・Hellingerを置きKLを参考値とする §2.1 の方針は正当。
- Fisher-Rao は √π 表現で全ペア・全経路点において数値破綻なく計算できた（arccos引数のクリップ発生も僅少）。高次元sparse TF-IDFでの g_FR 直接計算の不安定性は、経路計量にはtop-2000特徴部分集合＋√π表現の併用で回避した。
- SPH場の混合 π(r)=Σ w_i π_i は常に確率単体上に留まり（行和=1、非負）、情報幾何量が経路全体で well-defined。
- **限界**: εは全成分一様加算であり、頻度ゼロの意味論を持たない char n-gram には解釈上の含意が薄い。FR測地線の計量は部分語彙で構成しており、報告値のFR長（全語彙）とは基底が異なる。
"""

# ================= R6 evaluation =================
cs = np.array(e2g["per_doc_cosine"])
order = np.argsort(cs)
meta = list(csv.DictReader(open(OUT + "/corpus_metadata.csv")))
worst = [[int(d), fnum(cs[d], 3), meta[d]["title"][:60]] for d in order[:5]]
best = [[int(d), fnum(cs[d], 3), meta[d]["title"][:60]] for d in order[-5:]]
W["report_evaluation_metrics.md"] = HDR + f"""# R6: 評価指標報告書（§7 / E1, E2）

## 目的
E1: 2Dマップが元のTF-IDF cosine距離構造をどれだけ保存するか。E2: SPH補間が実在文献を再構成できるか（leave-one-out）。

## E1: 距離保存性 数値表
""" + tbl(["指標", "値"],
    [["normalized stress（目標距離 t=κ·α·d 基準）", fnum(e1["stress_vs_target"])],
     ["Spearman ρ（cos距離 vs 2D距離, 全4950ペア）", fnum(e1["spearman"])],
     ["Pearson r", fnum(e1["pearson"])],
     ["trustworthiness (k=5/7/10)", f'{fnum(e1["trustworthiness_k5"],3)} / {fnum(e1["trustworthiness_k7"],3)} / {fnum(e1["trustworthiness_k10"],3)}'],
     ["continuity (k=5/7/10)", f'{fnum(e1["continuity_k5"],3)} / {fnum(e1["continuity_k7"],3)} / {fnum(e1["continuity_k10"],3)}'],
     ["kNN preservation (k=5/7/10)", f'{fnum(e1["knn_preservation_k5"],3)} / {fnum(e1["knn_preservation_k7"],3)} / {fnum(e1["knn_preservation_k10"],3)}'],
     ["最近傍距離CV", fnum(e1["nn_dist_cv"], 3)]]) + \
f"""## E2: leave-one-out 再構成 数値表（N=100全件）
""" + tbl(["指標", "global h", "kNN adaptive h (k=8)"],
    [["LOO cosine 平均±SD", f'{fnum(e2g["loo_cosine_mean"],3)}±{fnum(e2g["loo_cosine_std"],3)}',
      f'{fnum(e2a["loo_cosine_mean"],3)}±{fnum(e2a["loo_cosine_std"],3)}'],
     ["LOO cosine 中央値 / 最小 / 最大", f'{fnum(e2g["loo_cosine_median"],3)} / {fnum(e2g["loo_cosine_min"],3)} / {fnum(e2g["loo_cosine_max"],3)}', "—"],
     ["top-20特徴回復率", fnum(e2g["topk_recovery_mean"], 3), fnum(e2a["topk_recovery_mean"], 3)],
     ["top-20特徴回復率（df≥2の共有特徴に限定）", fnum(e2g["topk_recovery_shared_df2_mean"], 3), fnum(e2a["topk_recovery_shared_df2_mean"], 3)],
     ["最近接論文回復率（top-1一致）", fnum(e2g["nearest_paper_recovery_rate"], 2), fnum(e2a["nearest_paper_recovery_rate"], 2)],
     ["最近接論文がtop-5に含まれる率", fnum(e2g["nearest_paper_in_top5_rate"], 2), fnum(e2a["nearest_paper_in_top5_rate"], 2)]]) + \
"""### 再構成 良/不良の代表例（global h）
""" + tbl(["doc_id", "LOO cosine", "title（worst 5）"], worst) + tbl(["doc_id", "LOO cosine", "title（best 5）"], best) + \
"""
## 図
![Appendix S6](appendix/figures/Appendix_S6_evaluation.png)

## 考察
- **E1**: stress 0.148（κ=1元スケール0.163）・Spearman 0.64・trustworthiness≈0.81 は「大域構造は中程度に保存、局所近傍は部分保存（kNN保存≈0.41）」を示す。v3.0厳密定義への切替（kNN保存@7=0.42）。残る歪みは、目標距離平均(≈1.45)が[-1,1]²の収容力(≈1.05)を超えるというchar n-gram cos距離の高次元集中に由来する構造的トレードオフ。
- **E2 LOO cosine 0.465±0.059** は、ランダム2文書間の平均cosine類似（≈0.19、cos距離行列平均0.81から）を明確に上回り、位置情報からの補間に予測力があることを示す。ただし N=23 時代の報告値より低く、文書数増と2D自由度の限界が影響。
- **top-k特徴回復が0**なのは重要な負の結果である。min_df=1・char n-gramでは各文書のtop特徴が文書固有のレアn-gram（高idf）に占められ、他文書の凸結合からは原理的に回復不可能。df≥2 に限定しても0であり、TF-IDF上位特徴の回復には (i) min_df≥2 での語彙構築、(ii) 回復対象をidf重みなしtf上位に変える、等の再設計が必要（次の一手）。
- **最近接論文回復はtop-1で6–7%、top-5で22%（global）/53%（adaptive）**。SPH再構成ベクトルの最近傍は2Dマップ上の近傍文書に引かれるため、高次元での真の最近傍とずれる。adaptive h はtop-5回復を大きく改善しており、局所化された場が検索的タスクに有利。
- **総合**: 「連続な意味場としての補間には中程度の予測力があるが、文書固有特徴の復元・最近傍検索の代替としては使えない」というのが本コーパスでの誠実な結論。フルペーパーでは E2 の主張を cosine 再構成に限定することを推奨する。
"""

# ================= R0 overall =================
gsum = {}
W["analysis_report.md"] = HDR + f"""# R0: 統括報告書 — Knowledge Manifold v5.0 解析（N=100）

## 1. 概要
本報告書は統合仕様 v5.0 に基づき、炭素繊維複合材料分野を中心とする100本の論文コーパス（100papers.md）に対して実施した知識多様体解析の統括である。個別詳細は R1–R6 を参照。

- コーパス: 100文書、クリーニング後 5,683,556 文字（extraction_log.json）
- TF-IDF: char_wb 4–7gram, sublinear tf, 語彙 662,050 → max_features 250,000, min_df=1, L2版・L1版(ε=1e-10)を並置保存
- 2Dマップ: 外周8＋中央1アンカー, v3.0 §2.4厳密定義, κ=0.90, margin=0.04, α={fnum(man['alpha'],4)}（外周のみ・平均比; 中央込みだと {fnum(man['alpha_if_central_included'],4)}, −6.7%）, 最終J={fnum(man['final_J'],4)}
- 監査: post_optimization_global_scaling=false, アンカー固定=true, 自由点最大半径 before==after（{fnum(man['free_point_max_radius_before_finalization'],4)}）→ **一括スケーリング検査 PASS**
- 連続場: SPH(cubic spline, global h)・SPHエントロピー・GPR(RBF ARD+White, SVD10次元潜在, n_restarts=10)
- 測地線: 6 endpoint pairs × 5計量（line/graph/SPH/GPR-w λ=1,4,9/FR）= 30本、**全て正式成功**

## 2. 主要結果（要約）
| 検証 | 報告書 | 主要数値 | 結論 |
|---|---|---|---|
| データサイズ | R1 | N=20/50/100 (Small/Medium/Large) | LOOはN増で改善しN=50で飽和。O(N²)項が大規模化の壁 |
| 安定性 | R2 | stress 0.147–0.152(seed/正則化), Procrustes 0.24–0.39(ジッター時) | 距離構造は安定・規定初期値なら決定論的に再現。δ・正則化に頑健 |
| 計量比較 | R3 | SPH測地線 R_E=+1.5〜32%, 中央→四隅で29–38%(v3.1参照値と整合), graphは高コスト | 各計量の役割分担が明瞭に分離 |
| λ感度 | R3 | 高不確かさ経路で平均u: 0.219→0.211 (λ0→9), 偏差単調増 | 不確かさ回避の選択的挙動を再現 |
| L2/L1 | R4 | SPH経路とFR経路が遠距離で分岐 | 方向幾何と配分幾何は異なる遷移を捉える |
| 情報幾何 | R5 | JS/Hel/FRはε不変, KLはε依存(17.6→9.3) | KL格下げ・JS/Hellinger主評価の方針を実証 |
| 距離保存・LOO | R6 | stress 0.148, Spearman 0.64, LOO cos 0.465 | 補間に予測力あり。top-k特徴回復は0（負の結果） |

## 3. 未実施項目とその理由（仕様 §10.1 の義務記載）
- **E8 専門家評価**: 人間の分野専門家によるLikert評価が必要であり、本自動解析では実施不可。評価用の材料（Figure A/G/H, 方向ベクトル言語化 knowledge_gradient.json）は整備済み。
- **E6（3D比較）**: 仕様 v5.0 自体が対象外と規定。
- **N=1000+ (Large)**: データ非提供のため未実施（R1に明記）。
- **アンカー数 3/5/6**: 本仕様のアンカー配置定義に存在しないため 4/8 のみ実施（R2に明記）。

## 4. 仕様からの逸脱（透明性のための明記）
""" + "".join(f"- {d}\n" for d in man["deviations_from_spec"]) + f"""
## 5. 全体考察
1. **中心主張の検証状況**: 「科学文献を補間・勾配・不確かさ・測地線を計算できる連続的意味場として扱える」という §0.1 の主張は、(i) LOO補間の予測力（R6）、(ii) 測地線の系統的なエネルギー削減と滑らかさ改善・v3.1参照値との整合（R3）、(iii) 較正されたGPR不確かさ（R2(e)）、(iv) 4象限での勾配方向の言語化（crack/fracture系 ↔ 分子・化学系 ↔ fiber/laminate系, knowledge_gradient.json）により、N=100 で定量的に支持された。
2. **複数計量比較の価値**: 単一計量では見えない役割分担（graph=説明可能性、SPH=滑らかさ、GPR-w=信頼性、FR=配分変化）が同一指標表（results_metrics.csv）上で分離できた。これは §0.2 の設計の中核的成果。
3. **最重要の限界**: (a) 2D埋め込みの局所近傍保存は kNN保存≈0.41 と部分的で、局所的な結論（特定文書の隣接関係）には2Dマップを使うべきでない。(b) top-k特徴回復0という負の結果は、char n-gram + min_df=1 の語彙設計と補間の相性問題を示す。(c) global h は場を過剰平滑化しており、主解析を knn_adaptive に切り替える価値がある。
4. **次の一手**: min_df≥2 語彙での再実験、adaptive h の主解析化、N=300級コーパスでのクラスタ別評価、専門家評価（E8）の実施、全パイプラインseed反復。

## 6. 成果物一覧
manifest.json（監査フィールド§9.5完備）, corpus_metadata.csv, extraction_log.json, tfidf_config.json, tfidf_vocab.json, coordinates_2d.csv, sph_config.json, gpr_info.json, endpoint_pairs.json, knowledge_gradient.json, geodesic_results.csv, results_metrics.csv, e1/e2/e4/e7/r5 各JSON, figures/（仕様§11準拠: Figure A,C,D,E,F,G,H,I）, tables/（Table A,B,C）, appendix/（補助図S1–S7）, 報告書 R1–R6。

R1–R6 はいずれも欠落なく生成済み（E8とLarge段階は上記の理由により対象外/未実施として明記）。
"""

for name, content in W.items():
    open(OUT + "/" + name, "w", encoding="utf-8").write(content)
    print("wrote", name, len(content), "chars")
