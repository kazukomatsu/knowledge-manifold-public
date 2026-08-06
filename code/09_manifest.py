# -*- coding: utf-8 -*-
"""§1.2/§9.3/§9.5 manifest.json 作成 (監査フィールド必須) + sph_config.json"""
import json, os, sys, hashlib, platform
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import OUT, DATA

def sha(f):
    return hashlib.sha256(open(f, "rb").read()).hexdigest()

audit = json.load(open(OUT + "/map_audit.json"))
tf = json.load(open(OUT + "/tfidf_config.json"))
gpr = json.load(open(OUT + "/gpr_info.json"))
ext = json.load(open(OUT + "/extraction_log.json"))

sph_config = {"kernel": "cubic_spline_2d", "h_mode_main": "global(max_dist/1.98)",
              "h_modes_sensitivity": ["global", "fixed", "knn_adaptive", "density_adaptive"],
              "finite_difference_delta": 1e-3, "delta_sensitivity": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]}
json.dump(sph_config, open(OUT + "/sph_config.json", "w"), indent=1)

import numpy
blas = "unknown"
try:
    cfg = numpy.show_config(mode="dicts")
    blas = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
except Exception:
    pass

manifest = {
  "project": "Knowledge Manifold v5.0 analysis",
  "spec": "2026_06_24 KnowledgeManifold_統合仕様_v5.0 (v4.0+v4.1統合)",
  "date": "2026-07-06",
  "N": ext["n_docs"],
  "input": {"source": ext.get("source", "derived input"),
            "source_sha256": ext.get("source_sha256",
                ext.get("freeze_manifest_hashes", {}).get("source_raw_file_hash", "n/a")),
            "derivation_id": ext.get("derivation_id"),
            "extraction": ext["extraction_method"], "ocr": False,
            "failed_pages": ext["failed_pages"]},
  "environment": {
      "python": platform.python_version(), "numpy": np.__version__,
      "blas_lapack": blas, "platform": platform.platform(),
      "scipy": __import__("scipy").__version__, "sklearn": __import__("sklearn").__version__,
      "note": "scipy/scikit-learn installed from user-provided wheels; spec-native implementations (sklearn TfidfVectorizer / scipy L-BFGS-B / scipy Delaunay / sklearn GPR n_restarts=10 / sklearn SMACOF)",
      "matplotlib": __import__("matplotlib").__version__},
  "tfidf": {k: tf[k] for k in ["analyzer", "ngram_range", "sublinear_tf", "smooth_idf",
            "max_features", "min_df", "norm", "vocab_size_full", "vocab_size",
            "epsilon_l1_smoothing", "implementation"]},
  "normalization": ["l2", "l1"],
  "embedding": {"dims": [2], "anchor": "perimeter8+central", "random_seed_main": 0,
                "seeds_sensitivity": [0, 1, 2, 3, 4, 5, 10, 20],
                "optimizer": audit["optimizer"],
                "lbfgs_total_iterations": audit.get("lbfgs_total_iterations"),
                "init": audit.get("init"),
                "objective": "J = E_stress + λ_rep E_rep + λ_center E_center + λ_cover E_cover + λ_edge E_edge + λ_angle E_angle + λ_area E_area",
                "aux_term_definitions_note": "v3.0 §2.4 の厳密定義を使用 (E_rep: σ=0.35 Gauss / E_cover: 12×12格子[-0.92,0.92] / E_edge: 指数壁 β=0.10 / E_angle: θ_target=30° / E_area: Var[log area]; v3.0受領 2026-07-06)",
                "alpha_definition": audit.get("alpha_definition"),
                "rep_selection": audit.get("rep_selection"),
                "stress_kappa1_original_scale": audit.get("stress_kappa1_original_scale")},
  "sph": sph_config,
  "gpr": gpr,
  "metrics": ["euclidean", "cosine", "graph", "sph", "gpr_weighted", "js", "hellinger", "fisher_rao"],
  "lambda_values": [0, 1, 4, 9],
  "geodesic": {"n_points": 31, "n_inner": 29,
               "multistart_amplitudes": [0, 0.01, -0.01, 0.03, -0.03, 0.05, -0.05],
               "optimizer": "custom L-BFGS (two-loop, Armijo backtracking; scipy L-BFGS-B unavailable)",
               "maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50,
               "eps_singular": 1e-10,
               "fr_top_feature_subset": 2000},
  "random_state_all": 0,
  # ---- §9.5 必須監査フィールド ----
  "post_optimization_global_scaling": audit["post_optimization_global_scaling"],
  "anchor_coordinates_fixed": audit["anchor_coordinates_fixed"],
  "kappa": audit["kappa"], "margin": audit["margin"],
  "lambda_cover": audit["lambda_cover"], "lambda_edge": audit["lambda_edge"],
  "lambda_angle": audit["lambda_angle"], "lambda_area": audit["lambda_area"],
  "lambda_rep": audit["lambda_rep"], "lambda_center": audit["lambda_center"],
  "alpha": audit["alpha"], "alpha_uses_central_anchor": False,
  "alpha_if_central_included": audit["alpha_if_central_included"],
  "free_point_max_radius_before_finalization": audit["free_point_max_radius_before_finalization"],
  "free_point_max_radius_after_finalization": audit["free_point_max_radius_after_finalization"],
  "anchor_positions_before_finalization": audit["anchor_positions_before_finalization"],
  "anchor_positions_after_finalization": audit["anchor_positions_after_finalization"],
  "optimizer_success": audit["optimizer_success"], "final_J": audit["final_J"],
  "random_seed": audit["random_seed"], "mds_init_hash": audit["mds_init_hash"],
  "scaling_autocheck": ("PASS: before==after and anchors fixed"
      if (audit["free_point_max_radius_before_finalization"] ==
          audit["free_point_max_radius_after_finalization"] and audit["anchor_coordinates_fixed"])
      else "FAIL"),
  "artifact_hashes": {os.path.basename(f): sha(f) for f in
      [OUT + "/coordinates_2d.csv", OUT + "/corpus_metadata.csv", OUT + "/results_metrics.csv",
       OUT + "/geodesic_results.csv", OUT + "/extraction_log.json"]},
  "deviations_from_spec": [
      "入力は論文PDFではなくユーザー提供のmarkdown変換ファイル(ユーザー指示)",
      "TF-IDFはsklearn TfidfVectorizer、最適化はscipy L-BFGS-B、DelaunayはScipy、GPRはsklearn(n_restarts=10)、SMACOFはsklearn (ユーザー提供wheelsで導入)",
      "マップ目的関数はv3.0 §2.2/2.3/2.4/2.5の厳密定義を使用。代表選定はC(100,9)全探索が非現実的なため貪欲+局所スワップ(v3.0が認める近似・方法記録済み)",
      "Dijkstra・kmeans・trustworthiness等の軽量アルゴリズムは検証済み自作実装を継続使用",
      "E8専門家評価は人間の専門家が必要なため未実施(R0に明記)",
      "N=1000+のLarge段階はデータ非提供のため未実施(R1に明記)",
      "anchor数感度は4/8のみ(3,5,6は本仕様のアンカー配置定義が存在しないため)",
      "Fisher-Rao測地線はtop-2000特徴部分集合で計算(§5.2の許容範囲)",
      "seed感度解析のseed>0は初期値へのN(0,0.05)ジッター付与(v3.0の初期値は決定論的なため感度評価用に導入・記録)"],
}
json.dump(manifest, open(OUT + "/manifest.json", "w"), indent=1, ensure_ascii=False)
h = sha(OUT + "/manifest.json")
open(OUT + "/manifest.sha256", "w").write(h)
print("manifest.json written, sha256 =", h)
