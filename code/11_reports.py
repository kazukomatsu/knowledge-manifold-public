# -*- coding: utf-8 -*-
"""Machine-written verification reports R0-R6 (spec §10).

Objective content only: every number below is read from the released JSON/CSV
artifacts of the run.  The 15 validation gates themselves are evaluated by
validate.py; report files R0-R6 and their manifest-hash headers are what
gates 14-15 check.
"""
import json, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import OUT, DATA

MH = open(OUT + "/manifest.sha256").read().strip()
N_DOCS = len(json.load(open(OUT + "/corpus/docs_clean.json")))
HDR = (f"> manifest.json sha256: `{MH}`  \n"
       f"> coordinates: `coordinates_2d.csv` (N={N_DOCS}, seed=0, 8 perimeter + central anchors)  \n"
       f"> machine-generated report; Knowledge Manifold spec v5.2\n\n")

def J(f): return json.load(open(OUT + "/" + f))

e1 = J("e1_distance_preservation.json")
e2g = J("e2_loo_global.json"); e2a = J("e2_loo_knn_adaptive.json")
seeds = J("e7_seeds.json"); anchors = J("e7_anchors.json"); regs = J("e7_regs.json")
sphh = J("e7_sph_h.json"); gprk = J("e7_gpr_kernels.json"); dlt = J("e7_delta.json")
ms = J("e7_multistart.json"); r1n20 = J("r1_N20.json")
eps = J("r5_epsilon_sensitivity.json"); lamdev = J("e4_lambda_path_deviation.json")
tfc = J("tfidf_config.json"); man = J("manifest.json")
metrics = [r for r in csv.DictReader(open(OUT + "/results_metrics.csv"))
           if not r["pair"].startswith("G")]
pairs = {k: v for k, v in json.load(open(DATA + "/pairs.json")).items()
         if not k.startswith("G")}

def fnum(x, d=4): return f"{float(x):.{d}f}"

def tbl(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s + "\n"

W = {}

# ================= R1 datasize =================
r1n50 = J("r1_N50.json")
import glob as _g
_nfull = max(int(f.split("_N")[1].split(".")[0]) for f in _g.glob(OUT + "/r1_N*.json"))
r1n100 = J(f"r1_N{_nfull}.json")
W["report_datasize.md"] = HDR + f"""# R1: corpus-size dependence

Nested subsets (seed 0) at N=20, N=50 and the full corpus (N={_nfull}).
The original Large tier of the spec (N=1000+) is out of scope for this corpus.

""" + tbl(
    ["tier", "N", "nonzero TF-IDF columns", "map time (s)", "LOO time (s)", "peak memory (MB)", "stress^2 (kappa scale)", "Spearman", "LOO cosine", "NN-distance CV"],
    [[t, r["N"], r["tfidf_nonzero_cols"], r["map_time_sec"], r["loo_time_sec"],
      round(r["peak_mem_mb"]), fnum(r["stress"]), fnum(r["spearman"]),
      fnum(r["loo_cosine_mean"], 3), fnum(r["nn_dist_cv"])]
     for t, r in [("Small", r1n20), ("Medium", r1n50), ("Full", r1n100)]]) + \
f"""LOO cosine (global h): N=20 {fnum(r1n20['loo_cosine_mean'],3)}±{fnum(r1n20['loo_cosine_std'],3)} / N=50 {fnum(r1n50['loo_cosine_mean'],3)}±{fnum(r1n50['loo_cosine_std'],3)} / N={_nfull} {fnum(r1n100['loo_cosine_mean'],3)}±{fnum(r1n100['loo_cosine_std'],3)}
"""

# ================= R2 stability =================
sstress = [r["stress"] for r in seeds]; sproc = [r["procrustes_vs_main"] for r in seeds]
sari = [r["cluster_ari_vs_seed0"] for r in seeds]; sdev = [r["geodesic_deviation"] for r in seeds]
W["report_stability.md"] = HDR + """# R2: stability and sensitivity (E8)

Settings: objective and optimizer per spec v5.2 (stress + repulsion + coverage +
edge terms; scipy L-BFGS-B, deterministic MDS/SMACOF initialization; seeds > 0
add N(0, 0.05) jitter to the initialization as a sensitivity probe).

## (a) initialization jitter (seeds 0,1,2,3,4,5,10,20)
""" + tbl(["seed", "stress", "kNN@7", "cluster ARI vs seed 0", "Procrustes vs main", "geodesic deviation"],
    [[r["seed"], fnum(r["stress"]), fnum(r["knn_preservation_k7"], 3), fnum(r["cluster_ari_vs_seed0"], 3),
      fnum(r["procrustes_vs_main"], 3), fnum(r["geodesic_deviation"], 3)] for r in seeds]) + \
f"""mean±SD: stress {fnum(np.mean(sstress))}±{fnum(np.std(sstress),5)} / Procrustes {fnum(np.mean(sproc),3)}±{fnum(np.std(sproc),3)} / ARI {fnum(np.mean(sari),3)}±{fnum(np.std(sari),3)} / geodesic deviation {fnum(np.mean(sdev),3)}±{fnum(np.std(sdev),3)}

## (b) perimeter anchors (4 corners vs 8 = corners + edge midpoints)
""" + tbl(["perimeter anchors", "alpha", "stress", "Procrustes vs main", "cluster ARI", "kNN@7"],
    [[r["n_perim"], fnum(r["alpha"], 3), fnum(r["stress"]), fnum(r["procrustes_vs_main"], 3),
      fnum(r["cluster_ari_vs_main"], 3), fnum(r["knn_preservation_k7"], 3)] for r in anchors]) + \
"""Note: 3/5/6 perimeter anchors are not defined by the anchor layout of the spec and are not run.

## (c) auxiliary-weight factors 0.25-4x (lambda_rep, lambda_cover, lambda_center)
""" + tbl(["parameter", "factor", "stress", "spread (coordinate SD)", "boundary occupancy (|p|>0.9)"],
    [[r["param"], r["factor"], fnum(r["stress"]), fnum(r["spread"], 3), fnum(r["boundary_crowding"], 2)]
     for r in regs]) + \
"""## (d) smoothing-length policies
""" + tbl(["policy", "LOO cosine", "mean entropy", "path cos smoothness", "path max jump"],
    [[f"{r['h_mode']}(k={r['knn_k']})" if "adaptive" in r["h_mode"] else r["h_mode"],
      f"{fnum(r['loo_cosine_mean'],3)}±{fnum(r['loo_cosine_std'],3)}", fnum(r["mean_entropy"], 3),
      fnum(r["path_cos_smoothness"], 5), fnum(r["path_max_jump"], 5)] for r in sphh]) + \
"""## (e) GPR kernels
""" + tbl(["kernel", "White term", "log marginal likelihood", "LOO RMSE (latent)", "corr(|err|, sigma)", "z-score SD"],
    [[r["kernel"], r["white"], fnum(r["lml"], 1), fnum(r["loo_rmse_latent"], 3),
      fnum(r["calibration_corr"], 3) if r["calibration_corr"] is not None else "-",
      fnum(r["z_std"], 3)] for r in gprk]) + \
f"""## (f) multi-start (7 candidates per geodesic)
- all adopted paths formally successful: {all(r['status']=='success' for r in ms)}
- median relative energy spread across candidates: {fnum(np.median([r['E_spread_rel'] for r in ms]),4)}

## (g) finite-difference scale delta
""" + tbl(["delta", "mean condition number", "max condition number", "mean rel. deviation vs 1e-3", "max"],
    [[r["delta"], fnum(r["mean_cond"], 3), fnum(r["max_cond"], 3),
      f"{r['mean_rel_dev_vs_1e-3']:.2e}", f"{r['max_rel_dev_vs_1e-3']:.2e}"] for r in dlt])

# ================= R3 metric comparison =================
mrows = [[r["pair"], r["method"], f'{float(r["sph_energy"]):.4g}', f'{float(r["sph_length"]):.3f}',
          f'{float(r["energy_reduction"]):+.3f}', f'{float(r["cos_smoothness"]):.2e}',
          f'{float(r["max_semantic_jump"]):.2e}', f'{float(r["mean_gpr_uncertainty"]):.3f}',
          f'{float(r["mean_sph_entropy"]):.3f}', f'{float(r["nearest_doc_dist"]):.3f}',
          f'{float(r["js_smoothness"]):.2e}', f'{float(r["hellinger_smoothness"]):.2e}'] for r in metrics]
lam_tbl = [[r["pair"], r["lambda"], fnum(r["path_deviation_from_lambda0"], 4), fnum(r["max_deviation"], 4)]
           for r in lamdev]
uncert = {}
for r in metrics:
    uncert.setdefault(r["pair"], {})[r["method"]] = float(r["mean_gpr_uncertainty"])
lam_u = [[p, fnum(uncert[p]["sph"], 4), fnum(uncert[p].get("gpr1", np.nan), 4),
          fnum(uncert[p].get("gpr4", np.nan), 4), fnum(uncert[p].get("gpr9", np.nan), 4)] for p in pairs]
W["report_metric_comparison.md"] = HDR + f"""# R3: geodesic family comparison (E4)

Settings: endpoint pairs selected deterministically per spec: {json.dumps(pairs)}.
Geodesics: 31 points, 7-candidate multi-start, L-BFGS.  All paths evaluated under
the common SPH-induced metric.  Full table: results_metrics.csv.

""" + tbl(["pair", "method", "E_g", "L_g", "R_E", "cos smooth", "max jump", "mean u", "mean H", "nearest-doc dist", "JS smooth", "Hellinger smooth"], mrows) + \
"""## lambda sensitivity: path deviation from lambda=0
""" + tbl(["pair", "lambda", "mean deviation", "max deviation"], lam_tbl) + \
"""## lambda sensitivity: mean path uncertainty
""" + tbl(["pair", "u(l=0)", "u(l=1)", "u(l=4)", "u(l=9)"], lam_u)

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
W["report_normalization_L2_L1.md"] = HDR + f"""# R4: L2 vs L1 normalization on identical routes

Settings: L2 = x/||x||_2 (directional geometry); L1 = (x+eps)/sum, eps={tfc['epsilon_l1_smoothing']}
(feature-allocation geometry).  Both built from the identical TF-IDF matrix.
Reference pair docs 0-1: JS {fnum(tfc['example_doc0_doc1']['js_divergence'],3)}, Hellinger {fnum(tfc['example_doc0_doc1']['hellinger'],3)}, Fisher-Rao {fnum(tfc['example_doc0_doc1']['fisher_rao'],3)}.

""" + tbl(["pair", "path", "L2: cos smooth", "L2: SPH length", "L1: JS smooth", "L1: Hellinger smooth", "L1: FR length"], r4rows)

# ================= R5 information geometry =================
er = eps["pair_docs_0_1"]
ep_rows = [[e, fnum(er[e]["KL(0||1)_ref"], 3), fnum(er[e]["JS_div"], 6), fnum(er[e]["Hellinger"], 6),
            fnum(er[e]["FisherRao"], 6)] for e in er]
pr = eps["path_far_max_cos_sph_step_means"]
pr_rows = [[e, fnum(pr[e]["KL_step_mean_ref"], 6), fnum(pr[e]["JS_step_mean"], 6),
            fnum(pr[e]["Hellinger_step_mean"], 6), fnum(pr[e]["FR_length"], 4)] for e in pr]
sc = eps["simplex_check"]
W["report_information_geometry.md"] = HDR + """# R5: information geometry on the simplex (epsilon sensitivity)

Settings: pi = (x+eps)/sum(x+eps); KL reported for reference only; Fisher-Rao via
the square-root embedding d_FR = 2 arccos sum sqrt(pi_k rho_k).

## (a) simplex closure (float64, all eps)
""" + tbl(["eps", "row-sum min", "row-sum max"],
    [[e, f'{sc[e]["rowsum_min"]:.15f}', f'{sc[e]["rowsum_max"]:.15f}'] for e in sc]) + \
"""## (b) eps sensitivity: distance between docs 0-1
""" + tbl(["eps", "KL(0||1) (reference)", "JS divergence", "Hellinger", "Fisher-Rao"], ep_rows) + \
"""## (c) eps sensitivity: step means along the far_max_cos SPH geodesic
""" + tbl(["eps", "KL step mean (reference)", "JS step mean", "Hellinger step mean", "FR length"], pr_rows)

# ================= R6 evaluation =================
cs = np.array(e2g["per_doc_cosine"])
order = np.argsort(cs)
meta = list(csv.DictReader(open(OUT + "/corpus_metadata.csv")))
worst = [[int(d), fnum(cs[d], 3), meta[d]["title"][:60]] for d in order[:5]]
best = [[int(d), fnum(cs[d], 3), meta[d]["title"][:60]] for d in order[-5:]]
W["report_evaluation_metrics.md"] = HDR + """# R6: distance preservation and leave-one-out interpolation (E1, E2)

## E1: distance preservation
""" + tbl(["metric", "value"],
    [["normalized stress (target t = kappa*alpha*d)", fnum(e1["stress_vs_target"])],
     ["Spearman rho (cos distance vs map distance, all pairs)", fnum(e1["spearman"])],
     ["Pearson r", fnum(e1["pearson"])],
     ["trustworthiness (k=5/7/10)", f'{fnum(e1["trustworthiness_k5"],3)} / {fnum(e1["trustworthiness_k7"],3)} / {fnum(e1["trustworthiness_k10"],3)}'],
     ["continuity (k=5/7/10)", f'{fnum(e1["continuity_k5"],3)} / {fnum(e1["continuity_k7"],3)} / {fnum(e1["continuity_k10"],3)}'],
     ["kNN preservation (k=5/7/10)", f'{fnum(e1["knn_preservation_k5"],3)} / {fnum(e1["knn_preservation_k7"],3)} / {fnum(e1["knn_preservation_k10"],3)}'],
     ["nearest-neighbour distance CV", fnum(e1["nn_dist_cv"], 3)]]) + \
"""## E2: leave-one-out interpolation (all documents)
""" + tbl(["metric", "global h", "kNN adaptive h (k=8)"],
    [["LOO cosine mean±SD", f'{fnum(e2g["loo_cosine_mean"],3)}±{fnum(e2g["loo_cosine_std"],3)}',
      f'{fnum(e2a["loo_cosine_mean"],3)}±{fnum(e2a["loo_cosine_std"],3)}'],
     ["LOO cosine median / min / max", f'{fnum(e2g["loo_cosine_median"],3)} / {fnum(e2g["loo_cosine_min"],3)} / {fnum(e2g["loo_cosine_max"],3)}', "-"],
     ["top-20 raw-TF-IDF feature recovery", fnum(e2g["topk_recovery_mean"], 3), fnum(e2a["topk_recovery_mean"], 3)],
     ["top-20 recovery restricted to shared (df>=2) features", fnum(e2g["topk_recovery_shared_df2_mean"], 3), fnum(e2a["topk_recovery_shared_df2_mean"], 3)],
     ["nearest-paper recovery (top-1)", fnum(e2g["nearest_paper_recovery_rate"], 2), fnum(e2a["nearest_paper_recovery_rate"], 2)],
     ["nearest paper within top-5", fnum(e2g["nearest_paper_in_top5_rate"], 2), fnum(e2a["nearest_paper_in_top5_rate"], 2)]]) + \
"""## per-document extremes (global h)
""" + tbl(["doc_id", "LOO cosine", "title (worst 5)"], worst) + tbl(["doc_id", "LOO cosine", "title (best 5)"], best)

# ================= R0 overall =================
chars = sum(len(d["text"]) for d in json.load(open(OUT + "/corpus/docs_clean.json")))
W["analysis_report.md"] = HDR + f"""# R0: run summary

""" + tbl(["item", "value"],
    [["documents", N_DOCS],
     ["cleaned characters", f"{chars:,}"],
     ["TF-IDF", f"char_wb 4-7gram, sublinear tf, max_features {tfc.get('max_features', 250000)}, min_df 1; L2 and L1 (eps={tfc['epsilon_l1_smoothing']}) variants stored"],
     ["map", f"8 perimeter + central anchors, alpha={fnum(man['alpha'],4)} (perimeter-only; with centre {fnum(man['alpha_if_central_included'],4)}), final J={fnum(man['final_J'],4)}"],
     ["post-hoc rescaling", f"none (free-point max radius before finalization = {fnum(man['free_point_max_radius_before_finalization'],4)})"],
     ["E1 Spearman / kNN@7", f'{fnum(e1["spearman"],3)} / {fnum(e1["knn_preservation_k7"],3)}'],
     ["E2 LOO cosine (global / adaptive)", f'{fnum(e2g["loo_cosine_mean"],3)} / {fnum(e2a["loo_cosine_mean"],3)}']]) + \
"""Details: R1 corpus-size dependence, R2 stability and sensitivity, R3 geodesic
family, R4 L2/L1 normalization, R5 information geometry, R6 evaluation metrics.
The 15 validation gates are evaluated by validate.py (paper SI, Table S1).
"""

for f, s in W.items():
    open(OUT + "/" + f, "w").write(s)
print("reports written (English, objective-only):", ", ".join(W))
