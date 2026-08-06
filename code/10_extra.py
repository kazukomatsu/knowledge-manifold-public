# -*- coding: utf-8 -*-
"""R5用: ε感度解析・D_KL参考値・Σπ=1検証(float64) / E4用: λ経路偏差"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import OUT, DATA, load_tfidf, sph_weights

X, l2n, l1s = load_tfidf()
P = np.load(OUT + "/coords.npy")
pairs = json.load(open(DATA + "/pairs.json"))

def info_dists(p, q):
    m = 0.5 * (p + q)
    kl_pq = float(np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300) / np.maximum(q, 1e-300)), 0)))
    js = float(0.5 * np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300) / m), 0))
               + 0.5 * np.sum(np.where(q > 0, q * np.log(np.maximum(q, 1e-300) / m), 0)))
    hel = float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2))
    fr = float(2 * np.arccos(np.clip(np.sum(np.sqrt(p * q)), 0, 1)))
    return kl_pq, js, hel, fr

# --- ε感度: doc0-1 と far_max_cos 経路のステップ平均 ---
i, j = pairs["far_max_cos"]
path = np.load(f"{DATA}/path_far_max_cos_sph.npy")
w = sph_weights(path, P, h_mode="global")
out = {"pair_docs_0_1": {}, "path_far_max_cos_sph_step_means": {}, "simplex_check": {}}
for eps in [1e-12, 1e-10, 1e-8, 1e-6]:
    Xl1 = (X.astype(np.float64) + eps)
    Xl1 /= Xl1.sum(1, keepdims=True)
    kl, js, hel, fr = info_dists(Xl1[0], Xl1[1])
    out["pair_docs_0_1"][str(eps)] = {"KL(0||1)_ref": kl, "JS_div": js, "Hellinger": hel, "FisherRao": fr}
    pi = w @ Xl1
    pi /= pi.sum(1, keepdims=True)
    kls, jss, hels, frs = [], [], [], []
    for k in range(len(pi) - 1):
        a, b, c, d = info_dists(pi[k], pi[k + 1])
        kls.append(a); jss.append(b); hels.append(c); frs.append(d)
    out["path_far_max_cos_sph_step_means"][str(eps)] = {
        "KL_step_mean_ref": float(np.mean(kls)), "JS_step_mean": float(np.mean(jss)),
        "Hellinger_step_mean": float(np.mean(hels)), "FR_length": float(np.sum(frs))}
    rs = Xl1.sum(1)
    out["simplex_check"][str(eps)] = {"rowsum_min": float(rs.min()), "rowsum_max": float(rs.max()),
                                      "computed_in": "float64"}
    print(eps, "done", flush=True)
json.dump(out, open(OUT + "/r5_epsilon_sensitivity.json", "w"), indent=1)

# --- E4: λ経路偏差 (λ=0基準の平均点距離) ---
rows = []
for pid in pairs:
    base = np.load(f"{DATA}/path_{pid}_sph.npy")
    for m, lam in [("gpr1", 1), ("gpr4", 4), ("gpr9", 9)]:
        f = f"{DATA}/path_{pid}_{m}.npy"
        if os.path.exists(f):
            p2 = np.load(f)
            rows.append({"pair": pid, "lambda": lam,
                         "path_deviation_from_lambda0": float(np.linalg.norm(p2 - base, axis=1).mean()),
                         "max_deviation": float(np.linalg.norm(p2 - base, axis=1).max())})
json.dump(rows, open(OUT + "/e4_lambda_path_deviation.json", "w"), indent=1)
print("done")
