# -*- coding: utf-8 -*-
"""E9: 測地線言語化の L1ベース vs L2ベース 比較.
- L2版: v(P)=normalize(Σw x̂) のコーパス平均方向との差分 top特徴 (従来手法)
- L1版: π(P)=Σw π のコーパス平均分布 π̄ に対する KL寄与 π_k·log(π_k/π̄_k) top特徴
同一経路の各通過点で両者を抽出し、Jaccard重複率・固有語彙を評価する。
出力: KM_OUT/e9_verbalization_L1_L2.json
usage: python3 14_e9_verbalization.py [pair] [method_l2path] [n_points]
"""
import json, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import OUT, DATA, load_tfidf, sph_weights, term_ok, load_stoplist

pair = sys.argv[1] if len(sys.argv) > 1 else "far_max_cos"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 11
TOPK = 15
EPS = 1e-10

P = np.load(OUT + "/coords.npy")
X, l2n, l1s = load_tfidf()
Xl2 = (X / l2n[:, None]).astype(np.float64)
Xl1 = ((X.astype(np.float64) + EPS) / (X.astype(np.float64) + EPS).sum(1, keepdims=True))
vocab = json.load(open(DATA + "/vocab.json"))
mean_l2 = Xl2.mean(0); mean_l2 /= np.linalg.norm(mean_l2)
mean_l1 = Xl1.mean(0)

_STOP = load_stoplist()
def pick(scores, k=TOPK):
    words = []
    for i in np.argsort(scores)[::-1]:
        t = vocab[i].strip()
        if term_ok(t, _STOP) and not any(t in x or x in t for x in words):
            words.append(t)
        if len(words) >= k:
            break
    return words

def resample(path, n):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])], 1)

out = {"pair": pair, "n_points": n, "topk": TOPK,
       "definitions": {
           "L2": "score_k = v_k(P) - mean_dir_k (方向差分; 従来の言語化)",
           "L1": "score_k = pi_k(P) * log(pi_k(P)/mean_pi_k) (KL寄与; 情報幾何的言語化)"},
       "paths": {}}

for method in ["sph", "fr"]:
    f = f"{DATA}/path_{pair}_{method}.npy"
    if not os.path.exists(f):
        continue
    pts = resample(np.load(f), n)
    steps = []
    for k in range(n):
        w = sph_weights(pts[k][None], P, h_mode="knn_adaptive", knn_k=8)[0]
        # L2
        v = w @ Xl2; v /= np.linalg.norm(v)
        wl2 = pick(v - mean_l2)
        # L1 (KL寄与)
        pi = w @ Xl1
        pi = pi / pi.sum()
        kl_contrib = pi * np.log(np.maximum(pi, 1e-300) / np.maximum(mean_l1, 1e-300))
        wl1 = pick(kl_contrib)
        s2, s1 = set(wl2), set(wl1)
        jac = len(s2 & s1) / len(s2 | s1)
        steps.append({"t": round(k / (n - 1), 3),
                      "xy": pts[k].round(3).tolist(),
                      "L2_top": wl2, "L1_top": wl1,
                      "jaccard": round(jac, 3),
                      "L2_only": sorted(s2 - s1)[:6], "L1_only": sorted(s1 - s2)[:6]})
        print(f"{method} t={k/(n-1):.1f} jac={jac:.2f}", flush=True)
    out["paths"][method] = {
        "waypoints": steps,
        "mean_jaccard": round(float(np.mean([s["jaccard"] for s in steps])), 3)}

json.dump(out, open(OUT + "/e9_verbalization_L1_L2.json", "w"), ensure_ascii=False, indent=1)
print("E9 done:", {m: out["paths"][m]["mean_jaccard"] for m in out["paths"]})
