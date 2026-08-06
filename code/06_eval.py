# -*- coding: utf-8 -*-
"""§7 E1(距離保存) E2(leave-one-out再構成). usage: python3 06_eval.py e1 | e2 [h_mode]"""
import json, os, sys, time, resource
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import (OUT, DATA, load_tfidf, sph_weights, cosine_dist_matrix,
                   spearman, trustworthiness_continuity, knn_preservation, normalized_stress)

phase = sys.argv[1]
t0 = time.time()
P = np.load(OUT + "/coords.npy")
X, l2n, l1s = load_tfidf()
Xl2 = (X / l2n[:, None]).astype(np.float64)
N = len(P)
D_high = cosine_dist_matrix(Xl2)
D_low = np.linalg.norm(P[:, None] - P[None], axis=2)

if phase == "e1":
    audit = json.load(open(OUT + "/map_audit.json"))
    T = audit["kappa"] * audit["alpha"] * D_high
    iu = np.triu_indices(N, 1)
    res = {
        "stress_vs_target": normalized_stress(T, D_low),
        "stress_vs_rawcos_alpha_scaled": normalized_stress(audit["alpha"] * D_high, D_low),
        "spearman": spearman(D_high[iu], D_low[iu]),
        "pearson": float(np.corrcoef(D_high[iu], D_low[iu])[0, 1]),
    }
    for k in (5, 7, 10):
        t, c = trustworthiness_continuity(D_high, D_low, k=k)
        res[f"trustworthiness_k{k}"] = t
        res[f"continuity_k{k}"] = c
        res[f"knn_preservation_k{k}"] = knn_preservation(D_high, D_low, k=k)
    # 最近傍距離CV (R1用)
    nn = np.sort(D_low + np.eye(N) * 1e9, axis=1)[:, 0]
    res["nn_dist_cv"] = float(nn.std() / nn.mean())
    res["peak_mem_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    json.dump(res, open(OUT + "/e1_distance_preservation.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

elif phase == "e2":
    h_mode = sys.argv[2] if len(sys.argv) > 2 else "global"
    TOPK = 20
    df = (X > 0).sum(axis=0)
    shared = df >= 2                       # 共有特徴 (df>=2): 補間で原理的に回復可能な部分
    cos_list, topk_list, topk_sh_list, nn_list, nn5_list = [], [], [], [], []
    for i in range(N):
        mask = np.arange(N) != i
        Pm, Xm = P[mask], Xl2[mask]
        w = sph_weights(P[i][None], Pm, h_mode=h_mode)[0]
        v = w @ Xm
        nv = np.linalg.norm(v)
        v = v / nv if nv > 0 else v
        cos_list.append(float(v @ Xl2[i]))
        ti = set(np.argsort(X[i])[::-1][:TOPK].tolist())
        tv = set(np.argsort(v)[::-1][:TOPK].tolist())
        topk_list.append(len(ti & tv) / TOPK)
        xi_sh = np.where(shared, X[i], 0)
        v_sh = np.where(shared, v, 0)
        ti2 = set(np.argsort(xi_sh)[::-1][:TOPK].tolist())
        tv2 = set(np.argsort(v_sh)[::-1][:TOPK].tolist())
        topk_sh_list.append(len(ti2 & tv2) / TOPK)
        # 最近接論文回復: 再構成vの最近文書 == 真の最近文書 (どちらも i以外)
        sim_v = Xm @ v
        sim_x = Xm @ Xl2[i]
        true_nn = int(np.argmax(sim_x))
        nn_list.append(int(np.argmax(sim_v) == true_nn))
        nn5_list.append(int(true_nn in np.argsort(sim_v)[::-1][:5]))
        if i % 25 == 0:
            print(i, f"{time.time()-t0:.0f}s", flush=True)
    res = {"h_mode": h_mode, "n": N, "topk": TOPK,
           "loo_cosine_mean": float(np.mean(cos_list)), "loo_cosine_std": float(np.std(cos_list)),
           "loo_cosine_min": float(np.min(cos_list)), "loo_cosine_max": float(np.max(cos_list)),
           "loo_cosine_median": float(np.median(cos_list)),
           "topk_recovery_mean": float(np.mean(topk_list)),
           "topk_recovery_shared_df2_mean": float(np.mean(topk_sh_list)),
           "nearest_paper_recovery_rate": float(np.mean(nn_list)),
           "nearest_paper_in_top5_rate": float(np.mean(nn5_list)),
           "per_doc_cosine": cos_list, "per_doc_topk": topk_list,
           "per_doc_topk_shared": topk_sh_list, "per_doc_nn": nn_list}
    json.dump(res, open(OUT + f"/e2_loo_{h_mode}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["loo_cosine_mean", "loo_cosine_std", "topk_recovery_mean",
          "topk_recovery_shared_df2_mean", "nearest_paper_recovery_rate",
          "nearest_paper_in_top5_rate"]}, indent=1))

print("elapsed", round(time.time() - t0, 1))
