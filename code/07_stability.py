# -*- coding: utf-8 -*-
"""§9 計算安定性・感度解析 (E7) + §8 データサイズ(R1).
usage: python3 07_stability.py <phase> [...]
 phases: seedmaps s1 [s2..] | seedpost | anchors | regs a b | sph_h | gpr_kernels | delta | multistart | datasize
"""
import json, os, sys, time, subprocess, resource
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import (OUT, DATA, load_tfidf, sph_weights, sph_entropy, cosine_dist_matrix,
                   L2Field, GPR, lbfgs, kmeans, ari, procrustes_dist,
                   knn_preservation, normalized_stress, spearman)

phase = sys.argv[1]
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))

def loo_cosine(P, Xl2, h_mode, knn_k=8):
    N = len(P)
    cs = []
    for i in range(N):
        mask = np.arange(N) != i
        w = sph_weights(P[i][None], P[mask], h_mode=h_mode,
                        h_fixed=0.35 if h_mode == "fixed" else None, knn_k=knn_k)[0]
        v = w @ Xl2[mask]
        n = np.linalg.norm(v)
        cs.append(float(v @ Xl2[i] / n) if n > 0 else 0.0)
    return float(np.mean(cs)), float(np.std(cs))

def small_geodesic(P, G, ps, pt, npts=31, maxiter=500):
    fld = L2Field(G, P, h_mode="global")
    DELTA = 1e-3
    def mfn(Q):
        g = fld.metric(Q, delta=DELTA)
        g[:, 0, 0] += 1e-10; g[:, 1, 1] += 1e-10
        return g
    t = np.linspace(0, 1, npts)[:, None]
    base = (1 - t) * ps[None] + t * pt[None]
    d = pt - ps
    perp = np.array([-d[1], d[0]]); perp /= (np.linalg.norm(perp) + 1e-12)
    def eg(x):
        path = np.vstack([ps, x.reshape(-1, 2), pt])
        K = len(path) - 1
        dp = np.diff(path, axis=0)
        mid = 0.5 * (path[:-1] + path[1:])
        sh = np.array([[0, 0], [DELTA, 0], [-DELTA, 0], [0, DELTA], [0, -DELTA]])
        allq = (mid[:, None, :] + sh[None]).reshape(-1, 2)
        gall = mfn(allq).reshape(K, 5, 2, 2)
        g0 = gall[:, 0]
        dgx = (gall[:, 1] - gall[:, 2]) / (2 * DELTA)
        dgy = (gall[:, 3] - gall[:, 4]) / (2 * DELTA)
        E = float(np.einsum("ka,kab,kb->", dp, g0, dp))
        gd = np.einsum("kab,kb->ka", g0, dp)
        qx = np.einsum("ka,kab,kb->k", dp, dgx, dp)
        qy = np.einsum("ka,kab,kb->k", dp, dgy, dp)
        grad = np.zeros_like(path)
        grad[:-1] += -2 * gd; grad[1:] += 2 * gd
        grad[:-1, 0] += 0.5 * qx; grad[1:, 0] += 0.5 * qx
        grad[:-1, 1] += 0.5 * qy; grad[1:, 1] += 0.5 * qy
        return E, grad[1:-1].ravel()
    best = None
    for a in [0.0]:
        x0 = (base + a * np.sin(np.pi * t) * perp[None])[1:-1].ravel()
        xf, Ef, suc, nit = lbfgs(eg, x0, maxiter=maxiter)
        if best is None or Ef < best[1]:
            best = (xf, Ef)
    return np.vstack([ps, best[0].reshape(-1, 2), pt]), best[1]

# ---------------------------------------------------------------
if phase == "seedmaps":
    for s in sys.argv[2:]:
        r = subprocess.run(["python3", f"{HERE}/12_map_v30.py", f"seed{s}", s, "8"],
                           capture_output=True, text=True)
        print("seed", s, "done", r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:], flush=True)

elif phase == "seedpost":
    X, l2n, l1s = load_tfidf()
    Xl2 = (X / l2n[:, None]).astype(np.float64)
    D = cosine_dist_matrix(Xl2)
    G = np.load(DATA + "/gram_l2.npy")
    P0 = np.load(OUT + "/coords.npy")
    audit = json.load(open(OUT + "/map_audit.json"))
    T = audit["kappa"] * audit["alpha"] * D
    lab0, _ = kmeans(P0, 5, seed=0)
    pairs = json.load(open(DATA + "/pairs.json"))
    i, j = pairs["far_max_cos"]
    path0, _ = small_geodesic(P0, G, P0[i], P0[j], maxiter=200)
    seeds = [int(a) for a in sys.argv[2:]] or [0, 1, 2, 3, 4, 5, 10, 20]
    old = []
    if os.path.exists(OUT + "/e7_seeds.json"):
        old = [r for r in json.load(open(OUT + "/e7_seeds.json")) if r["seed"] not in seeds]
    rows = old
    for s in seeds:
        f = OUT + f"/coords_seed{s}.npy"
        if not os.path.exists(f):
            print("missing", s); continue
        P = np.load(f)
        Dl = np.linalg.norm(P[:, None] - P[None], axis=2)
        lab, _ = kmeans(P, 5, seed=0)
        path, _ = small_geodesic(P, G, P[i], P[j], maxiter=200)
        # 測地線偏差: 主マップへProcrustes整列後の平均点距離
        A = P - P.mean(0); B = P0 - P0.mean(0)
        na, nb = np.linalg.norm(A), np.linalg.norm(B)
        U, sv, Vt = np.linalg.svd((A / na).T @ (B / nb))
        R = U @ Vt; sc = sv.sum() * nb / na
        path_t = (path - P.mean(0)) @ R * sc + P0.mean(0)
        dev = float(np.linalg.norm(path_t - path0, axis=1).mean())
        rows.append({"seed": s, "stress": normalized_stress(T, Dl),
                     "knn_preservation_k7": knn_preservation(D, Dl, k=7),
                     "cluster_ari_vs_seed0": ari(lab0.tolist(), lab.tolist()),
                     "procrustes_vs_main": procrustes_dist(P0, P),
                     "geodesic_deviation": dev})
        print(rows[-1], flush=True)
    json.dump(rows, open(OUT + "/e7_seeds.json", "w"), indent=1)

elif phase == "anchors":
    for np_ in ["4", "8"]:
        tagged = f"anch{np_}"
        if not os.path.exists(OUT + f"/coords_{tagged}.npy"):
            subprocess.run(["python3", f"{HERE}/12_map_v30.py", tagged, "0", np_],
                           capture_output=True, text=True)
    X, l2n, _ = load_tfidf()
    Xl2 = (X / l2n[:, None]).astype(np.float64)
    D = cosine_dist_matrix(Xl2)
    P0 = np.load(OUT + "/coords.npy")
    lab0, _ = kmeans(P0, 5, seed=0)
    rows = []
    for np_ in [4, 8]:
        P = np.load(OUT + f"/coords_anch{np_}.npy")
        aud = json.load(open(OUT + f"/map_audit_anch{np_}.json"))
        Dl = np.linalg.norm(P[:, None] - P[None], axis=2)
        lab, _ = kmeans(P, 5, seed=0)
        rows.append({"n_perim": np_, "alpha": aud["alpha"], "final_J": aud["final_J"],
                     "stress": normalized_stress(aud["kappa"] * aud["alpha"] * D, Dl),
                     "procrustes_vs_main": procrustes_dist(P0, P),
                     "cluster_ari_vs_main": ari(lab0.tolist(), lab.tolist()),
                     "knn_preservation_k7": knn_preservation(D, Dl, k=7)})
        print(rows[-1], flush=True)
    json.dump(rows, open(OUT + "/e7_anchors.json", "w"), indent=1)

elif phase == "regs":
    jobs = []
    for nm in ["lambda_rep", "lambda_cover", "lambda_center"]:
        for fc in [0.25, 0.5, 2.0, 4.0]:
            jobs.append((nm, fc))
    a, b = int(sys.argv[2]), int(sys.argv[3])
    for nm, fc in jobs[a:b]:
        tag = f"reg_{nm}_{fc}"
        subprocess.run(["python3", f"{HERE}/12_map_v30.py", tag, "0", "8", nm, str(fc)],
                       capture_output=True, text=True)
        print(tag, "done", flush=True)

elif phase == "regpost":
    X, l2n, _ = load_tfidf()
    Xl2 = (X / l2n[:, None]).astype(np.float64)
    D = cosine_dist_matrix(Xl2)
    rows = []
    for nm in ["lambda_rep", "lambda_cover", "lambda_center"]:
        for fc in [0.25, 0.5, 1.0, 2.0, 4.0]:
            if fc == 1.0:
                P = np.load(OUT + "/coords.npy"); aud = json.load(open(OUT + "/map_audit.json"))
            else:
                f = OUT + f"/coords_reg_{nm}_{fc}.npy"
                if not os.path.exists(f): continue
                P = np.load(f); aud = json.load(open(OUT + f"/map_audit_reg_{nm}_{fc}.json"))
            Dl = np.linalg.norm(P[:, None] - P[None], axis=2)
            spread = float(P.std())
            crowd = float((np.abs(P).max(axis=1) > 0.9).mean())
            rows.append({"param": nm, "factor": fc,
                         "stress": normalized_stress(aud["kappa"] * aud["alpha"] * D, Dl),
                         "spread": spread, "boundary_crowding": crowd, "final_J": aud["final_J"]})
            print(rows[-1], flush=True)
    json.dump(rows, open(OUT + "/e7_regs.json", "w"), indent=1)

elif phase == "sph_h":
    X, l2n, _ = load_tfidf()
    Xl2 = (X / l2n[:, None]).astype(np.float64)
    G = np.load(DATA + "/gram_l2.npy")
    P = np.load(OUT + "/coords.npy")
    N = len(P)
    path = np.load(DATA + "/path_far_max_cos_sph.npy")
    n = 25
    xs = np.linspace(-1, 1, n)
    gx, gy = np.meshgrid(xs, xs)
    GR = np.stack([gx.ravel(), gy.ravel()], 1)
    rows = []
    for mode, kk in [("global", 8), ("fixed", 8), ("knn_adaptive", 8), ("knn_adaptive", 4), ("density_adaptive", 8)]:
        lc, ls = loo_cosine(P, Xl2, mode, knn_k=kk)
        w = sph_weights(GR, P, h_mode=mode, h_fixed=0.35, knn_k=kk)
        H = float(sph_entropy(w, N).mean())
        wp = sph_weights(path, P, h_mode=mode, h_fixed=0.35, knn_k=kk)
        q = np.einsum("mi,ij,mj->m", wp, G, wp)
        c = wp / np.sqrt(np.maximum(q, 1e-300))[:, None]
        cosstep = 1 - np.clip(np.einsum("mi,ij,mj->m", c[:-1], G, c[1:]), -1, 1)
        rows.append({"h_mode": mode, "knn_k": kk, "loo_cosine_mean": lc, "loo_cosine_std": ls,
                     "mean_entropy": H, "path_cos_smoothness": float(cosstep.mean()),
                     "path_max_jump": float(cosstep.max())})
        print(rows[-1], flush=True)
    json.dump(rows, open(OUT + "/e7_sph_h.json", "w"), indent=1)

elif phase == "gpr_kernels":
    Z = np.load(DATA + "/svd_scores.npy")[:, :10]
    P = np.load(OUT + "/coords.npy")
    zs = Z.std(); Y = Z / zs
    rows = []
    for kern, white in [("rbf", True), ("rbf_iso", True), ("matern32", True),
                        ("matern52", True), ("rbf", False)]:
        gp = GPR(kernel=kern, seed=0, n_restarts=10).fit(P, Y, use_white=white)
        K = gp._K(P, P, gp.theta) + gp.noise * np.eye(len(P))
        Ki = np.linalg.inv(K)
        # GPR-LOO (closed form)
        loo_err = (Ki @ Y) / np.diag(Ki)[:, None]
        loo_var = 1.0 / np.diag(Ki)
        rmse = float(np.sqrt((loo_err ** 2).mean()))
        # calibration: |残差| と LOOσ の相関
        r = np.abs(loo_err).mean(1)
        s = np.sqrt(loo_var)
        calib = float(np.corrcoef(r, s)[0, 1])
        # z-score: 残差/σ の分布 (較正なら std≈1)
        z = (loo_err / s[:, None]).ravel()
        rows.append({"kernel": kern, "white": white, "lml": gp.lml_,
                     "theta_exp": np.exp(gp.theta).round(4).tolist(),
                     "loo_rmse_latent": rmse, "calibration_corr": calib,
                     "z_std": float(z.std())})
        print(rows[-1], flush=True)
    json.dump(rows, open(OUT + "/e7_gpr_kernels.json", "w"), indent=1)

elif phase == "delta":
    X, l2n, _ = load_tfidf()
    G = np.load(DATA + "/gram_l2.npy")
    P = np.load(OUT + "/coords.npy")
    fld = L2Field(G, P, h_mode="global")
    probes = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [0.0, 0.0], [0.3, -0.7]])
    deltas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    ref = fld.metric(probes, delta=1e-3)
    rows = []
    for d in deltas:
        g = fld.metric(probes, delta=d)
        ev = np.linalg.eigvalsh(g)
        cond = (ev[:, 1] / np.maximum(ev[:, 0], 1e-300))
        rel = np.linalg.norm((g - ref).reshape(len(probes), -1), axis=1) / \
              np.maximum(np.linalg.norm(ref.reshape(len(probes), -1), axis=1), 1e-300)
        rows.append({"delta": d, "mean_cond": float(cond.mean()), "max_cond": float(cond.max()),
                     "mean_rel_dev_vs_1e-3": float(rel.mean()), "max_rel_dev_vs_1e-3": float(rel.max())})
        print(rows[-1], flush=True)
    json.dump(rows, open(OUT + "/e7_delta.json", "w"), indent=1)

elif phase == "multistart":
    import glob
    rows = []
    for f in sorted(glob.glob(DATA + "/geo_*.json")):
        r = json.load(open(f))
        Es = [t["E"] for t in r["tries"]]
        rows.append({"pair": r["pair"], "method": r["method"], "status": r["status"],
                     "n_success_starts": sum(t["success"] for t in r["tries"]),
                     "n_starts": len(r["tries"]),
                     "E_best": min(Es), "E_worst": max(Es),
                     "E_spread_rel": (max(Es) - min(Es)) / (abs(min(Es)) + 1e-300),
                     "best_a": r["tries"][int(np.argmin(Es))]["a"]})
    json.dump(rows, open(OUT + "/e7_multistart.json", "w"), indent=1)
    sr = np.mean([r["n_success_starts"] / r["n_starts"] for r in rows])
    print("mean start success rate:", sr)
    print("all adopted success:", all(r["status"] == "success" for r in rows))

elif phase == "datasize":
    # N=<n> サブセット (§8): map + E1 + LOO + 時間/メモリ
    Nn = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    env = dict(os.environ, KM_SUBSET_N=str(Nn))
    t1 = time.time()
    subprocess.run(["python3", f"{HERE}/12_map_v30.py", f"N{Nn}", "0", "8"],
                   capture_output=True, text=True, env=env)
    t_map = time.time() - t1
    X, l2n, _ = load_tfidf()
    Xl2 = (X / l2n[:, None]).astype(np.float64)
    N_full = len(Xl2)
    if Nn < N_full:
        sub = np.load(OUT + f"/subset_idx_N{Nn}.npy")
        Pn = np.load(OUT + f"/coords_N{Nn}.npy")
        aud = json.load(open(OUT + f"/map_audit_N{Nn}.json"))
    else:
        sub = np.arange(N_full)
        Pn = np.load(OUT + "/coords.npy")
        aud = json.load(open(OUT + "/map_audit.json"))
    Xn = Xl2[sub]
    Dn = cosine_dist_matrix(Xn)
    Dl = np.linalg.norm(Pn[:, None] - Pn[None], axis=2)
    t2 = time.time()
    lc, ls = loo_cosine(Pn, Xn, "global")
    t_loo = time.time() - t2
    nn = np.sort(Dl + np.eye(Nn) * 1e9, axis=1)[:, 0]
    iu = np.triu_indices(Nn, 1)
    res = {"N": Nn, "subset_doc_ids": sub.tolist(), "map_time_sec": round(t_map, 1),
           "loo_time_sec": round(t_loo, 2),
           "tfidf_nonzero_cols": int((X[sub] > 0).any(axis=0).sum()),
           "stress": normalized_stress(aud["kappa"] * aud["alpha"] * Dn, Dl),
           "spearman": spearman(Dn[iu], Dl[iu]),
           "knn_preservation_k5": knn_preservation(Dn, Dl, k=5),
           "loo_cosine_mean": lc, "loo_cosine_std": ls,
           "nn_dist_cv": float(nn.std() / nn.mean()), "alpha": aud["alpha"],
           "peak_mem_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    json.dump(res, open(OUT + f"/r1_N{Nn}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["N","map_time_sec","loo_time_sec","stress","spearman","loo_cosine_mean","peak_mem_mb"]}, indent=1))

print("elapsed", round(time.time() - t0, 1))
