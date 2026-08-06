# -*- coding: utf-8 -*-
"""§5-6 計量族と多計量測地線. §6.1確実化手順(31点/7候補/L-BFGS/採用規則).
usage:
  python3 05_geodesics.py pairs
  python3 05_geodesics.py run <pair_id> <method>   # method: sph | gpr1|gpr4|gpr9 | fr
  python3 05_geodesics.py graph
  python3 05_geodesics.py eval
"""
import json, os, sys, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import (OUT, DATA, load_tfidf, sph_weights, sph_entropy, L2Field, GPR,
                   lbfgs, kmeans, knn_graph, is_connected, dijkstra, cosine_dist_matrix)

NPTS = 31          # 離散点数 (内部29)
DELTA = 1e-3
EPS_SING = 1e-10
TOPK_FR = 2000     # §5.2 top feature subset (FR計量の安定化)

t0 = time.time()
P = np.load(OUT + "/coords.npy")
N = len(P)
X, l2n, l1s = load_tfidf()
G = np.load(DATA + "/gram_l2.npy")
fld = L2Field(G, P, h_mode="global")
md = pickle.load(open(DATA + "/gpr_model.pkl", "rb"))
gp = GPR(kernel="rbf"); gp.theta = md["theta"]; gp.X = md["X"]; gp.alpha = md["alpha"]
gp.L = md["L"]; gp.noise = md["noise"]

# ---- FR用 top-K L1分布 ----
def build_Pi():
    EPS = 1e-10
    Xl1 = (X + EPS) / l1s[:, None]                  # (N,V)
    mass = Xl1.sum(0)
    top = np.argsort(mass)[::-1][:TOPK_FR]
    Pi = Xl1[:, top].astype(np.float64)
    Pi = Pi / Pi.sum(1, keepdims=True)              # 再正規化 (部分集合上の確率単体)
    return Pi
_PI = None
def get_Pi():
    global _PI
    if _PI is None:
        _PI = build_Pi()
    return _PI

# ---- 計量関数 (M,2)->(M,2,2) ----
sing_count = {"n": 0}
def regularize(g):
    tr = g[:, 0, 0] + g[:, 1, 1]
    det = g[:, 0, 0] * g[:, 1, 1] - g[:, 0, 1] ** 2
    bad = (det < 1e-14) | (tr < 1e-14)
    sing_count["n"] += int(bad.sum())
    g = g.copy()
    g[:, 0, 0] += EPS_SING
    g[:, 1, 1] += EPS_SING
    return g

def metric_sph(Q):
    return regularize(fld.metric(Q, delta=DELTA))

def metric_gpr(Q, lam):
    g = fld.metric(Q, delta=DELTA)
    u = gp.rel_uncertainty(Q)
    return regularize(g * (1 + lam * u ** 2)[:, None, None])

def metric_fr(Q):
    Pi = get_Pi()
    Q = np.atleast_2d(Q)
    M = len(Q)
    shifts = np.array([[DELTA, 0], [-DELTA, 0], [0, DELTA], [0, -DELTA], [0, 0]])
    allpts = (Q[:, None, :] + shifts[None]).reshape(-1, 2)
    w = sph_weights(allpts, P, h_mode="global")          # (5M,100)
    pi = (w @ Pi).reshape(M, 5, -1)                      # (M,5,K)
    dx = (pi[:, 0] - pi[:, 1]) / (2 * DELTA)
    dy = (pi[:, 2] - pi[:, 3]) / (2 * DELTA)
    p0 = np.maximum(pi[:, 4], 1e-300)
    g = np.empty((M, 2, 2))
    g[:, 0, 0] = np.sum(dx * dx / p0, axis=1)
    g[:, 1, 1] = np.sum(dy * dy / p0, axis=1)
    g[:, 0, 1] = g[:, 1, 0] = np.sum(dx * dy / p0, axis=1)
    return regularize(g)

METRICS = {"sph": metric_sph, "fr": metric_fr,
           "gpr1": lambda Q: metric_gpr(Q, 1.0),
           "gpr4": lambda Q: metric_gpr(Q, 4.0),
           "gpr9": lambda Q: metric_gpr(Q, 9.0)}

# ---- 経路エネルギーと勾配 ----
def path_energy(path, mfn):
    dp = np.diff(path, axis=0)                 # (K,2)
    mid = 0.5 * (path[:-1] + path[1:])
    g = mfn(mid)
    E = float(np.einsum("ka,kab,kb->", dp, g, dp))
    L = float(np.sum(np.sqrt(np.maximum(np.einsum("ka,kab,kb->k", dp, g, dp), 0))))
    return E, L

def energy_and_grad(inner, ps, pt, mfn):
    path = np.vstack([ps, inner.reshape(-1, 2), pt])
    K = len(path) - 1
    dp = np.diff(path, axis=0)
    mid = 0.5 * (path[:-1] + path[1:])
    # g と ∂g を一括評価
    shifts = np.array([[0, 0], [DELTA, 0], [-DELTA, 0], [0, DELTA], [0, -DELTA]])
    allq = (mid[:, None, :] + shifts[None]).reshape(-1, 2)
    gall = mfn(allq).reshape(K, 5, 2, 2)
    g0 = gall[:, 0]
    dgx = (gall[:, 1] - gall[:, 2]) / (2 * DELTA)
    dgy = (gall[:, 3] - gall[:, 4]) / (2 * DELTA)
    E = float(np.einsum("ka,kab,kb->", dp, g0, dp))
    # 勾配
    gd = np.einsum("kab,kb->ka", g0, dp)             # g Δp
    quadx = np.einsum("ka,kab,kb->k", dp, dgx, dp)
    quady = np.einsum("ka,kab,kb->k", dp, dgy, dp)
    grad = np.zeros_like(path)
    grad[:-1] += -2 * gd
    grad[1:] += 2 * gd
    grad[:-1, 0] += 0.5 * quadx; grad[1:, 0] += 0.5 * quadx
    grad[:-1, 1] += 0.5 * quady; grad[1:, 1] += 0.5 * quady
    return E, grad[1:-1].ravel()

def geodesic(ps, pt, mfn, maxiter=2000):
    t = np.linspace(0, 1, NPTS)[:, None]
    base = (1 - t) * ps[None] + t * pt[None]
    d = pt - ps
    perp = np.array([-d[1], d[0]])
    nrm = np.linalg.norm(perp)
    perp = perp / nrm if nrm > 0 else np.array([0.0, 1.0])
    E_straight, _ = path_energy(base, mfn)
    best = None
    tries = []
    for a in [0.0, 0.01, -0.01, 0.03, -0.03, 0.05, -0.05]:
        init = base + a * np.sin(np.pi * t) * perp[None]
        x0 = init[1:-1].ravel()
        xf, Ef, suc, nit = lbfgs(lambda x: energy_and_grad(x, ps, pt, mfn), x0,
                                 maxiter=maxiter, ftol=1e-12, gtol=1e-8, maxls=50)
        tries.append({"a": a, "E": Ef, "success": bool(suc), "nit": int(nit)})
        if best is None or Ef < best[1]:
            best = (xf, Ef, suc)
    xf, Ef, suc = best
    path = np.vstack([ps, xf.reshape(-1, 2), pt])
    if suc and Ef <= E_straight:
        status = "success"
    elif Ef < E_straight - 1e-12:
        status = "energy_improved"
    else:
        status = "failed"
    return path, Ef, E_straight, status, tries

# ================= phases =================
phase = sys.argv[1]

if phase == "pairs":
    Z = np.load(DATA + "/svd_scores.npy")[:, :10]
    lab, C = kmeans(Z, 5, seed=0)
    np.save(DATA + "/cluster_labels.npy", lab)
    Xl2 = (X / l2n[:, None]).astype(np.float64)
    D = cosine_dist_matrix(Xl2)
    d2d = np.linalg.norm(P[:, None] - P[None], axis=2)
    pairs = {}
    # near: 同一クラスタ内, 2D距離が中央値程度のペア (最大クラスタ)
    big = np.argmax(np.bincount(lab))
    idx = np.where(lab == big)[0]
    dd = d2d[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    med = np.median(dd[iu])
    k = np.argmin(np.abs(dd[iu] - med))
    pairs["near_intra"] = [int(idx[iu[0][k]]), int(idx[iu[1][k]])]
    # mid: 隣接クラスタ間 (セントロイド距離最小の2クラスタの代表=セントロイド最近傍文書)
    cd = np.linalg.norm(C[:, None] - C[None], axis=2) + np.eye(5) * 1e9
    a, b = np.unravel_index(np.argmin(cd), cd.shape)
    ra = int(np.where(lab == a)[0][np.argmin(np.linalg.norm(Z[lab == a] - C[a], axis=1))])
    rb = int(np.where(lab == b)[0][np.argmin(np.linalg.norm(Z[lab == b] - C[b], axis=1))])
    pairs["mid_adjacent"] = [ra, rb]
    # far: cos距離最大ペア
    i0, j0 = np.unravel_index(np.argmax(D), D.shape)
    pairs["far_max_cos"] = [int(i0), int(j0)]
    # cross: 2D距離最大ペア (クラスタ横断)
    i1, j1 = np.unravel_index(np.argmax(d2d), d2d.shape)
    pairs["cross_max_2d"] = [int(i1), int(j1)]
    # high uncertainty: 直線経路の平均uが最大のペア (2D距離>1に限定)
    cand = np.array(np.where(np.triu(d2d > 1.2, 1))).T
    rng = np.random.default_rng(0)
    if len(cand) > 400:
        cand = cand[rng.choice(len(cand), 400, replace=False)]
    best_u, best_pair = -1, None
    for i, j in cand:
        t = np.linspace(0, 1, 15)[:, None]
        line = (1 - t) * P[i][None] + t * P[j][None]
        mu = float(gp.rel_uncertainty(line).mean())
        if mu > best_u:
            best_u, best_pair = mu, (int(i), int(j))
    pairs["high_uncertainty"] = list(best_pair)
    # medium2: 中距離の別クラスタペア (遠すぎない)
    far_cd = np.unravel_index(np.argmax(cd * (cd < 1e8)), cd.shape)
    c1, c2 = far_cd
    r1 = int(np.where(lab == c1)[0][np.argmin(np.linalg.norm(Z[lab == c1] - C[c1], axis=1))])
    r2 = int(np.where(lab == c2)[0][np.argmin(np.linalg.norm(Z[lab == c2] - C[c2], axis=1))])
    pairs["far_centroids"] = [r1, r2]
    json.dump(pairs, open(DATA + "/pairs.json", "w"), indent=1)
    json.dump(pairs, open(OUT + "/endpoint_pairs.json", "w"), indent=1)
    print(json.dumps(pairs))
    print("cluster sizes:", np.bincount(lab).tolist())

elif phase == "run":
    pid, method = sys.argv[2], sys.argv[3]
    pairs = json.load(open(DATA + "/pairs.json"))
    i, j = pairs[pid]
    mfn = METRICS[method]
    path, Ef, Estr, status, tries = geodesic(P[i], P[j], mfn)
    np.save(f"{DATA}/path_{pid}_{method}.npy", path)
    rec = {"pair": pid, "docs": [i, j], "method": method, "E_final": Ef,
           "E_straight": Estr, "status": status, "tries": tries,
           "singular_regularized_points": sing_count["n"], "eps_sing": EPS_SING,
           "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(rec, open(f"{DATA}/geo_{pid}_{method}.json", "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ["pair", "method", "E_final", "E_straight", "status", "elapsed_sec"]}))

elif phase == "graph":
    # kNN graph geodesic (cosine, k=5, 連結性確認)
    Xl2 = (X / l2n[:, None]).astype(np.float64)
    D = cosine_dist_matrix(Xl2)
    k = 5
    adj = knn_graph(D, k)
    while not is_connected(adj):
        k += 1
        adj = knn_graph(D, k)
    pairs = json.load(open(DATA + "/pairs.json"))
    out = {"k": k, "connected": True, "paths": {}}
    for pid, (i, j) in pairs.items():
        dist, prev = dijkstra(adj, i)
        node = j; seq = [j]
        while prev[node] is not None:
            node = prev[node]; seq.append(node)
        seq = seq[::-1]
        assert seq[0] == i
        # 2D折れ線をNPTS点に弧長等分リサンプル
        pl = P[seq]
        seg = np.linalg.norm(np.diff(pl, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        s = np.linspace(0, cum[-1], NPTS)
        px = np.interp(s, cum, pl[:, 0]); py = np.interp(s, cum, pl[:, 1])
        path = np.stack([px, py], 1)
        np.save(f"{DATA}/path_{pid}_graph.npy", path)
        out["paths"][pid] = {"nodes": [int(v) for v in seq], "graph_dist": float(dist[j])}
    json.dump(out, open(DATA + "/graph_geo.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:800])

elif phase == "eval":
    # 全経路×全指標 (§6.2) → results_metrics.csv / geodesic_results.csv
    EPS = 1e-10
    Xl1_full = None
    pairs = json.load(open(DATA + "/pairs.json"))
    Xl2 = (X / l2n[:, None]).astype(np.float64)
    rows = []
    geo_rows = []
    for pid, (i, j) in pairs.items():
        t = np.linspace(0, 1, NPTS)[:, None]
        line = (1 - t) * P[i][None] + t * P[j][None]
        E_line_sph, L_line_sph = path_energy(line, metric_sph)
        methods = {"line": line}
        gpath = f"{DATA}/path_{pid}_graph.npy"
        if os.path.exists(gpath):
            methods["graph"] = np.load(gpath)
        for m in ["sph", "gpr1", "gpr4", "gpr9", "fr"]:
            f = f"{DATA}/path_{pid}_{m}.npy"
            if os.path.exists(f):
                methods[m] = np.load(f)
        for m, path in methods.items():
            E_sph, L_sph = path_energy(path, metric_sph)     # 共通基準: SPH計量
            R_E = (E_line_sph - E_sph) / E_line_sph if E_line_sph > 0 else 0.0
            # L2場でのステップ変化
            w = sph_weights(path, P, h_mode="global")
            q = np.einsum("mi,ij,mj->m", w, G, w)
            c = w / np.sqrt(np.maximum(q, 1e-300))[:, None]
            cosstep = 1 - np.clip(np.einsum("mi,ij,mj->m", c[:-1], G, c[1:]), -1, 1)
            # GPR不確かさ・エントロピー・最近文書距離
            u = gp.rel_uncertainty(path)
            H = sph_entropy(w, N)
            nd = np.linalg.norm(path[:, None] - P[None], axis=2).min(axis=1)
            # L1情報幾何 (フル語彙)
            if Xl1_full is None:
                Xl1_full = ((X + EPS) / l1s[:, None]).astype(np.float64)
            pi = w @ Xl1_full
            pi = pi / pi.sum(1, keepdims=True)
            def js_pair(a, b):
                mm = 0.5 * (a + b)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t1 = np.where(a > 0, a * np.log(np.maximum(a, 1e-300) / mm), 0).sum(1)
                    t2 = np.where(b > 0, b * np.log(np.maximum(b, 1e-300) / mm), 0).sum(1)
                return 0.5 * t1 + 0.5 * t2
            jss = js_pair(pi[:-1], pi[1:])
            hel = np.linalg.norm(np.sqrt(pi[:-1]) - np.sqrt(pi[1:]), axis=1) / np.sqrt(2)
            bc = np.clip((np.sqrt(pi[:-1] * pi[1:])).sum(1), 0, 1)
            frl = float(np.sum(2 * np.arccos(bc)))
            # 各法の自計量エネルギーと状態
            st = ""
            E_own = E_sph
            if m in ("sph", "gpr1", "gpr4", "gpr9", "fr"):
                rec = json.load(open(f"{DATA}/geo_{pid}_{m}.json"))
                st = rec["status"]; E_own = rec["E_final"]
            row = dict(pair=pid, doc_s=i, doc_t=j, method=m,
                       sph_energy=E_sph, sph_length=L_sph, energy_reduction=R_E,
                       cos_smoothness=float(cosstep.mean()), max_semantic_jump=float(cosstep.max()),
                       mean_gpr_uncertainty=float(u.mean()), mean_sph_entropy=float(H.mean()),
                       nearest_doc_dist=float(nd.mean()), js_smoothness=float(jss.mean()),
                       hellinger_smoothness=float(hel.mean()), fisher_rao_length=frl,
                       own_energy=E_own, status=st)
            rows.append(row)
            print(pid, m, "done", f"{time.time()-t0:.0f}s", flush=True)
    import csv
    with open(OUT + "/results_metrics.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)
    # geodesic_results.csv
    with open(OUT + "/geodesic_results.csv", "w", newline="") as f:
        cols = ["pair", "docs", "method", "E_final", "E_straight", "status",
                "singular_regularized_points", "n_multistart"]
        w_ = csv.writer(f); w_.writerow(cols)
        for pid in pairs:
            for m in ["sph", "gpr1", "gpr4", "gpr9", "fr"]:
                fp = f"{DATA}/geo_{pid}_{m}.json"
                if os.path.exists(fp):
                    r = json.load(open(fp))
                    w_.writerow([pid, r["docs"], m, r["E_final"], r["E_straight"],
                                 r["status"], r["singular_regularized_points"], len(r["tries"])])
    print("eval done")

print("elapsed", round(time.time() - t0, 1))
