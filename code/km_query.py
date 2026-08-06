# -*- coding: utf-8 -*-
"""Knowledge Manifold Query Kit — 自己完結の対話クエリモジュール (ChatGPT/Gemini/Claude用)
依存: numpy (必須), scipy (任意: 測地線最適化に使用。なければ勾配降下で代替)
データ: 同梱 data/ フォルダ (フル解析100MBを ~数MB に圧縮した凍結成果物)

使い方 (Python):
    import km_query as q
    q.point_semantics((-0.5, 0.3))          # 任意点の意味・エントロピー・不確かさ
    q.gradient_scan((-0.5, 0.3), r=0.15)    # 意味変化が最小/最大の方向
    q.find_gaps(k=3)                        # 手薄領域(research gap候補)
    q.trace_path("far_max_cos", n=20)       # 保存済み測地線をn点で言語化材料化
    q.compute_geodesic((0,0), (1,1))        # 新しい測地線をその場で計算
    q.doc_info(26)                          # 文書情報
CLI:
    python km_query.py point -0.5 0.3
    python km_query.py gap
    python km_query.py gradscan -0.5 0.3 0.15
    python km_query.py trace far_max_cos 20
    python km_query.py geodesic 0 0 1 1
"""
import json, os, sys, csv
import numpy as np

_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------- 凍結データの読み込み ----------
P = np.load(_D + "/coords.npy")                       # (N,2) 文書座標 (locked)
G = np.load(_D + "/gram_l2.npy")                      # (N,N) L2正規化TF-IDFのGram行列 (厳密内積)
_Xs = np.load(_D + "/X_sel.npy").astype(np.float32)   # (N,K) 文書特徴部分行列 (言語化用)
VOCAB = json.load(open(_D + "/vocab_sel.json", encoding="utf-8"))
_MEAN = np.load(_D + "/mean_sel.npy")                 # コーパス平均方向 (sel列)
CLUSTERS = np.load(_D + "/clusters.npy")
_g = np.load(_D + "/gpr.npz")                         # GPR凍結パラメータ
_PATHS = dict(np.load(_D + "/paths.npz"))             # 保存済み測地線 (31点×2)
PAIRS = json.loads(open(_D + "/pairs.json").read())
META = list(csv.DictReader(open(_D + "/meta_lite.csv", encoding="utf-8")))
N = len(P)

# ---------- プリミティブ1: SPH重み ----------
def sph_weights(pts, h_mode="knn_adaptive", knn_k=8):
    pts = np.atleast_2d(np.asarray(pts, float))
    d = np.linalg.norm(pts[:, None, :] - P[None, :, :], axis=2)
    if h_mode == "global":
        h = d.max(axis=1, keepdims=True) / 1.98
    else:
        h = np.sort(d, axis=1)[:, [knn_k]]
    q_ = d / np.maximum(h, 1e-12)
    W = np.zeros_like(q_)
    m1, m2 = q_ < 1, (q_ >= 1) & (q_ < 2)
    W[m1] = 1 - 1.5 * q_[m1] ** 2 + 0.75 * q_[m1] ** 3
    W[m2] = 0.25 * (2 - q_[m2]) ** 3
    s = W.sum(axis=1, keepdims=True)
    zero = s[:, 0] == 0
    if zero.any():
        idx = d[zero].argmin(axis=1)
        W[zero] = 0; W[zero, idx] = 1.0
        s = W.sum(axis=1, keepdims=True)
    return W / s

def entropy(w):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(w > 0, w * np.log(w), 0.0)
    return -t.sum(axis=-1) / np.log(N)

# ---------- プリミティブ2: L2場 (Gram表現で厳密) ----------
def _coef(pts, **kw):
    w = sph_weights(pts, **kw)
    q_ = np.einsum("mi,ij,mj->m", w, G, w)
    return w / np.sqrt(np.maximum(q_, 1e-300))[:, None]

def cos_between(p1, p2, **kw):
    c1, c2 = _coef(np.atleast_2d(p1), **kw), _coef(np.atleast_2d(p2), **kw)
    return np.clip(np.einsum("mi,ij,mj->m", c1, G, c2), -1, 1)

def metric(pts, delta=1e-3, **kw):
    pts = np.atleast_2d(np.asarray(pts, float))
    ex, ey = np.array([delta, 0.0]), np.array([0.0, delta])
    dx = (_coef(pts + ex, **kw) - _coef(pts - ex, **kw)) / (2 * delta)
    dy = (_coef(pts + ey, **kw) - _coef(pts - ey, **kw)) / (2 * delta)
    g = np.empty((len(pts), 2, 2))
    g[:, 0, 0] = np.einsum("mi,ij,mj->m", dx, G, dx)
    g[:, 1, 1] = np.einsum("mi,ij,mj->m", dy, G, dy)
    g[:, 0, 1] = g[:, 1, 0] = np.einsum("mi,ij,mj->m", dx, G, dy)
    return g

# ---------- プリミティブ3: GPR不確かさ u(r) ----------
def uncertainty(pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    th = _g["theta"]; Xg = _g["X"]; Kinv = _g["Kinv"]
    c, lx, ly = np.exp(th[0]), np.exp(th[1]), np.exp(th[2])
    d2 = ((pts[:, None, 0] - Xg[None, :, 0]) / lx) ** 2 + ((pts[:, None, 1] - Xg[None, :, 1]) / ly) ** 2
    Ks = c * np.exp(-0.5 * d2)
    var = np.maximum(c - np.einsum("mi,ij,mj->m", Ks, Kinv, Ks), 1e-15)
    return np.sqrt(var) / np.sqrt(c)

# ---------- プリミティブ4: 言語化材料 ----------
def top_features(v_sel, k=12):
    diff = v_sel - _MEAN
    words = []
    for i in np.argsort(diff)[::-1]:
        t = VOCAB[i].strip()
        if len(t) >= 4 and t.isascii() and not any(t in x or x in t for x in words):
            words.append(t)
        if len(words) >= k:
            break
    return words

def point_semantics(pt, k_feat=12, k_docs=3):
    """任意点の意味プロファイル。言語化はこの出力に基づくこと。"""
    pt = np.asarray(pt, float)
    w = sph_weights(pt[None])[0]
    v = w @ _Xs
    v = v / (np.linalg.norm(v) + 1e-300)
    docs = np.argsort(w)[::-1][:k_docs]
    return {
        "xy": [round(float(pt[0]), 3), round(float(pt[1]), 3)],
        "top_ngrams": top_features(v, k_feat),
        "contributing_docs": [{"doc": int(i), "weight": round(float(w[i]), 3),
                               "title": META[i]["title"][:70]} for i in docs if w[i] > 0.01],
        "entropy": round(float(entropy(w[None])[0]), 3),
        "gpr_uncertainty": round(float(uncertainty(pt[None])[0]), 3),
        "cluster_context": int(CLUSTERS[docs[0]]),
    }

# ---------- 応用1: 手薄領域 ----------
def find_gaps(k=3, box=0.8, n=61):
    xs = np.linspace(-box, box, n)
    gx, gy = np.meshgrid(xs, xs)
    GR = np.stack([gx.ravel(), gy.ravel()], 1)
    u = uncertainty(GR)
    nd = np.linalg.norm(GR[:, None] - P[None], axis=2).min(axis=1)
    un = (u - u.min()) / (u.max() - u.min() + 1e-300)
    ndn = (nd - nd.min()) / (nd.max() - nd.min() + 1e-300)
    score = un * ndn
    spots = []
    for i in np.argsort(score)[::-1]:
        if all(np.linalg.norm(GR[i] - np.array(s["xy"])) > 0.3 for s in spots):
            s = point_semantics(GR[i]); s["gap_score"] = round(float(score[i]), 3)
            spots.append(s)
        if len(spots) >= k:
            break
    return spots

# ---------- 応用2: 方向スキャン (局所計量+有限距離の両方; 検証内蔵) ----------
def gradient_scan(pt, r=0.15, ndir=72):
    pt = np.asarray(pt, float)
    g = metric(pt[None])[0]
    ev, evec = np.linalg.eigh(g)
    th = np.linspace(0, 2 * np.pi, ndir, endpoint=False)
    Q = pt[None] + r * np.stack([np.cos(th), np.sin(th)], 1)
    ch = 1 - cos_between(np.repeat(pt[None], ndir, 0), Q)
    i_min, i_max = int(np.argmin(ch)), int(np.argmax(ch))
    return {
        "point": pt.tolist(), "radius": r,
        "local_metric_min_direction_deg": round(float(np.degrees(np.arctan2(evec[1, 0], evec[0, 0])) % 180), 1),
        "local_anisotropy": round(float(ev[1] / max(ev[0], 1e-300)), 2),
        "finite_min_direction_deg": round(float(np.degrees(th[i_min])), 1),
        "finite_min_change": round(float(ch[i_min]), 5),
        "finite_max_direction_deg": round(float(np.degrees(th[i_max])), 1),
        "finite_max_change": round(float(ch[i_max]), 5),
        "note": "局所(計量)と有限距離(スキャン)の答えは場の非一様性により異なりうる。有限距離の値を優先し、両方を報告すること。",
    }

# ---------- 応用3: 経路トレース ----------
def _resample(path, n):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])], 1)

def trace_path(pair_or_path, method="sph", n=11):
    """保存済み測地線(pair名) または 座標配列 をn点で意味トレース。"""
    if isinstance(pair_or_path, str):
        key = f"path_{pair_or_path}_{method}"
        if key not in _PATHS:
            return {"error": f"保存済み経路がありません: {key}. 利用可能: {sorted(_PATHS)[:10]}... "
                             f"新規計算は compute_geodesic() を使うこと。"}
        path = _PATHS[key]
        docs = PAIRS.get(pair_or_path)
    else:
        path = np.asarray(pair_or_path, float)
        docs = None
    pts = _resample(path, n)
    steps = []
    prev_c = None
    for k in range(n):
        sem = point_semantics(pts[k])
        sem["t"] = round(k / (n - 1), 3)
        if prev_c is not None:
            sem["cos_change_from_prev"] = round(float(1 - cos_between(pts[k - 1][None], pts[k][None])[0]), 6)
        prev_c = True
        steps.append(sem)
    out = {"n_points": n, "waypoints": steps}
    if docs:
        out["from"] = {"doc": docs[0], "title": META[docs[0]]["title"][:70]}
        out["to"] = {"doc": docs[1], "title": META[docs[1]]["title"][:70]}
    return out

# ---------- 応用4: 新規測地線 (SPH計量, §6.1準拠の縮小版) ----------
def compute_geodesic(p_start, p_end, npts=31, maxiter=800):
    ps, pt_ = np.asarray(p_start, float), np.asarray(p_end, float)
    t = np.linspace(0, 1, npts)[:, None]
    base = (1 - t) * ps[None] + t * pt_[None]
    d = pt_ - ps
    perp = np.array([-d[1], d[0]]); perp /= (np.linalg.norm(perp) + 1e-12)
    DELTA = 1e-3
    def eg(x):
        path = np.vstack([ps, x.reshape(-1, 2), pt_])
        K = len(path) - 1
        dp = np.diff(path, axis=0)
        mid = 0.5 * (path[:-1] + path[1:])
        sh = np.array([[0, 0], [DELTA, 0], [-DELTA, 0], [0, DELTA], [0, -DELTA]])
        allq = (mid[:, None, :] + sh[None]).reshape(-1, 2)
        gall = metric(allq).reshape(K, 5, 2, 2)
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
    E_straight = eg(base[1:-1].ravel())[0]
    best = None
    for a in [0.0, 0.03, -0.03]:
        x0 = (base + a * np.sin(np.pi * t) * perp[None])[1:-1].ravel()
        try:
            from scipy.optimize import minimize
            res = minimize(eg, x0, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8))
            xf, Ef, nit = res.x, float(res.fun), int(res.nit)
        except ImportError:
            xf = x0.copy(); lr = 0.01; Ef = None
            for it in range(maxiter):
                E, gr = eg(xf); xf -= lr * gr; Ef = E
            nit = maxiter
        if best is None or Ef < best[1]:
            best = (xf, Ef, nit)
    xf, Ef, nit = best
    path = np.vstack([ps, xf.reshape(-1, 2), pt_])
    status = "success" if Ef <= E_straight else "failed(直線がエネルギー最小)"
    return {"E_straight": E_straight, "E_geodesic": Ef,
            "energy_reduction_percent": round((E_straight - Ef) / E_straight * 100, 1),
            "n_iterations": nit, "status": status, "path": path.round(4).tolist(),
            "note": "nit>0 と E_geodesic<=E_straight を必ず報告すること (実行深度の証明)"}

def doc_info(i):
    m = META[int(i)]
    return {"doc": int(i), "title": m["title"], "year": m.get("year", ""),
            "cluster": int(CLUSTERS[int(i)]), "xy": P[int(i)].round(3).tolist()}

# ---------- CLI ----------
if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__); sys.exit(0)
    cmd = a[0]
    if cmd == "point":
        out = point_semantics((float(a[1]), float(a[2])))
    elif cmd == "gap":
        out = find_gaps(int(a[1]) if len(a) > 1 else 3)
    elif cmd == "gradscan":
        out = gradient_scan((float(a[1]), float(a[2])), float(a[3]) if len(a) > 3 else 0.15)
    elif cmd == "trace":
        out = trace_path(a[1], a[2] if len(a) > 2 else "sph", int(a[3]) if len(a) > 3 else 11)
    elif cmd == "geodesic":
        out = compute_geodesic((float(a[1]), float(a[2])), (float(a[3]), float(a[4])))
    elif cmd == "doc":
        out = doc_info(int(a[1]))
    else:
        out = {"error": f"unknown command {cmd}"}
    print(json.dumps(out, ensure_ascii=False, indent=1))
