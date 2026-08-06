# -*- coding: utf-8 -*-
"""Knowledge Manifold v5.2 共通ライブラリ.

評価指標・SPH/GPR・測地線は NumPy で実装している。TF-IDF(TfidfVectorizer)、
SMACOF 初期値、L-BFGS-B、Delaunay、GPR カーネル最適化のみ sklearn/scipy を使う。

KM_OUT   成果物の出力先        (run_v50.sh が設定する)
KM_DATA  中間ファイルの置き場  (既定は KM_OUT/work)
"""
import numpy as np, json, heapq, os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("KM_OUT", os.path.join(_ROOT, "outputs", "run"))
DATA = os.environ.get("KM_DATA", os.path.join(OUT, "work"))

# ---------------- データ ----------------
def load_tfidf():
    X = np.load(DATA + "/X_raw.npy")
    l2n = np.load(DATA + "/l2_norms.npy")
    l1s = np.load(DATA + "/l1_sums.npy")
    return X, l2n, l1s

def cosine_dist_matrix(Xl2):
    G = Xl2 @ Xl2.T
    return np.clip(1.0 - G, 0.0, 2.0)

# ---------------- アンカー割当 (§2.2.2) ----------------
# 縮退許容の相対誤差。アンカー配置(4隅+4辺中点+中央)は正方形の二面体群 D4 の対称性を
# 持つため、最適割当は厳密に8個縮退する(実測の縮退幅 ~5e-15 = 0〜3 ulp)。次のコスト
# 水準までのギャップは相対 5.6e-3、BLAS/SciPy 版差による Gram のノイズは相対 5.1e-9 で、
# 1e-9 は両側に約6桁の余裕がある。
ANCHOR_TIE_RTOL = 1e-9

def assign_anchors(D, reps, anchor_pos, n_perim, tie_rtol=ANCHOR_TIE_RTOL):
    """代表文書を固定アンカー位置へ割り当てる (全順列を厳密探索).

    最小コストを `cost < best` の逐次比較だけで採ると、上記の縮退の勝者が丸め誤差で
    決まってしまい、BLAS/SciPy のビルドが変わるとマップ全体が鏡映・回転する。座標は
    変わるのに対距離ベースの指標は不変なので、検証ゲートを通り抜けてしまう。そこで
    許容誤差で同点集合を切り出し、文書番号タプルの辞書順最小でゲージを固定する。

    戻り値 (perm, alpha, n_tied): perm は anchor_pos の各位置に入る reps のインデックス。
    """
    import itertools
    reps = np.asarray(reps)
    a_total = len(anchor_pos)
    A2D = np.linalg.norm(anchor_pos[:, None] - anchor_pos[None], axis=2)
    iu_a = np.triu_indices(a_total, 1)
    iu_p = np.triu_indices(n_perim, 1)
    mean2D_p = A2D[:n_perim, :n_perim][iu_p].mean()
    cands = []
    for perm in itertools.permutations(range(a_total)):
        docs = reps[list(perm)]
        d_sub = D[np.ix_(docs, docs)]
        alpha = mean2D_p / d_sub[:n_perim, :n_perim][iu_p].mean()
        cost = float(np.sum((A2D[iu_a] - alpha * d_sub[iu_a]) ** 2))
        cands.append((cost, tuple(int(x) for x in docs), perm, alpha))
    cost_min = min(c[0] for c in cands)
    # 絶対床を足しておく: cost_min が 0 近傍だと相対許容だけでは縮退を捉えられない
    # (実データでは cost_min ~ 11.2 なので床は効かず、同点数は 8 のまま)。
    thresh = tie_rtol * abs(cost_min) + 1e-12
    tied = [c for c in cands if c[0] - cost_min <= thresh]
    _, _, perm, alpha = min(tied, key=lambda c: c[1])
    return perm, alpha, len(tied)

# ---------------- Delaunay (scipy) ----------------
def delaunay(pts):
    from scipy.spatial import Delaunay as _D
    try:
        tri = _D(np.asarray(pts, float))
        return [tuple(int(v) for v in t) for t in tri.simplices]
    except Exception:
        return []

# ---------------- Dijkstra ----------------
def dijkstra(adj, s):
    """adj: dict node -> list of (nbr, w)."""
    n = len(adj)
    dist = {v: np.inf for v in adj}
    prev = {v: None for v in adj}
    dist[s] = 0.0
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev

def knn_graph(D, k):
    """cosine距離行列からkNNグラフ(対称化)を構築."""
    N = len(D)
    adj = {i: [] for i in range(N)}
    edges = set()
    for i in range(N):
        nn = np.argsort(D[i])
        cnt = 0
        for j in nn:
            if j == i: continue
            e = tuple(sorted((i, int(j))))
            if e not in edges:
                edges.add(e)
            cnt += 1
            if cnt >= k: break
    for i, j in edges:
        adj[i].append((j, float(D[i, j])))
        adj[j].append((i, float(D[i, j])))
    return adj

def is_connected(adj):
    seen = {0}; stack = [0]
    while stack:
        u = stack.pop()
        for v, _ in adj[u]:
            if v not in seen:
                seen.add(v); stack.append(v)
    return len(seen) == len(adj)

# ---------------- SPH ----------------
def cubic_spline_W(q):
    q = np.asarray(q, float)
    W = np.zeros_like(q)
    m1 = q < 1
    m2 = (q >= 1) & (q < 2)
    W[m1] = 1 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
    W[m2] = 0.25 * (2 - q[m2]) ** 3
    return W

def sph_weights(P, Y, h_mode="global", h_fixed=None, knn_k=8):
    """P:(...,2) 評価点, Y:(N,2) 文書座標. 戻り値 w:(...,N)."""
    P = np.atleast_2d(P)
    d = np.linalg.norm(P[:, None, :] - Y[None, :, :], axis=2)  # (M,N)
    if h_mode == "global":
        h = d.max(axis=1, keepdims=True) / 1.98
    elif h_mode == "fixed":
        h = np.full((len(P), 1), h_fixed)
    elif h_mode == "knn_adaptive":
        h = np.sort(d, axis=1)[:, [knn_k]] / 1.0
    elif h_mode == "density_adaptive":
        h = np.sort(d, axis=1)[:, [max(2, knn_k // 2)]] * 1.5
    else:
        raise ValueError(h_mode)
    W = cubic_spline_W(d / np.maximum(h, 1e-12))
    s = W.sum(axis=1, keepdims=True)
    # 全ゼロ回避: 最近傍にフォールバック
    zero = (s[:, 0] == 0)
    if zero.any():
        idx = d[zero].argmin(axis=1)
        W[zero] = 0; W[zero, idx] = 1.0
        s = W.sum(axis=1, keepdims=True)
    return W / s

def sph_entropy(w, N=None):
    N = N or w.shape[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(w > 0, w * np.log(w), 0.0)
    return -t.sum(axis=-1) / np.log(N)

# ---------------- L2場のGram表現 ----------------
class L2Field:
    """v(P)=normalize(Σ w_i x̂_i). Gram行列で内積を厳密計算."""
    def __init__(self, G, Y, h_mode="global", h_fixed=None, knn_k=8):
        self.G, self.Y = G, Y
        self.kw = dict(h_mode=h_mode, h_fixed=h_fixed, knn_k=knn_k)
    def coef(self, P):
        w = sph_weights(P, self.Y, **self.kw)         # (M,N)
        q = np.einsum("mi,ij,mj->m", w, self.G, w)     # ||Σ w x̂||²
        return w / np.sqrt(np.maximum(q, 1e-300))[:, None]
    def inner(self, c1, c2):
        return np.einsum("mi,ij,mj->m", c1, self.G, c2)
    def cos_between(self, P1, P2):
        c1, c2 = self.coef(P1), self.coef(P2)
        return np.clip(self.inner(c1, c2), -1, 1)
    def metric(self, P, delta=1e-3):
        """g_ab = <∂a v, ∂b v> at each P. 戻り値 (M,2,2)."""
        P = np.atleast_2d(P)
        ex = np.array([delta, 0.0]); ey = np.array([0.0, delta])
        cxp, cxm = self.coef(P + ex), self.coef(P - ex)
        cyp, cym = self.coef(P + ey), self.coef(P - ey)
        dx = (cxp - cxm) / (2 * delta)
        dy = (cyp - cym) / (2 * delta)
        g = np.empty((len(P), 2, 2))
        g[:, 0, 0] = self.inner(dx, dx)
        g[:, 1, 1] = self.inner(dy, dy)
        g[:, 0, 1] = g[:, 1, 0] = self.inner(dx, dy)
        return g

# ---------------- GPR (RBF/Matern + White, ARD 2D入力) ----------------
class GPR:
    def __init__(self, kernel="rbf", seed=0, n_restarts=10):
        self.kernel = kernel; self.seed = seed; self.n_restarts = n_restarts
    def _K(self, A, B, th):
        c, lx, ly = np.exp(th[0]), np.exp(th[1]), np.exp(th[2])
        d2 = ((A[:, None, 0] - B[None, :, 0]) / lx) ** 2 + ((A[:, None, 1] - B[None, :, 1]) / ly) ** 2
        if self.kernel == "rbf":
            return c * np.exp(-0.5 * d2)
        if self.kernel == "rbf_iso":
            l = np.exp(th[1])
            d2i = ((A[:, None, 0] - B[None, :, 0]) / l) ** 2 + ((A[:, None, 1] - B[None, :, 1]) / l) ** 2
            return c * np.exp(-0.5 * d2i)
        if self.kernel == "matern32":
            r = np.sqrt(np.maximum(d2, 0))
            return c * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
        if self.kernel == "matern52":
            r = np.sqrt(np.maximum(d2, 0))
            return c * (1 + np.sqrt(5) * r + 5 * d2 / 3) * np.exp(-np.sqrt(5) * r)
        raise ValueError(self.kernel)
    def _lml(self, th, X, Y, use_white=True):
        n = len(X)
        noise = np.exp(th[3]) if use_white else 1e-10
        K = self._K(X, X, th) + noise * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e12
        a = np.linalg.solve(L.T, np.linalg.solve(L, Y))
        m = Y.shape[1]
        return float(-0.5 * np.sum(Y * a) - m * np.sum(np.log(np.diag(L))) - 0.5 * n * m * np.log(2 * np.pi))
    def fit(self, X, Y, use_white=True):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel, Matern
        if self.kernel == "rbf":
            base = C(1.0) * RBF([0.5, 0.5])
        elif self.kernel == "rbf_iso":
            base = C(1.0) * RBF(0.5)
        elif self.kernel == "matern32":
            base = C(1.0) * Matern([0.5, 0.5], nu=1.5)
        elif self.kernel == "matern52":
            base = C(1.0) * Matern([0.5, 0.5], nu=2.5)
        else:
            raise ValueError(self.kernel)
        k = base + WhiteKernel(0.01) if use_white else base
        gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=self.n_restarts,
                                      random_state=self.seed, alpha=1e-10, normalize_y=False)
        gp.fit(X, Y)
        prod = gp.kernel_.k1 if use_white else gp.kernel_
        c = float(prod.k1.constant_value)
        ls = np.atleast_1d(prod.k2.length_scale)
        lx = float(ls[0]); ly = float(ls[1]) if len(ls) > 1 else float(ls[0])
        noise = float(gp.kernel_.k2.noise_level) if use_white else 1e-10
        self.theta = np.array([np.log(c), np.log(lx), np.log(ly), np.log(max(noise, 1e-300))])
        self.lml_ = float(gp.log_marginal_likelihood_value_)
        self.X, self.Y_ = X, Y
        self.noise = noise; self.use_white = use_white
        K = self._K(X, X, self.theta) + noise * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y))
        self._sk = gp
        return self
    def predict(self, P, return_std=True):
        Ks = self._K(P, self.X, self.theta)
        mu = Ks @ self.alpha
        if not return_std:
            return mu
        v = np.linalg.solve(self.L, Ks.T)
        kss = np.exp(self.theta[0])
        var = np.maximum(kss - np.sum(v * v, axis=0), 1e-15)
        return mu, np.sqrt(var)
    def rel_uncertainty(self, P):
        _, s = self.predict(np.atleast_2d(P))
        return s / np.sqrt(np.exp(self.theta[0]))   # σ_post/σ_prior

# ---------------- 最適化: Nelder-Mead / L-BFGS ----------------
def nelder_mead(f, x0, maxiter=500, seed=0, step=0.5, tol=1e-10):
    n = len(x0)
    sim = [np.array(x0, float)]
    for i in range(n):
        x = np.array(x0, float); x[i] += step
        sim.append(x)
    fs = [f(x) for x in sim]
    for _ in range(maxiter):
        o = np.argsort(fs)
        sim = [sim[i] for i in o]; fs = [fs[i] for i in o]
        if abs(fs[-1] - fs[0]) < tol:
            break
        cen = np.mean(sim[:-1], axis=0)
        xr = cen + (cen - sim[-1]); fr = f(xr)
        if fr < fs[0]:
            xe = cen + 2 * (cen - sim[-1]); fe = f(xe)
            if fe < fr: sim[-1], fs[-1] = xe, fe
            else: sim[-1], fs[-1] = xr, fr
        elif fr < fs[-2]:
            sim[-1], fs[-1] = xr, fr
        else:
            xc = cen + 0.5 * (sim[-1] - cen); fc = f(xc)
            if fc < fs[-1]:
                sim[-1], fs[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    sim[i] = sim[0] + 0.5 * (sim[i] - sim[0]); fs[i] = f(sim[i])
    o = np.argsort(fs)
    return sim[o[0]]

def lbfgs(f_and_g, x0, maxiter=2000, m=10, ftol=1e-12, gtol=1e-8, maxls=50, bounds=None):
    """scipy L-BFGS-B ラッパ. 戻り値 (x, f, success, nit)."""
    from scipy.optimize import minimize
    res = minimize(f_and_g, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                   bounds=bounds,
                   options=dict(maxiter=maxiter, maxcor=m, ftol=ftol, gtol=gtol, maxls=maxls))
    return res.x, float(res.fun), bool(res.success), int(res.nit)

# ---------------- 統計 ----------------
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = _tie_rank(a); rb = _tie_rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))

def _tie_rank(a):
    order = np.argsort(a)
    ranks = np.empty(len(a), float)
    sa = np.asarray(a)[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks

def trustworthiness_continuity(D_high, D_low, k=7):
    """T&C (Venna & Kaski)."""
    N = len(D_high)
    rh = np.argsort(np.argsort(D_high + np.eye(N) * 1e9, axis=1), axis=1)
    rl = np.argsort(np.argsort(D_low + np.eye(N) * 1e9, axis=1), axis=1)
    T = 0.0; C = 0.0
    for i in range(N):
        nl = set(np.where(rl[i] < k)[0]); nh = set(np.where(rh[i] < k)[0])
        for j in nl - nh:
            T += rh[i, j] - k + 1
        for j in nh - nl:
            C += rl[i, j] - k + 1
    norm = 2.0 / (N * k * (2 * N - 3 * k - 1))
    return 1 - norm * T, 1 - norm * C

def knn_preservation(D_high, D_low, k=7):
    """高次元・低次元の k 近傍集合の重なり率 (0〜1).

    自己は対角に 1e9 を足すことで除外する。以前は自己除外を二重に書いていて
    (`[:k+1]` と `- {i}` の併用)、マスク済みで自分は末尾に来るため `- {i}` が
    無効化され、k+1 近傍を k で割っていた。同一 geometry で (k+1)/k を返す
    (k=7 で 1.143) 状態だった。trustworthiness_continuity は上の `rl[i] < k`
    により元から k 個ちょうどを見ていて、こちらの影響は受けていない。
    """
    N = len(D_high)
    p = 0.0
    for i in range(N):
        nh = set(np.argsort(D_high[i] + np.eye(N)[i] * 1e9)[:k])
        nl = set(np.argsort(D_low[i] + np.eye(N)[i] * 1e9)[:k])
        p += len(nh & nl) / k
    return p / N

def normalized_stress(D_target, D_low):
    iu = np.triu_indices(len(D_target), 1)
    t, r = D_target[iu], D_low[iu]
    return float(np.sum((r - t) ** 2) / np.sum(t ** 2))

def procrustes_dist(A, B):
    """相似変換を許すProcrustes距離 (正規化)."""
    A = A - A.mean(0); B = B - B.mean(0)
    na, nb = np.linalg.norm(A), np.linalg.norm(B)
    if na == 0 or nb == 0: return np.nan
    A, B = A / na, B / nb
    U, s, Vt = np.linalg.svd(A.T @ B)
    return float(np.sqrt(max(0.0, 1 - s.sum() ** 2)))

def kmeans(X, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    # k-means++
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min([(np.linalg.norm(X - c, axis=1) ** 2) for c in C], axis=0)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(np.linalg.norm(X[:, None] - C[None], axis=2), axis=1)
        Cn = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(Cn, C): break
        C = Cn
    return lab, C

def ari(l1, l2):
    from math import comb
    n = len(l1)
    cats1, cats2 = sorted(set(l1)), sorted(set(l2))
    M = np.zeros((len(cats1), len(cats2)), int)
    for a, b in zip(l1, l2):
        M[cats1.index(a), cats2.index(b)] += 1
    sij = sum(comb(int(v), 2) for v in M.flat if v >= 2)
    si = sum(comb(int(v), 2) for v in M.sum(1) if v >= 2)
    sj = sum(comb(int(v), 2) for v in M.sum(0) if v >= 2)
    sn = comb(n, 2)
    exp = si * sj / sn
    mx = (si + sj) / 2
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


# ---------------- 言語化レンズ用の語フィルタ ----------------
import re as _re
_TERM_RE = _re.compile(r"[a-z0-9\- ]+")
def load_stoplist(path=None):
    """term_stoplist.txt (1行1語, #コメント) を読む。無ければ空。"""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "term_stoplist.txt")
    if os.path.exists(path):
        return [l.strip().lower() for l in open(path) if l.strip() and not l.startswith("#")]
    return []
def term_ok(t, stoplist=()):
    """言語化に用いてよい n-gram か。TeX/数式/記号断片と著者名・書誌断片を除外する。"""
    t = t.strip().lower()
    if len(t) < 4: return False
    if not _TERM_RE.fullmatch(t): return False
    if sum(c.isalpha() for c in t) < 4: return False
    if any(t in name for name in stoplist): return False
    return True
