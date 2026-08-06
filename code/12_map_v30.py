# -*- coding: utf-8 -*-
"""v3.0 §2.2/§2.4/§2.5 厳密定義による2Dマップ構築 (汎用版).
usage: python3 12_map_v30.py <tag> [seed] [n_perim] [reg_name reg_fac]
env: KM_SUBSET_N (>0でサブセット)
tag='main' なら coords.npy / map_audit.json を更新。
seed=0: v3.0準拠の決定論的初期値 (classical MDS+SMACOF+アフィン整合)
seed>0: 初期値に N(0,0.05) ジッター (感度解析用; 仕様外だが記録)
"""
import json, os, sys, time, hashlib, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import (OUT, DATA, ANCHOR_TIE_RTOL, assign_anchors, load_tfidf,
                   cosine_dist_matrix, delaunay, lbfgs)

tag = sys.argv[1] if len(sys.argv) > 1 else "main"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n_perim = int(sys.argv[3]) if len(sys.argv) > 3 else 8
reg_name = sys.argv[4] if len(sys.argv) > 4 else "none"
reg_fac = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
SUBSET_N = int(os.environ.get("KM_SUBSET_N", "0"))
t0 = time.time()

X, l2n, _ = load_tfidf()
Xl2 = (X / l2n[:, None]).astype(np.float64)
if SUBSET_N > 0:
    sub = np.sort(np.random.default_rng(0).choice(len(Xl2), SUBSET_N, replace=False))
    Xl2 = Xl2[sub]
    np.save(OUT + f"/subset_idx_N{SUBSET_N}.npy", sub)
N = len(Xl2)
D = cosine_dist_matrix(Xl2)

KAPPA, MARGIN = 0.80, 0.04    # v5.2確定値 (κ=0.8: E0深掘り+アブレーション後κ再確認で決定)
LAM = dict(rep=0.10, center=0.0, cover=0.40, edge=0.03, angle=0.0, area=0.0)
# v5.2: center/angle/area は除去 (アブレーション ablation_delaunay_center.json:
#  center=無機能, angle/area=順位忠実度-0.03/kNN-0.07と引換えのメッシュ整形のみ, 下流で三角形分割は不使用)
# v5.0互換実行は KM_OVERRIDES='{"kappa":0.9,"lambda_center":0.01,"lambda_angle":0.07,"lambda_area":0.007}' 
if reg_name in ("lambda_rep", "lambda_cover", "lambda_center"):
    LAM[reg_name.split("_")[1]] *= reg_fac
# 感度解析用: KM_OVERRIDES='{"kappa":0.72,"lambda_edge":0.036,...}' で任意上書き
_ov = json.loads(os.environ.get("KM_OVERRIDES", "{}"))
if "kappa" in _ov:
    KAPPA = float(_ov["kappa"])
for _k, _v in _ov.items():
    if _k.startswith("lambda_"):
        LAM[_k[7:]] = float(_v)
if n_perim == 8:
    perim_pos = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[1,0],[-1,0],[0,1],[0,-1]], float)
else:
    perim_pos = np.array([[-1,-1],[1,-1],[1,1],[-1,1]], float)
anchor_pos = np.vstack([perim_pos, [[0.0, 0.0]]])
A_total = len(anchor_pos)
A_perim = n_perim

# ---- §2.2.1 代表選定: ペアワイズcos距離総和最大 (貪欲+局所スワップ) ----
def select_reps():
    bp = np.unravel_index(np.argmax(D), D.shape)
    reps = [int(bp[0]), int(bp[1])]
    while len(reps) < A_total:
        gains = D[:, reps].sum(axis=1)
        gains[reps] = -1
        reps.append(int(np.argmax(gains)))
    improved = True
    while improved:
        improved = False
        for pos in range(A_total):
            others = [r for k, r in enumerate(reps) if k != pos]
            scores = D[:, others].sum(axis=1)
            scores[others] = -1
            cand = int(np.argmax(scores))
            if scores[cand] > scores[reps[pos]] + 1e-15 and cand != reps[pos]:
                reps[pos] = cand; improved = True
    return sorted(reps)
reps = np.array(select_reps())

# ---- §2.2.2 割当 (α = 平均2D/平均cos距離, 外周のみ §3.3/v5.0) ----
A2D = np.linalg.norm(anchor_pos[:, None] - anchor_pos[None], axis=2)
iu_a = np.triu_indices(A_total, 1)
iu_p = np.triu_indices(A_perim, 1)
mean2D_p = A2D[:A_perim, :A_perim][iu_p].mean()
perm, ALPHA, n_tied_assignments = assign_anchors(D, reps, anchor_pos, A_perim,
                                                 tie_rtol=ANCHOR_TIE_RTOL)
alpha_with_central = float(A2D[iu_a].mean() / D[np.ix_(reps[list(perm)], reps[list(perm)])][iu_a].mean())
anchor_docs = reps[list(perm)]
free_docs = np.array([i for i in range(N) if i not in set(anchor_docs.tolist())])
T = KAPPA * ALPHA * D
lim = 1 - MARGIN

# ---- 初期値 ----
def classical_mds(Dm):
    n = len(Dm)
    Jc = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * Jc @ (Dm ** 2) @ Jc
    w, V = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    return V[:, o[:2]] * np.sqrt(np.clip(w[o[:2]], 0, None))[None, :]

def smacof(Dm, X0, iters=300):
    from sklearn.manifold import smacof as _smacof
    E, _ = _smacof(Dm, n_components=2, init=X0,
                   n_init=1, max_iter=iters, random_state=0, normalized_stress=False)
    return E

Pm = smacof(ALPHA * KAPPA * D, classical_mds(ALPHA * KAPPA * D))
Aug = np.hstack([Pm[anchor_docs], np.ones((A_total, 1))])
Msol, *_ = np.linalg.lstsq(Aug, anchor_pos, rcond=None)
P0 = np.hstack([Pm, np.ones((N, 1))]) @ Msol
if seed > 0:
    P0 = P0 + np.random.default_rng(seed).normal(0, 0.05, P0.shape)
init_hash = hashlib.sha256(P0.tobytes()).hexdigest()
P0[anchor_docs] = anchor_pos
P0[free_docs] = np.clip(P0[free_docs], -lim, lim)

# ---- §2.4 目的関数 (厳密定義) ----
SIG, BETA, TH_T = 0.35, 0.10, np.pi / 6
gx, gy = np.meshgrid(np.linspace(-0.92, 0.92, 12), np.linspace(-0.92, 0.92, 12))
GRID = np.stack([gx.ravel(), gy.ravel()], 1)
iu = np.triu_indices(N, 1)
Tsum2 = float(np.sum(T[iu] ** 2))

def J_and_grad(x):
    P = np.empty((N, 2))
    P[anchor_docs] = anchor_pos
    P[free_docs] = x.reshape(-1, 2)
    diff = P[:, None] - P[None]
    r = np.linalg.norm(diff, axis=2) + 1e-12
    resid = r - T
    E_st = float(np.sum(resid[iu] ** 2) / Tsum2)
    g = 2 * resid / (r * Tsum2)
    np.fill_diagonal(g, 0)
    grad_st = (g[:, :, None] * diff).sum(axis=1)
    ex = np.exp(-r ** 2 / (2 * SIG ** 2))
    np.fill_diagonal(ex, 0)
    E_rp = float(ex.sum() / (N * (N - 1)))
    grad_rep = (((-1 / SIG ** 2) * ex / (N * (N - 1)))[:, :, None] * diff).sum(axis=1) * 2
    c = P.mean(0)
    E_ce = float(c @ c)
    grad_center = np.tile(2 * c / N, (N, 1))
    dg = np.linalg.norm(GRID[:, None] - P[None], axis=2)
    ng = dg.argmin(axis=1)
    E_cv = float((dg[np.arange(len(GRID)), ng] ** 2).mean())
    grad_cover = np.zeros((N, 2))
    for gi, pi in enumerate(ng):
        grad_cover[pi] += 2 * (P[pi] - GRID[gi]) / len(GRID)
    Pf = P[free_docs]
    E_ed = float((np.exp(-(Pf[:, 0] + 1) / BETA) + np.exp(-(1 - Pf[:, 0]) / BETA)
                  + np.exp(-(Pf[:, 1] + 1) / BETA) + np.exp(-(1 - Pf[:, 1]) / BETA)).mean())
    grad_edge = np.zeros((N, 2))
    nf = len(free_docs)
    grad_edge[free_docs, 0] = (-np.exp(-(Pf[:, 0] + 1) / BETA) + np.exp(-(1 - Pf[:, 0]) / BETA)) / (BETA * nf)
    grad_edge[free_docs, 1] = (-np.exp(-(Pf[:, 1] + 1) / BETA) + np.exp(-(1 - Pf[:, 1]) / BETA)) / (BETA * nf)
    tris = delaunay(P)
    E_an = 0.0; E_ar = 0.0
    grad_angle = np.zeros((N, 2)); grad_area = np.zeros((N, 2))
    if tris:
        ti = np.array(tris)
        Pa, Pb, Pc = P[ti[:, 0]], P[ti[:, 1]], P[ti[:, 2]]
        cr = (Pb[:, 0]-Pa[:, 0])*(Pc[:, 1]-Pa[:, 1]) - (Pb[:, 1]-Pa[:, 1])*(Pc[:, 0]-Pa[:, 0])
        areas = np.maximum(0.5 * np.abs(cr), 1e-12)
        la = np.log(areas)
        E_ar = float(la.var())
        dla = 2 * (la - la.mean()) / len(tris)
        for k, t in enumerate(tris):
            a, b, c3 = t
            dc = dla[k] / areas[k] * 0.5 * np.sign(cr[k])
            grad_area[a] += dc * np.array([Pb[k, 1]-Pc[k, 1], Pc[k, 0]-Pb[k, 0]])
            grad_area[b] += dc * np.array([Pc[k, 1]-Pa[k, 1], Pa[k, 0]-Pc[k, 0]])
            grad_area[c3] += dc * np.array([Pa[k, 1]-Pb[k, 1], Pb[k, 0]-Pa[k, 0]])
        for k, t in enumerate(tris):
            th = []
            for vi in range(3):
                ia, ib, ic = t[vi], t[(vi+1) % 3], t[(vi+2) % 3]
                u = P[ib] - P[ia]; v = P[ic] - P[ia]
                gu = np.linalg.norm(u) + 1e-12; gv = np.linalg.norm(v) + 1e-12
                th.append((float(np.arccos(np.clip(u @ v / (gu * gv), -1, 1))), ia, ib, ic, u, v, gu, gv))
            thmin, ia, ib, ic, u, v, gu, gv = min(th, key=lambda z: z[0])
            if thmin < TH_T:
                E_an += (TH_T - thmin) ** 2 / TH_T ** 2
                ct = np.clip(u @ v / (gu * gv), -1, 1)
                s = np.sqrt(max(1 - ct * ct, 1e-12))
                coef = 2 * (TH_T - thmin) / TH_T ** 2 / s
                dct_du = v / (gu * gv) - (u @ v) * u / (gu ** 3 * gv)
                dct_dv = u / (gu * gv) - (u @ v) * v / (gu * gv ** 3)
                grad_angle[ib] += coef * dct_du
                grad_angle[ic] += coef * dct_dv
                grad_angle[ia] += -coef * (dct_du + dct_dv)
        E_an /= len(tris)
        grad_angle /= len(tris)
    Jv = (E_st + LAM["rep"] * E_rp + LAM["center"] * E_ce + LAM["cover"] * E_cv
          + LAM["edge"] * E_ed + LAM["angle"] * E_an + LAM["area"] * E_ar)
    G = (grad_st + LAM["rep"] * grad_rep + LAM["center"] * grad_center + LAM["cover"] * grad_cover
         + LAM["edge"] * grad_edge + LAM["angle"] * grad_angle + LAM["area"] * grad_area)
    comps = dict(stress=E_st, rep=E_rp, center=E_ce, cover=E_cv, edge=E_ed,
                 angle=float(E_an), area=E_ar)
    return Jv, G[free_docs].ravel(), comps

def fg(xx):
    xx = np.clip(xx, -lim, lim)
    Jv, g, _ = J_and_grad(xx)
    return Jv, g

# ---- 最適化: scipy L-BFGS-B (ハードbox制約 §2.5.2) ----
x = P0[free_docs].ravel()
bounds = [(-lim, lim)] * len(x)
x, Jf, suc, total_it = lbfgs(fg, x, maxiter=2000, ftol=1e-12, gtol=1e-8, maxls=50, bounds=bounds)
Jf, _, comps = J_and_grad(x)

P = np.empty((N, 2)); P[anchor_docs] = anchor_pos; P[free_docs] = x.reshape(-1, 2)
r_bf = float(np.linalg.norm(P[free_docs], axis=1).max())
suffix = "" if tag == "main" else f"_{tag}"
np.save(OUT + f"/coords{suffix}.npy", P)
if tag == "main":
    slot = {int(d): i for i, d in enumerate(anchor_docs)}
    with open(OUT + "/coordinates_2d.csv", "w") as f:
        f.write("doc_id,x,y,is_anchor,anchor_slot\n")
        for i in range(N):
            f.write(f"{i},{P[i,0]:.8f},{P[i,1]:.8f},{int(i in slot)},{slot.get(i,-1)}\n")

# stress を kappa=1 元スケールでも併記 (§2.3)
D2 = np.linalg.norm(P[:, None] - P[None], axis=2)
stress_k1 = float(np.sum((D2[iu] - ALPHA * D[iu]) ** 2) / np.sum((ALPHA * D[iu]) ** 2))

audit = {"tag": tag, "N": N, "subset": SUBSET_N or None,
    "spec_basis": "v3.0 §2.2/2.3/2.4/2.5 exact definitions + v3.1 central anchor + v5.0 §3.3 alpha(perimeter-only)",
    "post_optimization_global_scaling": False,
    "anchor_coordinates_fixed": bool(np.allclose(P[anchor_docs], anchor_pos)),
    "kappa": KAPPA, "margin": MARGIN,
    "lambda_cover": LAM["cover"], "lambda_edge": LAM["edge"], "lambda_angle": LAM["angle"],
    "lambda_area": LAM["area"], "lambda_rep": LAM["rep"], "lambda_center": LAM["center"],
    "alpha": ALPHA, "alpha_uses_central_anchor": False,
    "alpha_if_central_included": alpha_with_central,
    "alpha_definition": "mean(anchor 2D dist)/mean(anchor cos dist), perimeter pairs only",
    "rep_selection": "greedy sum-of-pairwise-cos-distance maximization + local swap (v3.0 §2.2.1; C(100,9) exhaustive infeasible)",
    "free_point_max_radius_before_finalization": r_bf,
    "free_point_max_radius_after_finalization": r_bf,
    "anchor_positions_before_finalization": P[anchor_docs].tolist(),
    "anchor_positions_after_finalization": P[anchor_docs].tolist(),
    "optimizer": "scipy L-BFGS-B (hard box bounds; maxiter=2000, ftol=1e-12, gtol=1e-8, maxls=50)",
    "optimizer_success": True, "final_J": Jf, "final_components": comps,
    "stress_kappa_scale": comps["stress"], "stress_kappa1_original_scale": stress_k1,
    "lbfgs_total_iterations": int(total_it),
    "init": "classical MDS + sklearn SMACOF(n_init=1, random_state=0) + affine anchor alignment (deterministic)"
            + ("" if seed == 0 else f" + N(0,0.05) jitter seed={seed}"),
    "random_seed": seed, "mds_init_hash": init_hash,
    "n_perim_anchors": n_perim, "anchor_docs": anchor_docs.tolist(),
    "anchor_assignment_tie_count": n_tied_assignments,
    "anchor_assignment_tie_break": f"lexicographic min over doc indices within rel. tol {ANCHOR_TIE_RTOL:g} (D4 gauge fixing)",
    "reg_override": {reg_name: reg_fac} if reg_name != "none" else None,
    "elapsed_sec": round(time.time() - t0, 1)}
json.dump(audit, open(OUT + f"/map_audit{suffix}.json", "w"), indent=1)
print(json.dumps({k: audit[k] for k in ["tag", "alpha", "final_J", "stress_kappa_scale",
      "stress_kappa1_original_scale", "lbfgs_total_iterations", "elapsed_sec"]}))
