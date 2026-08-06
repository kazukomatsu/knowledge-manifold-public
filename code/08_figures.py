# -*- coding: utf-8 -*-
"""§11 図表生成. usage: python3 08_figures.py <group> [--fullfig]
group: maps | fields | paths | reports

既定では論文で使用する図のみを出力する。--fullfig を付けると、
概念図・診断用の補助図（figC / figD / figR1_scaling）も生成する。
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import OUT, DATA

FIG = OUT + "/figures"
os.makedirs(FIG, exist_ok=True)
group = sys.argv[1]
FULL = "--fullfig" in sys.argv[2:]   # 補助図（概念図・診断図）まで出すか
t0 = time.time()
P = np.load(OUT + "/coords.npy")
lab = np.load(DATA + "/cluster_labels.npy")
audit = json.load(open(OUT + "/map_audit.json"))
anch = audit["anchor_docs"]
pairs = json.load(open(DATA + "/pairs.json"))
CL = plt.cm.tab10(np.linspace(0, 1, 10))
METHOD_STYLE = {"line": ("k", "--", "Euclidean line"), "graph": ("tab:orange", "-.", "Graph geodesic"),
                "sph": ("tab:blue", "-", "SPH geodesic"), "gpr1": ("tab:green", "-", "GPR-w $\\lambda$=1"),
                "gpr4": ("tab:red", "-", "GPR-w $\\lambda$=4"), "gpr9": ("tab:purple", "-", "GPR-w $\\lambda$=9"),
                "fr": ("tab:brown", ":", "Fisher-Rao geodesic")}

def draw_map(ax, annotate=False):
    for c in range(lab.max() + 1):
        m = lab == c
        ax.scatter(P[m, 0], P[m, 1], s=28, color=CL[c], label=f"cluster {c}", zorder=3)
    ax.scatter(P[anch, 0], P[anch, 1], marker="s", s=90, facecolor="none",
               edgecolor="k", linewidth=1.5, zorder=4, label="anchors")
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ls=":", color="gray"))
    if annotate:
        for i in range(len(P)):
            ax.annotate(str(i), P[i], fontsize=5, alpha=0.6)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")

if group == "maps":
    # Figure A
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_map(ax, annotate=True)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Figure A: 2D knowledge map (N=100, 8 perimeter + central anchors)")
    fig.savefig(FIG + "/figA_knowledge_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    if FULL:
        # Figure C: L2 vs L1 schematic
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        th = np.linspace(0, np.pi / 2, 100)
        axes[0].plot(np.cos(th), np.sin(th), "k-")
        for a, c in [(0.35, "tab:blue"), (0.6, "tab:red"), (1.1, "tab:green")]:
            axes[0].annotate("", xy=(np.cos(a), np.sin(a)), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="->", color=c, lw=2))
        axes[0].annotate("", xy=(1.25, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="gray"))
        axes[0].annotate("", xy=(0, 1.25), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="gray"))
        axes[0].set_title("L2: directions on unit sphere $S^{V-1}$\n(cosine / angular distance)")
        axes[0].set_aspect("equal"); axes[0].axis("off")
        tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
        axes[1].plot(tri[:, 0], tri[:, 1], "k-")
        pts = np.array([[0.3, 0.15], [0.55, 0.35], [0.45, 0.55]])
        for p_, c in zip(pts, ["tab:blue", "tab:red", "tab:green"]):
            axes[1].scatter(*p_, color=c, s=70, zorder=3)
        axes[1].plot(pts[:2, 0], pts[:2, 1], "k:", lw=1)
        axes[1].text(0.02, -0.06, "$e_1$"); axes[1].text(0.95, -0.06, "$e_2$"); axes[1].text(0.48, 0.9, "$e_3$")
        axes[1].set_title("L1: distributions on simplex $\\Delta^{V-1}$\n(JS / Hellinger / Fisher-Rao)")
        axes[1].set_aspect("equal"); axes[1].axis("off")
        fig.suptitle("Figure C: L2 direction geometry vs L1 information geometry")
        fig.savefig(FIG + "/figC_L2_vs_L1_schematic.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    # R1: map thumbnails N=20 vs N=100
    P20 = np.load(OUT + "/coords_N20.npy")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(P20[:, 0], P20[:, 1], s=30, c="tab:blue")
    axes[0].set_title("N=20 (subset) map"); axes[0].set_aspect("equal")
    axes[0].add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ls=":", color="gray"))
    for c in range(lab.max() + 1):
        m = lab == c
        axes[1].scatter(P[m, 0], P[m, 1], s=20, color=CL[c])
    axes[1].set_title("N=100 map"); axes[1].set_aspect("equal")
    axes[1].add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ls=":", color="gray"))
    fig.suptitle("R1: map thumbnails by data size")
    fig.savefig(FIG + "/figR1_map_thumbnails.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("maps done")

elif group == "fields":
    gz = np.load(DATA + "/grid_fields.npz")
    xs, H, U, g = gz["xs"], gz["entropy"], gz["uncertainty"], gz["g"]
    ext = [xs[0], xs[-1], xs[0], xs[-1]]
    if FULL:
        # Figure D: metric tensor heatmaps
        tr = g[:, :, 0, 0] + g[:, :, 1, 1]
        det = g[:, :, 0, 0] * g[:, :, 1, 1] - g[:, :, 0, 1] ** 2
        ev = np.linalg.eigvalsh(g.reshape(-1, 2, 2)).reshape(g.shape[0], g.shape[1], 2)
        aniso = ev[:, :, 1] / np.maximum(ev[:, :, 0], 1e-300)
        coup = g[:, :, 0, 1] / np.sqrt(np.maximum(g[:, :, 0, 0] * g[:, :, 1, 1], 1e-300))
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, Z, name, cm in [(axes[0, 0], tr, "tr(g)", "viridis"),
                                (axes[0, 1], np.log10(np.maximum(det, 1e-12)), "log10 det(g)", "viridis"),
                                (axes[1, 0], aniso, "anisotropy $\\lambda_1/\\lambda_2$", "magma"),
                                (axes[1, 1], coup, "coupling $g_{12}/\\sqrt{g_{11}g_{22}}$", "coolwarm")]:
            im = ax.imshow(Z, origin="lower", extent=ext, cmap=cm, aspect="equal")
            ax.scatter(P[:, 0], P[:, 1], s=6, c="w", edgecolor="k", linewidth=0.3)
            ax.set_title(name); fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle("Figure D: SPH-induced metric tensor fields (L2, global h, $\\delta$=1e-3)")
        fig.savefig(FIG + "/figD_metric_tensor.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    # Figure E: GPR uncertainty
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(U, origin="lower", extent=ext, cmap="inferno", aspect="equal")
    ax.scatter(P[:, 0], P[:, 1], s=10, c="cyan", edgecolor="k", linewidth=0.3)
    fig.colorbar(im, ax=ax, label=r"$u(r)=\sigma_{post}/\sigma_{prior}$")
    ax.set_title("Figure E: GPR relative uncertainty (research-gap indicator)")
    fig.savefig(FIG + "/figE_gpr_uncertainty.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    # Figure F: SPH entropy
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(H, origin="lower", extent=ext, cmap="cividis", aspect="equal")
    ax.scatter(P[:, 0], P[:, 1], s=10, c="w", edgecolor="k", linewidth=0.3)
    fig.colorbar(im, ax=ax, label="$H_{norm}$")
    ax.set_title("Figure F: SPH entropy (bridge / ambiguous region candidates)")
    fig.savefig(FIG + "/figF_sph_entropy.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("fields done")

elif group == "paths":
    gz = np.load(DATA + "/grid_fields.npz")
    xs, U = gz["xs"], gz["uncertainty"]
    ext = [xs[0], xs[-1], xs[0], xs[-1]]
    # Figure G: 6ペア × 全経路
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    for ax, (pid, (i, j)) in zip(axes.ravel(), pairs.items()):
        ax.scatter(P[:, 0], P[:, 1], s=10, c=[CL[c] for c in lab], alpha=0.45)
        for m in ["line", "graph", "sph", "gpr4", "fr"]:
            fp = f"{DATA}/path_{pid}_{m}.npy"
            if m == "line":
                t = np.linspace(0, 1, 31)[:, None]
                path = (1 - t) * P[i][None] + t * P[j][None]
            elif os.path.exists(fp):
                path = np.load(fp)
            else:
                continue
            c, ls, lb = METHOD_STYLE[m]
            ax.plot(path[:, 0], path[:, 1], color=c, ls=ls, lw=1.8, label=lb)
        ax.scatter(*P[i], marker="*", s=200, c="k", zorder=5)
        ax.scatter(*P[j], marker="*", s=200, c="k", zorder=5)
        ax.set_title(f"{pid}  (doc {i} → doc {j})", fontsize=10)
        ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Figure G: path comparison across metric families")
    fig.savefig(FIG + "/figG_path_comparison.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    # Figure H: lambda sensitivity (high_uncertainty pair) over uncertainty map
    pid = "high_uncertainty"; i, j = pairs[pid]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(U, origin="lower", extent=ext, cmap="inferno", alpha=0.85, aspect="equal")
    fig.colorbar(im, ax=ax, label="u(r)")
    for m, lam in [("sph", 0), ("gpr1", 1), ("gpr4", 4), ("gpr9", 9)]:
        path = np.load(f"{DATA}/path_{pid}_{m}.npy")
        c = {0: "w", 1: "tab:green", 4: "tab:red", 9: "tab:purple"}[lam]
        ax.plot(path[:, 0], path[:, 1], color=c, lw=2, label=f"$\\lambda$={lam}")
    ax.scatter(P[:, 0], P[:, 1], s=8, c="cyan", alpha=0.6)
    ax.scatter(*P[i], marker="*", s=220, c="w", edgecolor="k", zorder=5)
    ax.scatter(*P[j], marker="*", s=220, c="w", edgecolor="k", zorder=5)
    ax.legend(); ax.set_title(f"Figure H: GPR-weighted geodesics, $\\lambda$=0,1,4,9 ({pid})")
    fig.savefig(FIG + "/figH_lambda_sensitivity.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    # Figure I: path profile (far_max_cos)
    import csv
    from kmlib import sph_weights, sph_entropy, GPR
    import pickle
    X = np.load(DATA + "/X_raw.npy"); l2n = np.load(DATA + "/l2_norms.npy")
    l1s = np.load(DATA + "/l1_sums.npy")
    G = np.load(DATA + "/gram_l2.npy")
    md = pickle.load(open(DATA + "/gpr_model.pkl", "rb"))
    gp = GPR(kernel="rbf"); gp.theta = md["theta"]; gp.X = md["X"]; gp.alpha = md["alpha"]
    gp.L = md["L"]; gp.noise = md["noise"]
    Xl1 = ((X + 1e-10) / l1s[:, None]).astype(np.float64)
    # Figure I: path profile (論文と同じ2パネル: per-step 意味変化 + 混合エントロピー)
    # 文書グラフ経路は場の量ではなくマップへの射影を反映するため除外する
    pid = "far_max_cos"; i, j = pairs[pid]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    for m in ["line", "sph", "gpr4", "fr"]:
        if m == "line":
            t = np.linspace(0, 1, 31)[:, None]
            path = (1 - t) * P[i][None] + t * P[j][None]
        else:
            fp = f"{DATA}/path_{pid}_{m}.npy"
            if not os.path.exists(fp): continue
            path = np.load(fp)
        c, ls, lb = METHOD_STYLE[m]
        w = sph_weights(path, P, h_mode="global")
        q = np.einsum("mi,ij,mj->m", w, G, w)
        cf = w / np.sqrt(np.maximum(q, 1e-300))[:, None]
        cosstep = 1 - np.clip(np.einsum("mi,ij,mj->m", cf[:-1], G, cf[1:]), -1, 1)
        H = sph_entropy(w, len(P))
        s_ = np.linspace(0, 1, 31); sm = 0.5 * (s_[:-1] + s_[1:])
        axes[0].plot(sm, cosstep, color=c, ls=ls, lw=1.8, label=lb)
        axes[1].plot(s_, H, color=c, ls=ls, lw=1.8)
    axes[0].set_ylabel("per-step semantic change\n$1-\\cos$  ($\\ell^2$ field)", fontsize=9)
    axes[1].set_ylabel("mixture entropy $H$", fontsize=9)
    axes[1].set_xlabel("normalized arc position")
    # 右軸: 有効寄与文書数 N^H
    tw = axes[1].twinx(); tw.set_ylim(axes[1].get_ylim())
    tk = np.linspace(*axes[1].get_ylim(), 5)
    tw.set_yticks(tk); tw.set_yticklabels([f"{len(P)**h:.0f}" for h in tk], fontsize=8)
    tw.set_ylabel("effective no. of documents $N^{H}$", fontsize=9)
    for ax in axes: ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Figure I: path profiles ({pid}, doc {i} -> doc {j}; global h)")
    fig.tight_layout()
    fig.savefig(FIG + "/figI_path_profile.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("paths done")

elif group == "reports":
    import pickle
    from kmlib import GPR, spearman
    # R6: 論文と同じ2パネル (LOO 分布 + LOO と不確かさの関係)
    e2 = json.load(open(OUT + "/e2_loo_knn_adaptive.json"))
    cs = np.array(e2["per_doc_cosine"])
    G = np.load(DATA + "/gram_l2.npy")
    iu = np.triu_indices(len(P), 1)
    chance = float(G[iu].mean())
    md = pickle.load(open(DATA + "/gpr_model.pkl", "rb"))
    gp = GPR(kernel="rbf"); gp.theta = md["theta"]; gp.X = md["X"]
    gp.alpha = md["alpha"]; gp.L = md["L"]; gp.noise = md["noise"]
    u = gp.rel_uncertainty(P)
    r = np.linalg.norm(P, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].hist(cs, bins=22, color="tab:blue", alpha=0.85)
    axes[0].axvline(chance, color="k", ls="--", lw=1.8, label=f"chance = {chance:.3f}")
    axes[0].axvline(cs.mean(), color="tab:red", lw=1.8, label=f"mean = {cs.mean():.3f}")
    axes[0].set_xlabel("leave-one-out cosine to the held-out document")
    axes[0].set_ylabel("number of documents")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[0].set_title("(a)", loc="left", fontsize=11)

    sc = axes[1].scatter(u, cs, c=r, cmap="viridis", s=34, edgecolor="k", linewidth=0.3)
    rho = spearman(u, cs)
    z = np.polyfit(u, cs, 1); xs_ = np.linspace(u.min(), u.max(), 50)
    axes[1].plot(xs_, np.polyval(z, xs_), "r-", lw=1.6, label=f"$\\rho={rho:.2f}$")
    axes[1].axhline(chance, color="k", ls="--", lw=1.2)
    axes[1].set_xlabel("calibrated uncertainty $u$ at the document's position")
    axes[1].set_ylabel("leave-one-out cosine")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[1].set_title("(b)", loc="left", fontsize=11)
    fig.colorbar(sc, ax=axes[1], label="distance from map centre")
    fig.tight_layout()
    fig.savefig(FIG + "/figR6_evaluation.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    if FULL:
        # R1: scaling curve (診断用)
        r1 = json.load(open(OUT + "/r1_N20.json"))
        r5 = json.load(open(OUT + "/r1_N50.json"))
        rN = json.load(open(OUT + "/r1_N100.json"))
        NN = [r1["N"], r5["N"], rN["N"]]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(NN, [d["map_time_sec"] for d in (r1, r5, rN)], "o-", label="map optimization")
        axes[0].plot(NN, [d["loo_time_sec"] for d in (r1, r5, rN)], "s-", label="field LOO evaluation")
        axes[0].set_xlabel("N"); axes[0].set_ylabel("time (s)"); axes[0].legend()
        axes[0].set_title("computation time vs N"); axes[0].grid(alpha=0.3)
        axes[1].plot(NN, [d["peak_mem_mb"] for d in (r1, r5, rN)], "o-", color="tab:red")
        axes[1].set_xlabel("N"); axes[1].set_ylabel("peak memory (MB)")
        axes[1].set_title("peak memory vs N"); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG + "/figR1_scaling.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("reports done")

print("elapsed", round(time.time() - t0, 1))
