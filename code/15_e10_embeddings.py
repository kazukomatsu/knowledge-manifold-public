# -*- coding: utf-8 -*-
"""E10: 既存次元削減手法 (PCA / t-SNE / MDS / [UMAP]) との比較.
同一のTF-IDF cosine距離を入力に各手法で2D化し、
 (a) E1系指標: Spearman・kNN保存・trustworthiness/continuity
 (b) 本枠組み固有機能の可否: SPH-LOO補間力 (E2)・固定枠(アンカー)・決定論性
で本手法(constrained v5.0 map)と比較する。UMAPは環境に無い場合スキップし理由記録。
出力: KM_OUT/e10_embedding_comparison.json, figures/figE10_embeddings.png
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import (OUT, DATA, load_tfidf, cosine_dist_matrix, sph_weights,
                   spearman, knn_preservation, trustworthiness_continuity)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
P_ours = np.load(OUT + "/coords.npy")
X, l2n, _ = load_tfidf()
Xl2 = (X / l2n[:, None]).astype(np.float64)
D = cosine_dist_matrix(Xl2)
N = len(D)

def loo_cosine(P):
    cs = []
    for i in range(N):
        mask = np.arange(N) != i
        w = sph_weights(P[i][None], P[mask], h_mode="knn_adaptive", knn_k=8)[0]
        v = w @ Xl2[mask]
        nv = np.linalg.norm(v)
        cs.append(float(v @ Xl2[i] / nv) if nv > 0 else 0.0)
    return float(np.mean(cs)), float(np.std(cs))

def norm_box(P):
    P = P - P.mean(0)
    return P / max(np.abs(P).max(), 1e-12)

embeds = {}
notes = {}

# ours
embeds["v5.0_constrained"] = P_ours
notes["v5.0_constrained"] = "アンカー9点固定・目的関数J・決定論的(seed0で一意)"

# PCA (L2ベクトルのSVDスコア上位2 = 厳密PCA相当)
Z = np.load(DATA + "/svd_scores.npy")[:, :2]
embeds["PCA"] = norm_box(Z.copy())
notes["PCA"] = "線形・決定論的・大域構造のみ"

# MDS (metric, precomputed) — sklearnバージョン互換シム
from sklearn.manifold import MDS
import inspect
kw = dict(n_components=2, n_init=4, max_iter=300, random_state=0, normalized_stress="auto")
sig = inspect.signature(MDS.__init__).parameters
if "dissimilarity" in sig:
    kw["dissimilarity"] = "precomputed"
else:
    kw["metric"] = "precomputed"
mds = MDS(**kw)
embeds["MDS"] = norm_box(mds.fit_transform(D))
notes["MDS"] = "距離保存特化・初期値依存の局所解あり"

# t-SNE (precomputed distance)
from sklearn.manifold import TSNE
ts = TSNE(n_components=2, metric="precomputed", init="random", random_state=0,
          perplexity=15)
embeds["tSNE"] = norm_box(ts.fit_transform(D))
notes["tSNE"] = "局所近傍特化・大域距離は非保存・seed依存"

# UMAP (任意)
try:
    import umap
    um = umap.UMAP(n_components=2, metric="precomputed", random_state=0)
    embeds["UMAP"] = norm_box(um.fit_transform(D))
    notes["UMAP"] = "近傍グラフベース"
except ImportError:
    notes["UMAP"] = "未実施: umap-learn が環境に無い (wheels追加で実施可)"

rows = {}
for name, P in embeds.items():
    D2 = np.linalg.norm(P[:, None] - P[None], axis=2)
    iu = np.triu_indices(N, 1)
    t7, c7 = trustworthiness_continuity(D, D2, k=7)
    lc, ls = loo_cosine(P)
    rows[name] = {
        "spearman": round(spearman(D[iu], D2[iu]), 4),
        "knn_preservation_k7": round(knn_preservation(D, D2, k=7), 3),
        "trustworthiness_k7": round(t7, 3),
        "continuity_k7": round(c7, 3),
        "sph_loo_cosine": round(lc, 4),
        "sph_loo_std": round(ls, 4),
        "fixed_frame_anchors": name == "v5.0_constrained",
        "deterministic": name in ("v5.0_constrained", "PCA"),
        "note": notes[name],
    }
    print(name, rows[name], flush=True)

out = {"input": "TF-IDF char_wb cosine distance (共通)",
       "framework_specific_benefits": [
           "固定枠(アンカー9点): 実行間・コーパス間で座標系が比較可能",
           "有界領域[-1,1]^2: SPH/GPR連続場・測地線・回帰テストが定義可能",
           "決定論性: seed0で座標が一意に再現 (t-SNE/UMAPはseed依存)",
           "目標距離スケールt=καdの明示: stressが物理量として解釈可能"],
       "methods": rows, "umap_status": notes.get("UMAP"),
       "elapsed_sec": round(time.time() - t0, 1)}
json.dump(out, open(OUT + "/e10_embedding_comparison.json", "w"), ensure_ascii=False, indent=1)

# 各手法の2D座標を保存 (gnuplot等での再作図用; doc x y cluster)
gd = OUT + "/gnuplot_data"
os.makedirs(gd, exist_ok=True)
lab_ = np.load(DATA + "/cluster_labels.npy")
for name, Pm in embeds.items():
    safe = name.replace(".", "").replace("-", "_")
    with open(f"{gd}/fig3_{safe}.dat", "w") as fo:
        fo.write("# doc x y cluster\n")
        for i in range(N):
            fo.write(f"{i} {Pm[i,0]:.6f} {Pm[i,1]:.6f} {lab_[i]}\n")
print("gnuplot coords written:", gd)

# 図
ks = [k for k in embeds]
fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4.4))
for ax, name in zip(np.atleast_1d(axes), ks):
    P = embeds[name]
    ax.scatter(P[:, 0], P[:, 1], s=14, c="tab:blue")
    if name == "v5.0_constrained":
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ls=":", color="gray"))
    r = rows[name]
    ax.set_title(f"{name}\nSp={r['spearman']:.2f} kNN7={r['knn_preservation_k7']:.2f} LOO={r['sph_loo_cosine']:.2f}",
                 fontsize=9)
    ax.set_aspect("equal")
fig.suptitle("E10: embedding comparison on identical TF-IDF cosine distances")
os.makedirs(OUT + "/figures", exist_ok=True)
fig.savefig(OUT + "/figures/figE10_embeddings.png", dpi=140, bbox_inches="tight")
print("E10 done", out["elapsed_sec"], "s")
