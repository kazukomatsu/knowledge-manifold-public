# -*- coding: utf-8 -*-
"""後処理: 完成した run のマップを固定したまま、任意の k で再クラスタリングする.

原作は共著者 (川越) の code/recluster.py (repository_release_20260915.zip)。クラスタの
定義は原作・パイプラインと同じで、SVD 潜在の先頭 10 次元に kmeans(Z, k, seed=0)
(kmlib.kmeans, k-means++ 初期化) を当てる (05_geodesics.py の標準 k=5 と同じ呼び出し)。
番号は文献数の多い順に 0,1,2,... へ振り直す。文献数が同じクラスタは k-means の元の
番号順に並べる (安定ソート)。座標・アンカー・場・標準の cluster_labels.npy には触れない。

クラスタ番号は run と k ごとに固有で、別の run・別の k の番号とは意味が対応しない。
k=5 の結果は標準の cluster_labels.npy と同じ分割を番号だけ付け替えたものになる
(clusters_k5_meta.json の standard_partition に対応表を記録する)。

usage:
  python3 code/recluster.py --out <KM_OUT> --k 4 6 7 [--data <KM_DATA>] [--outdir <dir>]
      [--no-terms] [--force]
出力 (既定 <KM_OUT>/recluster/, k ごとに別ファイル):
  cluster_labels_k{K}.npy   coordinates_2d_k{K}.csv (doc_id x y is_anchor cluster)
  clusters_k{K}.json        文献数・上位 n-gram・代表文献・2D 重心
  clusters_k{K}_meta.json   入力 hash・設定・環境・標準分割との対応
  fig_map_k{K}.png          fig_map_k{K}.dat (gnuplot: doc x y cluster is_anchor)
--no-terms は X_raw.npy / vocab.json / corpus_metadata.csv を読まず、上位 n-gram と
代表文献を省く (同梱の data/derived だけでも動く)。
"""
import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
from kmlib import kmeans, load_stoplist, term_ok
from postproc_lib import (ArtifactError, check_run, load_docs_clean, load_labels, load_metadata,
                          load_vocab, provenance, refuse_overwrite, refuse_repo_data, rel_or_none,
                          require_files, run_cli, sha256_file, write_json)

SVD_DIMS = 10        # 05_geodesics.py の標準クラスタリングと同じ
SEED = 0
TERM_POOL = 600      # 上位 n-gram の候補数 (原作どおり。語彙が小さければ語彙数まで)


def validate_ks(ks, Z):
    if len(set(ks)) != len(ks):
        raise ValueError(f"--k has duplicates: {ks}")
    n_distinct = len(np.unique(Z, axis=0))
    bad = [k for k in ks if not 2 <= k <= n_distinct]
    if bad:
        raise ValueError(f"k must be between 2 and the number of distinct documents in the "
                         f"SVD space ({n_distinct}); got {bad}")


def size_ranked_kmeans(Z, K, seed=SEED):
    """kmeans の番号を文献数の降順 (同数は元の番号順) に振り直す. (labels, raw_labels) を返す."""
    raw, _ = kmeans(Z, K, seed=seed)
    counts = np.bincount(raw, minlength=K)
    if (counts == 0).any():
        raise ArtifactError(f"k={K}: k-means left {int((counts == 0).sum())} cluster(s) empty "
                            f"(sizes {counts.tolist()}); this k cannot be realised on this map")
    order = np.argsort(-counts, kind="stable")
    remap = np.empty(K, dtype=np.int64)
    remap[order] = np.arange(K)
    return remap[raw], raw


def top_terms(vec, vocab, stop, n=8):
    """原作どおり: 重心ベクトルの値の大きい n-gram から、語フィルタと部分文字列の重複を除いて n 個."""
    pool = min(TERM_POOL, len(vec))
    cand = np.argpartition(vec, -pool)[-pool:]
    cand = cand[np.argsort(vec[cand])[::-1]]
    words = []
    for i in cand:
        t = vocab[i].strip()
        if not term_ok(t, stop):
            continue
        if any(t in x or x in t for x in words):
            continue
        words.append(t)
        if len(words) >= n:
            break
    return words


def standard_partition(lab, std):
    """標準ラベルと同じ分割か. 同じなら {標準番号: 再番号} を返す."""
    pairs = sorted(set(zip(std.tolist(), lab.tolist())))
    same = len(pairs) == len(set(std.tolist())) == len(set(lab.tolist()))
    return {"same_partition": bool(same),
            "standard_to_recluster": {str(a): b for a, b in pairs} if same else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="パイプラインの出力ディレクトリ (KM_OUT)")
    ap.add_argument("--data", help="中間ファイルのディレクトリ (KM_DATA, 既定 <KM_OUT>/work)")
    ap.add_argument("--k", type=int, nargs="+", required=True, help="クラスタ数 (複数可)")
    ap.add_argument("--outdir", help="出力先 (既定 <KM_OUT>/recluster)")
    ap.add_argument("--no-terms", action="store_true",
                    help="上位 n-gram と代表文献を省く (X_raw.npy / vocab.json / corpus_metadata.csv 不要)")
    ap.add_argument("--force", action="store_true", help="既存の k 別出力を上書きする")
    a = ap.parse_args(argv)
    OUT = os.path.abspath(a.out)
    DATA = os.path.abspath(a.data or os.path.join(OUT, "work"))
    OD = os.path.abspath(a.outdir or os.path.join(OUT, "recluster"))
    refuse_repo_data(OD)

    need = {"coords.npy": os.path.join(OUT, "coords.npy"),
            "map_audit.json": os.path.join(OUT, "map_audit.json"),
            "svd_scores.npy": os.path.join(DATA, "svd_scores.npy")}
    if not a.no_terms:
        need.update({"corpus_metadata.csv": os.path.join(OUT, "corpus_metadata.csv"),
                     "X_raw.npy": os.path.join(DATA, "X_raw.npy"),
                     "l2_norms.npy": os.path.join(DATA, "l2_norms.npy"),
                     "vocab.json": os.path.join(DATA, "vocab.json")})
    require_files(need)

    P = np.load(need["coords.npy"])
    N = len(P)
    if P.shape != (N, 2) or not np.isfinite(P).all():
        raise ArtifactError(f"{need['coords.npy']}: expected finite (N, 2) coordinates, got {P.shape}")
    Zall = np.load(need["svd_scores.npy"])
    if Zall.ndim != 2 or len(Zall) != N:
        raise ArtifactError(f"{need['svd_scores.npy']}: expected {N} rows, got {Zall.shape}")
    Z = Zall[:, :SVD_DIMS]              # 05_geodesics.py と同じ切り方 (列が 10 未満ならあるだけ)
    with open(need["map_audit.json"], encoding="utf-8") as f:
        audit = json.load(f)
    if "anchor_docs" not in audit:
        raise ArtifactError(f"{need['map_audit.json']}: no anchor_docs")
    anchors = sorted(set(int(d) for d in audit["anchor_docs"]))
    if any(not 0 <= d < N for d in anchors):
        raise ArtifactError(f"{need['map_audit.json']}: anchor_docs outside 0..{N - 1}")
    validate_ks(a.k, Z)

    if not a.no_terms:
        meta = load_metadata(need["corpus_metadata.csv"], N)
        X = np.load(need["X_raw.npy"])
        l2n = np.load(need["l2_norms.npy"])
        if X.shape[0] != N or l2n.shape != (N,):
            raise ArtifactError(f"X_raw.npy {X.shape} / l2_norms.npy {l2n.shape} do not match N={N}")
        Xl2 = (X / l2n[:, None]).astype(np.float32)
        vocab = load_vocab(need["vocab.json"], X.shape[1])
        stop = load_stoplist()

    # 手元にある成果物どうしが同じ run のものか (無いものは照合せず meta に記録)
    opt = lambda p: p if os.path.isfile(p) else None
    gram_p, gpr_p = opt(os.path.join(DATA, "gram_l2.npy")), opt(os.path.join(DATA, "gpr_model.pkl"))
    docs_p = None if a.no_terms else opt(os.path.join(OUT, "corpus", "docs_clean.json"))
    gpr = None
    if gpr_p:
        with open(gpr_p, "rb") as f:
            gpr = pickle.load(f)
    checks = check_run(OUT, P, Z=Zall, G=np.load(gram_p) if gram_p else None, gpr=gpr,
                       X=None if a.no_terms else X, l2n=None if a.no_terms else l2n,
                       vocab=None if a.no_terms else vocab,
                       docs=load_docs_clean(docs_p, meta) if docs_p else None,
                       meta=None if a.no_terms else meta)
    if not a.no_terms:
        del X

    std_path = os.path.join(DATA, "cluster_labels.npy")
    std = load_labels(std_path, N)[0] if os.path.isfile(std_path) else None

    # 全 k を先に計算する: どれか 1 つでも実現できなければ何も書かない
    results = {K: size_ranked_kmeans(Z, K) for K in a.k}
    names = lambda K: [f"{OD}/{s}" for s in (f"cluster_labels_k{K}.npy", f"coordinates_2d_k{K}.csv",
                                             f"clusters_k{K}.json", f"clusters_k{K}_meta.json",
                                             f"fig_map_k{K}.png", f"fig_map_k{K}.dat")]
    refuse_overwrite([p for K in a.k for p in names(K)], a.force)
    os.makedirs(OD, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    inputs = {k: {"path": rel_or_none(p, OUT), "sha256": sha256_file(p)} for k, p in need.items()}
    prov = provenance(CODE, [os.path.abspath(__file__), os.path.join(CODE, "kmlib.py"),
                             os.path.join(CODE, "postproc_lib.py")])

    for K in a.k:
        lab, raw = results[K]
        sizes = np.bincount(lab, minlength=K)
        np.save(f"{OD}/cluster_labels_k{K}.npy", lab)
        with open(f"{OD}/coordinates_2d_k{K}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "x", "y", "is_anchor", "cluster"])
            for i in range(N):
                w.writerow([i, f"{P[i, 0]:.8f}", f"{P[i, 1]:.8f}", int(i in anchors), int(lab[i])])
        with open(f"{OD}/fig_map_k{K}.dat", "w") as f:
            f.write("# doc x y cluster(size-ranked) is_anchor\n")
            for i in range(N):
                f.write(f"{i} {P[i, 0]:.6f} {P[i, 1]:.6f} {lab[i]} {int(i in anchors)}\n")
        summ = []
        for c in range(K):
            m = np.where(lab == c)[0]
            rec = dict(cluster=c, size=int(len(m)), top_ngrams=None, representative_doc=None,
                       centroid_xy=[round(float(P[m, 0].mean()), 3), round(float(P[m, 1].mean()), 3)])
            if not a.no_terms:
                cen = Xl2[m].mean(0)
                rep = int(m[np.argmax(Xl2[m] @ (cen / (np.linalg.norm(cen) + 1e-12)))])
                rec["top_ngrams"] = top_terms(cen, vocab, stop)
                rec["representative_doc"] = dict(doc=rep, title=meta[rep].get("title", "")[:80])
            summ.append(rec)
        write_json(f"{OD}/clusters_k{K}.json", summ)
        write_json(f"{OD}/clusters_k{K}_meta.json", {
            "tool": "recluster.py", "k": K, "n_documents": N, "sizes": sizes.tolist(),
            "clustering": {"features": f"svd_scores.npy[:, :{SVD_DIMS}]",
                           "algorithm": "kmlib.kmeans (k-means++ init, iters=100)", "seed": SEED,
                           "numbering": "descending cluster size; equal sizes keep the raw k-means order"},
            "raw_kmeans_labels_of_ranked_clusters": [int(raw[lab == c][0]) for c in range(K)],
            "terms": None if a.no_terms else {
                "top_ngrams": "raw char n-grams with the largest values of the cluster's mean "
                              "L2-normalised TF-IDF vector (no subtraction of the corpus mean), "
                              f"term_ok filter, substring de-duplication, pool {TERM_POOL}",
                "representative_doc": "cluster member with the largest cosine to that mean vector"},
            "standard_partition": None if std is None else dict(
                standard_labels_sha256=sha256_file(std_path), **standard_partition(lab, std)),
            "note": "cluster numbers are specific to this run and this k; they do not correspond "
                    "to the numbers of another run or another k",
            "consistency_checks": checks, "inputs": inputs, "provenance": prov})
        fig, ax = plt.subplots(figsize=(7.2, 6.6))
        for c in range(K):
            m = P[lab == c]
            ax.scatter(m[:, 0], m[:, 1], s=18, color=cmap(c % 10), label=f"C{c} (n={len(m)})")
        A = P[anchors]
        ax.scatter(A[:, 0], A[:, 1], marker="s", s=60, facecolor="none", edgecolor="k", linewidth=1.1)
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ls=":", color="gray"))
        ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"k = {K} (labels size-ranked; map fixed)", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(f"{OD}/fig_map_k{K}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
        note = "  (k > 10: tab10 colours repeat; use the legend)" if K > 10 else ""
        print(f"k={K}: sizes {sizes.tolist()}  -> {OD}{note}")
    print("done")


if __name__ == "__main__":
    run_cli(main)
