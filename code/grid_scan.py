# -*- coding: utf-8 -*-
"""後処理: 完成した run の固定マップ上で [-1,1]^2 の全格子点を走査する.

原作は共著者 (川越) の scan/k5_grid_scan.py (scan_code.zip)。各量の定義は原作のまま
で、変えたのは格子の一意性・入力の検証・出力先・決定論性・固有値の扱いだけ
(docs/postprocessing_ja.md に対応表)。再クラスタリング・再配置はしない。LLM は呼ばない。

各格子点について CSV に 1 行:
  空白度    d_nn, d_k5, d_k8 (1・5・8 番目に近い文献までの距離), nearest_doc_id
  混合      p_C0..p_C{K-1} = global SPH 重みのクラスタ和, H_cluster (log K で正規化)
  支持      H_field (log N で正規化), N_eff = N^H_field, H_field_adaptive (knn_k=8),
            max_doc_weight, top3_weight_share
  不確実性  u_RBF (run の gpr_model.pkl), u_M32 (Matern 3/2 を同じターゲットで再学習)
  計量      g_ab = <d_a v, d_b v> (L2 場, 中心差分 delta), 固有値 lam1 >= lam2, 主方向 e1,
            dirgrad = sqrt(lam), grad_norm = sqrt(tr g)
  フラグ    lam2_clipped, e1_ill_defined (主方向が数値的に定まらない), metric_kink (h が折れる点)
語彙 (--no-words で省略) は make_evidence.py と共通の evidence_lib.py (同点規則も同じ) で n-gram を語に復元して
details.json に書く。verbalization_50w 列は空欄で出す (文章化は LLM 側の別作業)。

usage:
  python3 code/grid_scan.py --out <KM_OUT> [--data <KM_DATA>] [--dx 0.1] [--delta 1e-3]
      [--labels <cluster_labels.npy> [--custom-labels]] [--topk-l2 10] [--topk-l1 8] [--no-words]
      [--outdir <dir>] [--force]
出力 (既定 <KM_OUT>/grid_scan/k{K}_dx{dx}_l2-{n}_l1-{n}/、数値のみは k{K}_dx{dx}_nowords/、
      delta を既定から変えたときは末尾に _delta{delta}):
  k{K}_grid_scan_raw.csv      1 行 = 1 格子点 (ix, iy は格子の添字)
  k{K}_grid_scan_details.json 各点のレンズ語と寄与文献 top10 (語彙付きのときのみ)
  k{K}_grid_scan_values.npz   丸めない値 (計量テンソル g を含む)
  k{K}_grid_scan_meta.json    格子・設定・許容誤差・入力 hash・環境 (再実行で不変)
  k{K}_grid_scan_runinfo.json 実行時刻・所要時間・コマンド (実行ごとに変わる)
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
from evidence_lib import CANDIDATE_POOL, WordResolver, pick_ngrams, rank_documents
from kmlib import GPR, L2Field, load_stoplist, sph_entropy, sph_weights
from postproc_lib import (ArtifactError, check_run, load_docs_clean, load_labels, load_metadata,
                          load_vocab, provenance, refuse_overwrite, refuse_repo_data, rel_or_none,
                          require_files, run_cli, sha256_file, write_json)

MAX_DECIMALS = 6        # 格子座標は小数 6 桁以内で厳密に書ける値に限る
MAX_PER_AXIS = 201      # dx >= 0.01 (40401 点) まで
MIN_DOCS = 9            # d_k8 は 8 番目, H_field_adaptive の h は 9 番目の近傍を使う
PROB_SUM_ATOL = 1e-9    # SPH 重みとクラスタ確率の和 = 1 の検査 (原作の assert と同じ)
EIG_NEG_RTOL = 1e-12    # lam2 < 0 でも |lam2| <= 1e-12 lam1 なら丸め誤差として 0 に切る
EIG_GAP_RTOL = 1e-3     # (lam1 - lam2) / lam1 がこれ未満なら主方向は数値的に定まらない
TOPK_DOCS = 10


# ---------------------------------------------------------------- 格子
def grid_axis(dx):
    """[-1,1] を間隔 dx で刻んだ軸 (xs, 小数桁数, 実際の間隔).

    2/dx が整数で、dx が小数 MAX_DECIMALS 桁以内で書ける場合だけ受ける。そうでない dx は
    np.linspace が黙って別の間隔に丸めてしまうため、記録と実体がずれないよう拒否する。
    """
    dx = float(dx)
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError(f"--dx must be a positive number, got {dx}")
    steps = int(round(2.0 / dx))
    if steps < 1 or abs(2.0 / dx - steps) > 1e-9 * (2.0 / dx):
        raise ValueError(f"--dx {dx} does not divide [-1, 1] evenly (2/dx = {2.0 / dx:.6g}); "
                         "use a dx for which 2/dx is an integer, e.g. 0.1, 0.05, 0.04, 0.025, 0.02")
    dec = next((d for d in range(1, MAX_DECIMALS + 1)
                if abs(dx * 10 ** d - round(dx * 10 ** d)) < 1e-6), None)
    if dec is None:
        raise ValueError(f"--dx {dx}: grid coordinates would need more than {MAX_DECIMALS} decimals")
    n = steps + 1
    if n > MAX_PER_AXIS:
        raise ValueError(f"--dx {dx} gives {n} points per axis; at most {MAX_PER_AXIS} (dx >= 0.01)")
    xs = np.round(np.linspace(-1.0, 1.0, n), dec) + 0.0     # + 0.0 で -0.0 を 0.0 にする
    return xs, dec, 2.0 / steps


def make_grid(xs):
    """行 r = ix * n + iy (x が外側) の格子点. 原作の meshgrid(indexing="ij").ravel() と同じ並び."""
    n = len(xs)
    ix, iy = np.divmod(np.arange(n * n), n)
    return np.column_stack([xs[ix], xs[iy]]), ix, iy


def point_key(x, y, dec):
    return f"{x:.{dec}f},{y:.{dec}f}"


# ---------------------------------------------------------------- 入力
def load_run(out, data, labels_path, words, custom_labels=False):
    """必要な成果物を読み、互いに同じ run のものかを検査する (postproc_lib.check_run)."""
    need = {"coords.npy": os.path.join(out, "coords.npy"),
            "corpus_metadata.csv": os.path.join(out, "corpus_metadata.csv"),
            "cluster_labels": labels_path,
            "gram_l2.npy": os.path.join(data, "gram_l2.npy"),
            "svd_scores.npy": os.path.join(data, "svd_scores.npy"),
            "gpr_model.pkl": os.path.join(data, "gpr_model.pkl")}
    if words:
        need.update({"X_raw.npy": os.path.join(data, "X_raw.npy"),
                     "l2_norms.npy": os.path.join(data, "l2_norms.npy"),
                     "vocab.json": os.path.join(data, "vocab.json"),
                     "docs_clean.json": os.path.join(out, "corpus", "docs_clean.json")})
    require_files(need)
    R = {"paths": need}
    P = np.load(need["coords.npy"])
    if P.ndim != 2 or P.shape[1] != 2 or not np.isfinite(P).all():
        raise ArtifactError(f"{need['coords.npy']}: expected finite (N, 2) coordinates, got {P.shape}")
    N = len(P)
    if N < MIN_DOCS:
        raise ArtifactError(f"the map has {N} documents; the scan needs at least {MIN_DOCS} "
                            "(d_k8 uses the 8th and H_field_adaptive the 9th nearest document)")
    R["P"], R["N"] = P, N
    R["meta"] = load_metadata(need["corpus_metadata.csv"], N)
    R["lab"], R["K"] = load_labels(labels_path, N)
    if R["K"] < 2:
        raise ArtifactError(f"{labels_path}: a single cluster; H_cluster needs K >= 2")
    G = np.load(need["gram_l2.npy"])
    if G.shape != (N, N) or not np.isfinite(G).all():
        raise ArtifactError(f"{need['gram_l2.npy']}: expected finite ({N}, {N}), got {G.shape}")
    if np.abs(G - G.T).max() > 1e-10 or np.abs(np.diag(G) - 1).max() > 1e-4:
        raise ArtifactError(f"{need['gram_l2.npy']}: not a symmetric Gram matrix of unit vectors")
    R["G"] = G
    Z = np.load(need["svd_scores.npy"])
    if Z.ndim != 2 or len(Z) != N:
        raise ArtifactError(f"{need['svd_scores.npy']}: expected {N} rows, got {Z.shape}")
    R["Z"] = Z[:, :10]                  # 04_fields.py gpr と同じ切り方 (列が 10 未満ならあるだけ)
    with open(need["gpr_model.pkl"], "rb") as f:
        gm = pickle.load(f)
    if gm.get("kernel", "rbf") != "rbf":
        raise ArtifactError(f"{need['gpr_model.pkl']}: kernel {gm.get('kernel')}, expected rbf")
    gp = GPR(kernel="rbf")
    for k in ("theta", "X", "alpha", "L", "noise"):
        setattr(gp, k, gm[k])
    R["gp"] = gp
    if words:
        X = np.load(need["X_raw.npy"])
        l2n = np.load(need["l2_norms.npy"])
        if X.ndim != 2 or len(X) != N or l2n.shape != (N,):
            raise ArtifactError(f"X_raw.npy {X.shape} / l2_norms.npy {l2n.shape} do not match N={N}")
        if not (l2n > 0).all():
            raise ArtifactError(f"{need['l2_norms.npy']}: a document has zero norm")
        R["X"], R["l2n"] = X, l2n
        R["vocab"] = load_vocab(need["vocab.json"], X.shape[1])
        R["docs"] = load_docs_clean(need["docs_clean.json"], R["meta"])
    R["checks"] = check_run(out, P, Z=Z, G=G, gpr=gm, X=R.get("X"), l2n=R.get("l2n"), vocab=R.get("vocab"),
                            docs=R.get("docs"), meta=R["meta"], labels=R["lab"], custom_labels=custom_labels)
    return R


# ---------------------------------------------------------------- 数値
def eigen_analysis(g):
    """2x2 計量テンソルの固有解析. 定義は原作どおり (lam1 >= lam2, e1 は e1x >= 0 に符号を揃える).

    g は Gram 行列による内積なので半正定値で、lam2 < 0 は丸め誤差でしか起きない。
    |lam2| <= EIG_NEG_RTOL * lam1 は 0 に切り (lam2_clipped)、それより負なら入力が壊れて
    いるとみなして失敗する。(lam1 - lam2) / lam1 < EIG_GAP_RTOL の点は主方向が数値的に
    定まらないので e1_ill_defined を立てる (向きを解釈しないこと)。
    """
    M = len(g)
    lam_raw = np.empty((M, 2))
    e1 = np.empty((M, 2))
    for i in range(M):
        vals, vecs = np.linalg.eigh(g[i])
        lam_raw[i] = vals[::-1]
        v1 = vecs[:, 1]
        if v1[0] < 0 or (v1[0] == 0 and v1[1] < 0):
            v1 = -v1
        e1[i] = v1
    if (lam_raw[:, 0] < 0).any():
        raise ArtifactError("metric tensor with a negative largest eigenvalue — gram_l2.npy is not PSD")
    tol = EIG_NEG_RTOL * lam_raw[:, 0]
    if (lam_raw[:, 1] < -tol).any():
        i = int(np.argmin(lam_raw[:, 1] / np.maximum(lam_raw[:, 0], 1e-300)))
        raise ArtifactError(f"metric tensor eigenvalue {lam_raw[i, 1]:.3e} (lam1 {lam_raw[i, 0]:.3e}) "
                            "is negative beyond rounding error")
    clipped = lam_raw[:, 1] < 0
    lam = lam_raw.copy()
    lam[clipped, 1] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.where(lam[:, 0] > 0, (lam[:, 0] - lam[:, 1]) / lam[:, 0], 0.0)
        aniso = np.where(lam[:, 1] > 0, lam[:, 0] / lam[:, 1], np.inf)
    theta1 = np.degrees(np.arctan2(e1[:, 1], e1[:, 0]))
    return dict(lam_raw=lam_raw, lam=lam, e1=e1, theta1=theta1, aniso=aniso, gap_rel=gap,
                ill=gap < EIG_GAP_RTOL, clipped=clipped, dirgrad=np.sqrt(lam))


def metric_kink(Q, P, delta, rtol=1e-12):
    """global SPH の h(P) = max_i |P - p_i| / 1.98 が中心差分のステンシル内で滑らかでない点.

    最遠の文献が入れ替わる (または同距離で並ぶ) 所で h は折れ、場 v(P) の左右の片側微分が一致しない。
    中心差分は delta -> 0 で片側微分の平均に近づき (誤差 O(delta))、g はその平均した微分どうしの
    内積から作られる。これは左右の片側計量の平均とは一般に異なり (折れを横切る成分では
    |d+v - d-v|^2 / 4 だけ小さい)、主方向も意味を持たない。隅の 4 点はアンカーで固定され、ほかの
    文献は箱拘束で内側にあるので、どの run でも x = 0 と y = 0 の線上が該当する。
    """
    def farthest(Qs):
        d = np.linalg.norm(Qs[:, None, :] - P[None, :, :], axis=2)
        s = np.sort(d, axis=1)
        return d.argmax(1), (s[:, -1] - s[:, -2]) <= rtol * s[:, -1]
    a0, kink = farthest(Q)
    for e in ((delta, 0.0), (-delta, 0.0), (0.0, delta), (0.0, -delta)):
        a, tie = farthest(Q + np.array(e))
        kink = kink | tie | (a != a0)
    return kink


def numeric_fields(Q, R, delta):
    P, lab, K, N = R["P"], R["lab"], R["K"], R["N"]
    M = len(Q)
    F = {}
    dmat = np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=2)
    F["near"] = dmat.argmin(1)
    dsort = np.sort(dmat, axis=1)
    F["d_nn"], F["d_k5"], F["d_k8"] = dsort[:, 0], dsort[:, 4], dsort[:, 7]
    w = sph_weights(Q, P, h_mode="global")
    if np.abs(w.sum(1) - 1).max() > PROB_SUM_ATOL:
        raise ArtifactError("SPH weights do not sum to 1")
    pc = np.zeros((M, K))
    for c in range(K):
        pc[:, c] = w[:, lab == c].sum(1)
    if np.abs(pc.sum(1) - 1).max() > PROB_SUM_ATOL:
        raise ArtifactError("cluster probabilities do not sum to 1")
    with np.errstate(divide="ignore", invalid="ignore"):
        F["H_cluster"] = -np.where(pc > 0, pc * np.log(pc), 0).sum(1) / np.log(K)
    # 同値のクラスタ確率は番号の小さい方を先にする (K 個程度の短い配列では原作の既定ソートと同じ結果)
    ordc = np.argsort(-pc, axis=1, kind="stable")
    F["dom"], F["sec"] = ordc[:, 0], ordc[:, 1]
    F["p_dom"] = pc[np.arange(M), F["dom"]]
    F["p_sec"] = pc[np.arange(M), F["sec"]]
    F["pc"] = pc
    F["H_field"] = sph_entropy(w, N)
    F["N_eff"] = N ** F["H_field"]
    F["max_w"] = w.max(1)
    F["top3"] = np.sort(w, axis=1)[:, -3:].sum(1)
    F["H_field_adaptive"] = sph_entropy(sph_weights(Q, P, h_mode="knn_adaptive", knn_k=8), N)
    F["u_RBF"] = R["gp"].rel_uncertainty(Q)
    zs = R["Z"].std()
    gp_m32 = GPR(kernel="matern32", seed=0, n_restarts=10).fit(P, R["Z"] / zs)
    F["u_M32"] = gp_m32.rel_uncertainty(Q)
    F["matern_fit"] = {"theta": [float(t) for t in gp_m32.theta], "lml": gp_m32.lml_}
    g = L2Field(R["G"], P, h_mode="global").metric(Q, delta=delta)
    F["g"] = g
    F["grad_norm"] = np.sqrt(g[:, 0, 0] + g[:, 1, 1])
    F.update(eigen_analysis(g))
    F["kink"] = metric_kink(Q, P, delta)
    F["w"] = w
    for k in ("d_nn", "d_k5", "d_k8", "H_cluster", "p_dom", "p_sec", "H_field", "N_eff", "max_w",
              "top3", "H_field_adaptive", "u_RBF", "u_M32", "grad_norm", "g", "lam", "e1", "theta1",
              "dirgrad", "pc"):
        if not np.isfinite(F[k]).all():
            raise ArtifactError(f"non-finite values in {k} at {int((~np.isfinite(F[k])).sum())} entries")
    return F


# ---------------------------------------------------------------- 語彙
class LensWords:
    """evidence_lib の語解決を、grid_scan の出力形式 (term, df, df が 1〜3 なら source_docs) で包む.

    n-gram の選び方・語の併合・同点の規則は make_evidence.py と共通 (evidence_lib.py)。同じ
    readout (global) と語数なら、make_evidence.py と同じ語・df・出典文献になる。
    """

    def __init__(self, vocab, stoplist, docs, meta):
        self.vocab, self.stop, self.meta = vocab, stoplist, meta
        self.resolver = WordResolver(vocab, docs)

    def pick_idx(self, sc, k):
        return pick_ngrams(sc, self.vocab, self.stop, k)

    def lens_words(self, idx_list):
        out = []
        for g in self.resolver.resolve(idx_list):
            rec = {"term": g["term"], "df": len(g["docs"])}
            if 0 < rec["df"] <= 3:
                rec["source_docs"] = [{"doc": int(d), "title": self.meta[d]["title"][:60]} for d in g["docs"]]
            out.append(rec)
        return out


def word_details(Q, keys, gix, F, R, topk_l2, topk_l1, progress=None):
    X, l2n, meta = R["X"], R["l2n"], R["meta"]
    lens = LensWords(R["vocab"], load_stoplist(), R["docs"], meta)
    Xl2 = (X / l2n[:, None]).astype(np.float64)
    md = Xl2.mean(0)
    md /= np.linalg.norm(md)
    Xf = X.astype(np.float64) + 1e-10
    Xl1 = Xf / Xf.sum(1, keepdims=True)
    ml1 = Xl1.mean(0)
    del Xf
    details = {}
    w = F["w"]
    for i in range(len(Q)):
        wi = w[i]
        v = wi @ Xl2
        nv = np.linalg.norm(v)
        if not (nv > 0 and np.isfinite(v).all()):
            raise ArtifactError(f"zero or non-finite L2 field vector at {keys[i]}")
        v /= nv
        pi = wi @ Xl1
        if not np.isfinite(pi).all():
            raise ArtifactError(f"non-finite L1 field vector at {keys[i]}")
        pi /= pi.sum()
        kl = pi * np.log(np.maximum(pi, 1e-300) / np.maximum(ml1, 1e-300))
        top = rank_documents(wi, TOPK_DOCS)         # 重みの降順、同じ重みは doc_id の昇順
        details[keys[i]] = {
            "grid_index": [int(gix[0][i]), int(gix[1][i])],
            "theme_lens_L2": lens.lens_words(lens.pick_idx(v - md, topk_l2)),
            "concentration_lens_L1": lens.lens_words(lens.pick_idx(kl, topk_l1)),
            "contributing_documents_top10": [{"doc": int(j), "weight": round(float(wi[j]), 4),
                                              "title": meta[j]["title"][:70]} for j in top]}
        if progress and (i + 1) % progress == 0:
            print(f"  words {i + 1}/{len(Q)}", flush=True)
    return details


# ---------------------------------------------------------------- 出力
def csv_columns(K):
    return (["x", "y", "nearest_doc_id", "nearest_title", "d_nn", "d_k5", "d_k8"]
            + [f"p_C{c}" for c in range(K)]
            + ["H_cluster", "dominant_C", "p_dom", "second_C", "p_second", "p_gap",
               "H_field", "N_eff", "H_field_adaptive", "max_doc_weight", "top3_weight_share",
               "u_RBF", "u_M32", "grad_norm", "lam1", "lam2", "aniso", "theta1_deg",
               "e1x", "e1y", "dirgrad1", "dirgrad2",
               "ix", "iy", "lam_gap_rel", "e1_ill_defined", "lam2_clipped", "metric_kink",
               "verbalization_50w"])


def write_csv(path, Q, dec, gix, F, R):
    K, meta = R["K"], R["meta"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(csv_columns(K))
        for i in range(len(Q)):
            n_i = int(F["near"][i])
            lam, e1 = F["lam"][i], F["e1"][i]
            wcsv.writerow([f"{Q[i, 0]:.{dec}f}", f"{Q[i, 1]:.{dec}f}", n_i, meta[n_i]["title"][:50],
                           f"{F['d_nn'][i]:.4f}", f"{F['d_k5'][i]:.4f}", f"{F['d_k8'][i]:.4f}"]
                          + [f"{F['pc'][i, c]:.4f}" for c in range(K)]
                          + [f"{F['H_cluster'][i]:.4f}", int(F["dom"][i]), f"{F['p_dom'][i]:.4f}",
                             int(F["sec"][i]), f"{F['p_sec'][i]:.4f}", f"{F['p_dom'][i] - F['p_sec'][i]:.4f}",
                             f"{F['H_field'][i]:.4f}", f"{F['N_eff'][i]:.1f}", f"{F['H_field_adaptive'][i]:.4f}",
                             f"{F['max_w'][i]:.4f}", f"{F['top3'][i]:.4f}",
                             f"{F['u_RBF'][i]:.4f}", f"{F['u_M32'][i]:.4f}",
                             f"{F['grad_norm'][i]:.4f}", f"{lam[0]:.5f}", f"{lam[1]:.5f}",
                             f"{F['aniso'][i]:.2f}", f"{F['theta1'][i]:.1f}",
                             f"{e1[0]:.4f}", f"{e1[1]:.4f}",
                             f"{F['dirgrad'][i, 0]:.4f}", f"{F['dirgrad'][i, 1]:.4f}",
                             int(gix[0][i]), int(gix[1][i]), f"{F['gap_rel'][i]:.6f}",
                             int(F["ill"][i]), int(F["clipped"][i]), int(F["kink"][i]), ""])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="パイプラインの出力ディレクトリ (KM_OUT)")
    ap.add_argument("--data", help="中間ファイルのディレクトリ (KM_DATA, 既定 <KM_OUT>/work)")
    ap.add_argument("--labels", help="クラスタラベル (既定 <KM_DATA>/cluster_labels.npy = 標準 k=5; "
                                     "recluster.py の cluster_labels_k{K}.npy も可)")
    ap.add_argument("--dx", type=float, default=0.1, help="格子間隔 (2/dx が整数になる値, 既定 0.1)")
    ap.add_argument("--delta", type=float, default=1e-3, help="計量の中心差分の幅 (既定 1e-3)")
    ap.add_argument("--topk-l2", type=int, default=10, help="L2 (テーマ) レンズの語数")
    ap.add_argument("--topk-l1", type=int, default=8, help="L1 (集中) レンズの語数")
    ap.add_argument("--no-words", action="store_true", help="語彙解決を省略 (数値のみ)")
    ap.add_argument("--custom-labels", action="store_true",
                    help="この run の k-means 分割ではないラベルを意図して使う (分割の照合を省き、meta に記録)")
    ap.add_argument("--outdir", help="出力先 (既定 <KM_OUT>/grid_scan/k{K}_dx{dx}_l2-{n}_l1-{n}[_delta{d}] "
                                     "または ..._nowords)")
    ap.add_argument("--force", action="store_true", help="既存の出力を上書きする")
    a = ap.parse_args(argv)
    t0 = time.time()
    if not (np.isfinite(a.delta) and 0 < a.delta <= 0.05):
        raise ValueError(f"--delta must be in (0, 0.05], got {a.delta}")
    if a.topk_l2 < 1 or a.topk_l1 < 1:
        raise ValueError("--topk-l2 and --topk-l1 must be at least 1")
    xs, dec, spacing = grid_axis(a.dx)
    OUT = os.path.abspath(a.out)
    DATA = os.path.abspath(a.data or os.path.join(OUT, "work"))
    labels_path = os.path.abspath(a.labels or os.path.join(DATA, "cluster_labels.npy"))
    words = not a.no_words
    R = load_run(OUT, DATA, labels_path, words, a.custom_labels)
    K = R["K"]
    dxs = f"{spacing:.{dec}f}"
    name = f"k{K}_dx{dxs}" + (f"_l2-{a.topk_l2}_l1-{a.topk_l1}" if words else "_nowords")
    if a.delta != 1e-3:
        name += f"_delta{a.delta:g}"
    OD = os.path.abspath(a.outdir or os.path.join(OUT, "grid_scan", name))
    refuse_repo_data(OD)
    pre = os.path.join(OD, f"k{K}_grid_scan")
    outputs = [pre + s for s in ("_raw.csv", "_values.npz", "_meta.json", "_runinfo.json")]
    if words:
        outputs.append(pre + "_details.json")
    refuse_overwrite(outputs, a.force)
    # --force でも、今回書かないファイル (前回の語彙・前回の走査から作った図と候補一覧) が残ると
    # 新しい meta と食い違うので、残っていれば止める
    stale = ([] if words else [pre + "_details.json"]) + [
        os.path.join(OD, f) for f in ("heatmap_Hcluster.png", "heatmap_white_support_uncertainty.png",
                                      f"k{K}_candidate_prelist.csv", f"k{K}_candidate_prelist_meta.json")]
    stale = [p for p in stale if os.path.exists(p)]
    if stale:
        raise ArtifactError("these files in the output directory were made from an earlier scan and would "
                            "no longer match it; remove them or choose another --outdir:\n  " + "\n  ".join(stale))

    Q, ix, iy = make_grid(xs)
    keys = [point_key(x, y, dec) for x, y in Q]
    if len(set(keys)) != len(keys):
        raise AssertionError("grid keys are not unique")   # grid_axis が保証する; 起きたらバグ
    F = numeric_fields(Q, R, a.delta)
    details = word_details(Q, keys, (ix, iy), F, R, a.topk_l2, a.topk_l1, progress=100) if words else None

    os.makedirs(OD, exist_ok=True)
    write_csv(pre + "_raw.csv", Q, dec, (ix, iy), F, R)
    if words:
        write_json(pre + "_details.json", details)
    np.savez(pre + "_values.npz", x=Q[:, 0], y=Q[:, 1], ix=ix, iy=iy, nearest_doc_id=F["near"],
             d_nn=F["d_nn"], d_k5=F["d_k5"], d_k8=F["d_k8"], p_cluster=F["pc"], H_cluster=F["H_cluster"],
             dominant_C=F["dom"], second_C=F["sec"], H_field=F["H_field"], N_eff=F["N_eff"],
             H_field_adaptive=F["H_field_adaptive"], max_doc_weight=F["max_w"], top3_weight_share=F["top3"],
             u_RBF=F["u_RBF"], u_M32=F["u_M32"], metric_g=F["g"], grad_norm=F["grad_norm"],
             lam_raw=F["lam_raw"], lam=F["lam"], e1=F["e1"], theta1_deg=F["theta1"], aniso=F["aniso"],
             dirgrad=F["dirgrad"], lam_gap_rel=F["gap_rel"], e1_ill_defined=F["ill"],
             lam2_clipped=F["clipped"], metric_kink=F["kink"])
    u = F["u_RBF"], F["u_M32"]
    code_files = [os.path.abspath(__file__), os.path.join(CODE, "kmlib.py"), os.path.join(CODE, "postproc_lib.py")]
    meta_out = {
        "tool": "grid_scan.py", "schema_version": 1,
        "grid": {"range": [-1, 1], "dx": spacing, "dx_requested": a.dx, "n_per_axis": len(xs),
                 "n_points": len(Q), "coordinate_decimals": dec,
                 "key_format": f"'{{x:.{dec}f}},{{y:.{dec}f}}'",
                 "row_order": "row r has ix = r // n_per_axis, iy = r % n_per_axis (x varies slowest)",
                 "axis": [float(v) for v in xs]},
        "K": K, "n_documents": R["N"], "cluster_sizes": np.bincount(R["lab"], minlength=K).tolist(),
        "settings": {"sph_h": "global (max_dist/1.98)", "sph_h_adaptive_column": "knn_adaptive, knn_k=8",
                     "delta": a.delta,
                     "gpr_main": "ARD-RBF+White (the run's gpr_model.pkl, not refitted)",
                     "gpr_sensitivity": "Matern32+White refitted on the same target (svd_scores[:, :10] / std), "
                                        "n_restarts=10, random_state=0",
                     "cluster_labels": rel_or_none(labels_path, OUT) or os.path.basename(labels_path),
                     "custom_labels": a.custom_labels,
                     "words": words,
                     "lens": (f"L2 top{a.topk_l2} / L1 top{a.topk_l1}; same word resolution as make_evidence.py; "
                              f"candidate pool {CANDIDATE_POOL} n-grams (or the vocabulary size if smaller)"
                              if words else None),
                     "tie_rules": ("n-grams by score descending then feature index ascending; contributing "
                                   "documents by weight descending then doc_id ascending; words by document "
                                   "count, then length (merged terms), then alphabetically" if words else None),
                     "readout_note": "all fields use the global SPH rule; make_evidence.py defaults to the "
                                     "LOO-selected readout tier, so compare with --readout global"},
        "tolerances": {"prob_sum_atol": PROB_SUM_ATOL, "eig_neg_rtol": EIG_NEG_RTOL,
                       "eig_gap_rtol": EIG_GAP_RTOL, "metric_kink_farthest_tie_rtol": 1e-12},
        "flags": {"lam2_clipped": "lam2 was negative within eig_neg_rtol * lam1 (rounding) and is reported as 0",
                  "e1_ill_defined": "(lam1 - lam2) / lam1 < eig_gap_rtol: the principal direction is not "
                                    "numerically determined; do not interpret theta1_deg / e1",
                  "metric_kink": "the global SPH h(P) = max distance / 1.98 is not differentiable inside the "
                                 "finite-difference stencil (the farthest document changes; with corner "
                                 "anchors this is the lines x = 0 and y = 0): the central difference tends to "
                                 "the mean of the one-sided derivatives and g is built from inner products of "
                                 "that mean, which in general differs from the mean of the one-sided metrics; "
                                 "g moves O(delta) with delta and its eigen-directions should not be interpreted"},
        "eigen_summary": {"lam2_clipped_points": int(F["clipped"].sum()),
                          "lam2_raw_min": float(F["lam_raw"][:, 1].min()),
                          "e1_ill_defined_points": int(F["ill"].sum()),
                          "metric_kink_points": int(F["kink"].sum())},
        "matern_fit": F["matern_fit"],
        "u_corr_RBF_M32": float(np.corrcoef(*u)[0, 1]),
        "inputs": {k: {"path": rel_or_none(p, OUT), "sha256": sha256_file(p)} for k, p in R["paths"].items()},
        "document_order": "map index i = row i of coords.npy = doc_id i of corpus_metadata.csv "
                          "= entry i of the labels = entry i of docs_clean.json",
        "consistency_checks": R["checks"],
        "provenance": provenance(CODE, code_files),
        "note": "candidate screening input only; not an optimum, a finding, or a Discovery Score",
    }
    write_json(pre + "_meta.json", meta_out)
    write_json(pre + "_runinfo.json", {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runtime_s": round(time.time() - t0, 1),
        "argv": sys.argv if argv is None else ["grid_scan.py"] + list(argv),
        "km_out": OUT, "km_data": DATA, "labels": labels_path, "outdir": OD})
    print(f"rows: {len(Q)} ({len(xs)}x{len(xs)}, dx={dxs}) | K={K} | u corr(RBF,M32): "
          f"{meta_out['u_corr_RBF_M32']:.4f} | e1 ill-defined: {int(F['ill'].sum())} | "
          f"metric kink: {int(F['kink'].sum())} | "
          f"runtime: {time.time() - t0:.1f} s")
    print("->", OD)


if __name__ == "__main__":
    run_cli(main)
