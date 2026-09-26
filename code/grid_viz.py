# -*- coding: utf-8 -*-
"""後処理: grid_scan.py の出力からヒートマップ 2 枚と候補点一覧 (pre-list) を作る.

原作は共著者 (川越) の scan/k5_grid_viz.py (scan_code.zip)。閾値 (パーセンタイル) と
同時条件は原作のまま。候補は暫定条件によるスクリーニングの結果であって、最適解・研究上の
発見・最終的な Discovery Score ではない。

原作は CSV の行数の平方根と行の並びから格子を推定していた。ここでは meta.json の格子定義と
各行の ix, iy から格子を組み立て、行の欠落・重複・座標の食い違いがあれば失敗する。重ねて
描く文献座標とクラスタラベルは、走査に使ったものと sha256 が一致することを確かめる。

usage:
  python3 code/grid_viz.py --scan <grid_scan.py の出力ディレクトリ> --out <KM_OUT>
      [--labels <cluster_labels.npy>] [--force]
出力 (--scan のディレクトリ内):
  heatmap_Hcluster.png, heatmap_white_support_uncertainty.png,
  k{K}_candidate_prelist.csv, k{K}_candidate_prelist_meta.json
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
from postproc_lib import ArtifactError, refuse_overwrite, require_files, run_cli, sha256_file, write_json

FIELDS = ("H_cluster", "d_nn", "N_eff", "u_RBF", "grad_norm")
PRELIST_COLS = ["tier", "region", "x", "y", "H_cluster", "d_nn", "N_eff", "u_RBF",
                "grad_norm", "dominant_C", "second_C", "p_dom", "p_second", "ix", "iy"]


def find_meta(scan_dir):
    metas = sorted(glob.glob(os.path.join(scan_dir, "k*_grid_scan_meta.json")))
    if len(metas) != 1:
        raise ArtifactError(f"{scan_dir}: expected exactly one k*_grid_scan_meta.json, found {len(metas)}")
    with open(metas[0], encoding="utf-8") as f:
        meta = json.load(f)
    return metas[0], meta


def load_grid(csv_path, meta):
    """CSV を格子 (ix, iy) に並べ直す. 戻り値 (rows_by_cell[n][n], {列: (n, n) 配列})."""
    g = meta["grid"]
    n, dec, axis = g["n_per_axis"], g["coordinate_decimals"], g["axis"]
    if len(axis) != n or g["n_points"] != n * n:
        raise ArtifactError(f"{csv_path}: inconsistent grid definition in meta.json")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != n * n:
        raise ArtifactError(f"{csv_path}: {len(rows)} rows, the grid in meta.json has {n}x{n} = {n * n}")
    missing_cols = [c for c in FIELDS + ("ix", "iy", "x", "y") if rows and c not in rows[0]]
    if missing_cols:
        raise ArtifactError(f"{csv_path}: missing column(s) {missing_cols}")
    cell = [[None] * n for _ in range(n)]
    V = {k: np.full((n, n), np.nan) for k in FIELDS}
    for r in rows:
        try:
            i, j = int(r["ix"]), int(r["iy"])
        except ValueError:
            raise ArtifactError(f"{csv_path}: non-integer grid index {r['ix']!r}, {r['iy']!r}") from None
        if not (0 <= i < n and 0 <= j < n):
            raise ArtifactError(f"{csv_path}: grid index ({i}, {j}) outside 0..{n - 1}")
        if cell[i][j] is not None:
            raise ArtifactError(f"{csv_path}: grid point ({i}, {j}) appears twice")
        if r["x"] != f"{axis[i]:.{dec}f}" or r["y"] != f"{axis[j]:.{dec}f}":
            raise ArtifactError(f"{csv_path}: row ({r['x']}, {r['y']}) does not match grid index ({i}, {j})")
        cell[i][j] = r
        for k in FIELDS:
            V[k][i, j] = float(r[k])
    bad = [k for k in FIELDS if not np.isfinite(V[k]).all()]
    if bad:
        raise ArtifactError(f"{csv_path}: non-finite values in {bad}")
    return cell, V


def regions(mask):
    """4 近傍連結成分. 番号は (ix, iy) の走査順 (原作と同じ)."""
    n = mask.shape[0]
    rid = np.full(mask.shape, -1, int)
    k = 0
    for i in range(n):
        for j in range(n):
            if mask[i, j] and rid[i, j] < 0:
                dq = collections.deque([(i, j)])
                rid[i, j] = k
                while dq:
                    x, y = dq.popleft()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        p, q = x + dx, y + dy
                        if 0 <= p < n and 0 <= q < n and mask[p, q] and rid[p, q] < 0:
                            rid[p, q] = k
                            dq.append((p, q))
                k += 1
    return rid, k


def thresholds(V):
    Hc, dnn, Neff, u, gn = (V[k] for k in FIELDS)
    thrB = {"H": np.percentile(Hc, 70), "d": np.percentile(dnn, 30), "N": np.percentile(Neff, 30),
            "u": np.percentile(u, 70), "g": np.percentile(gn, 50)}
    thrA = {"H": np.percentile(Hc, 85), "d": np.percentile(dnn, 40), "N": np.percentile(Neff, 30),
            "u": np.percentile(u, 60), "g": np.percentile(gn, 70)}
    mk = lambda t: (Hc >= t["H"]) & (dnn >= t["d"]) & (Neff >= t["N"]) & (u <= t["u"]) & (gn >= t["g"])
    return thrB, thrA, mk(thrB), mk(thrA)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True, help="grid_scan.py の出力ディレクトリ")
    ap.add_argument("--out", required=True, help="走査した run の KM_OUT (文献座標の重ね描き用)")
    ap.add_argument("--labels", help="走査に使ったクラスタラベル (既定: meta.json に記録された KM_OUT 内の相対パス)")
    ap.add_argument("--force", action="store_true", help="既存の図と候補一覧を上書きする")
    a = ap.parse_args(argv)
    SD, OUT = os.path.abspath(a.scan), os.path.abspath(a.out)
    meta_path, meta = find_meta(SD)
    K = meta["K"]
    pre = meta_path[:-len("_meta.json")]
    csv_path = pre + "_raw.csv"
    coords_path = os.path.join(OUT, "coords.npy")
    lab_rel = meta["inputs"]["cluster_labels"]["path"]
    if a.labels:
        labels_path = os.path.abspath(a.labels)
    elif lab_rel:
        labels_path = os.path.join(OUT, lab_rel)
    else:
        raise ArtifactError("the scan's labels were outside KM_OUT; pass them with --labels")
    require_files({"scan CSV": csv_path, "coords.npy": coords_path, "cluster labels": labels_path})
    for name, path in (("coords.npy", coords_path), ("cluster_labels", labels_path)):
        if sha256_file(path) != meta["inputs"][name]["sha256"]:
            raise ArtifactError(f"{path} is not the {name} this scan was computed from (sha256 differs)")
    outs = [os.path.join(SD, "heatmap_Hcluster.png"), os.path.join(SD, "heatmap_white_support_uncertainty.png"),
            os.path.join(SD, f"k{K}_candidate_prelist.csv"), os.path.join(SD, f"k{K}_candidate_prelist_meta.json")]
    refuse_overwrite(outs, a.force)

    cell, V = load_grid(csv_path, meta)
    n = meta["grid"]["n_per_axis"]
    half = meta["grid"]["dx"] / 2
    P = np.load(coords_path)
    lab = np.load(labels_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def hm(ax, v, title, cmap):
        # 画素の中心を格子点に合わせる (原作は extent=[-1,1,-1,1] で半セルずれていた)
        im = ax.imshow(v.T, origin="lower", extent=[-1 - half, 1 + half, -1 - half, 1 + half],
                       cmap=cmap, aspect="equal")
        ax.scatter(P[:, 0], P[:, 1], c=lab, cmap="tab10", s=6, edgecolor="k", linewidth=0.2)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    hm(ax, V["H_cluster"], f"H_cluster (k={K} mixing entropy)", "viridis")
    fig.tight_layout(); fig.savefig(outs[0], dpi=140); plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 9.2))
    for ax, k, cm in zip(axs.ravel(), ["d_nn", "N_eff", "u_RBF", "grad_norm"],
                         ["magma", "viridis", "inferno", "cividis"]):
        hm(ax, V[k], k, cm)
    fig.tight_layout(); fig.savefig(outs[1], dpi=140); plt.close(fig)

    thrB, thrA, mB, mA = thresholds(V)
    ridA, kA = regions(mA)
    ridB, kB = regions(mB)
    with open(outs[2], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(PRELIST_COLS)
        for i in range(n):
            for j in range(n):
                if not mB[i, j]:
                    continue
                r = cell[i][j]
                w.writerow(["A" if mA[i, j] else "B", f"A{ridA[i, j]}" if mA[i, j] else f"B{ridB[i, j]}",
                            r["x"], r["y"], r["H_cluster"], r["d_nn"], r["N_eff"], r["u_RBF"],
                            r["grad_norm"], r["dominant_C"], r["second_C"], r["p_dom"], r["p_second"], i, j])
    write_json(outs[3], {
        "tierB": {k: float(v) for k, v in thrB.items()},
        "tierA": {k: float(v) for k, v in thrA.items()},
        "counts": {"tierB_points": int(mB.sum()), "tierB_regions": kB,
                   "tierA_points": int(mA.sum()), "tierA_regions": kA},
        "scan_meta_sha256": sha256_file(meta_path),
        "note": "B = loose (p70/p30 family), A = strict (p85/p60/p70 family); percentiles over this scan's "
                "grid points; region = 4-neighbour connected component. Provisional screening conditions: "
                "not an optimum, not a finding, not a final Discovery Score."})
    print(f"tierB {int(mB.sum())}pts/{kB}regions, tierA {int(mA.sum())}pts/{kA}regions -> {SD}")


if __name__ == "__main__":
    run_cli(main)
