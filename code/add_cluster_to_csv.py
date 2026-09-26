# -*- coding: utf-8 -*-
"""後処理: 座標 CSV にクラスタ列を付けた新しい CSV を書く (入力は変更しない).

原作は共著者 (川越) の code/add_cluster_to_csv.py (repository_release_20260915.zip)。
原作はカレントディレクトリの coordinates_2d.csv と work/cluster_labels.npy を行順で
突き合わせていた。ここでは行順ではなく doc_id で突き合わせ、doc_id が 0..N-1 を
ちょうど 1 回ずつ含むこと (欠損・重複なし) とラベル数 N の一致を確かめてから書く。
ラベルの添字 i は map index i (= coords.npy の行 i = coordinates_2d.csv の doc_id i)。
--run を渡すと、CSV の x, y が <KM_OUT>/coords.npy と一致すること、ラベルがその run の
k-means 分割 (kmeans(svd[:, :10], K, seed=0)。標準ラベルも recluster.py のラベルも該当)
であることも照合する。意図して別のラベルを付けるときは --custom-labels。

usage:
  python3 code/add_cluster_to_csv.py --run <KM_OUT>
      [--coords <csv>] [--labels <npy>] [--output <csv>] [--column cluster] [--custom-labels] [--force]
既定:
  --coords  <KM_OUT>/coordinates_2d.csv
  --labels  <KM_OUT>/work/cluster_labels.npy  (標準 k=5。recluster.py の
            cluster_labels_k{K}.npy も渡せる)
  --output  <KM_OUT>/coordinates_2d_clusters.csv
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postproc_lib import (COORD_CSV_ATOL, ArtifactError, load_labels, refuse_overwrite, refuse_repo_data,
                          require_files, run_cli, same_partition)


def _same_file(a, b):
    a, b = os.path.abspath(a), os.path.abspath(b)
    return a == b or (os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b))


def check_against_run(run, ids, body, header, lab, custom_labels):
    """CSV の座標とラベルが run のものか. 戻り値 {検査: 結果}. 食い違えば ArtifactError."""
    res = {}
    coords_npy = os.path.join(run, "coords.npy")
    if os.path.isfile(coords_npy) and "x" in header and "y" in header:
        P = np.load(coords_npy)
        jx, jy = header.index("x"), header.index("y")
        if len(P) != len(ids) or max(max(abs(float(r[jx]) - P[i, 0]), abs(float(r[jy]) - P[i, 1]))
                                     for r, i in zip(body, ids)) > COORD_CSV_ATOL:
            raise ArtifactError(f"the CSV coordinates are not those of {coords_npy} (another run?)")
        res["CSV x, y match coords.npy"] = "ok"
    else:
        res["CSV x, y match coords.npy"] = "not available"
    svd = os.path.join(run, "work", "svd_scores.npy")
    if custom_labels:
        res["labels are this run's k-means partition"] = "skipped: --custom-labels"
    elif os.path.isfile(svd):
        from kmlib import kmeans
        raw, _ = kmeans(np.load(svd)[:, :10], int(lab.max()) + 1, seed=0)
        if not same_partition(lab, raw):
            raise ArtifactError(f"the labels are not the k-means partition of {svd} (labels of another "
                                "run?); pass --custom-labels if they are deliberately different")
        res["labels are this run's k-means partition"] = "ok"
    else:
        res["labels are this run's k-means partition"] = "not available"
    return res


def add_cluster_column(coords_csv, labels_npy, output_csv, column="cluster", force=False,
                       run=None, custom_labels=False):
    """coords_csv の各行に labels[doc_id] を column 列として足し、output_csv に書く.

    戻り値 (行数, run との照合結果)。run が None なら照合しない。
    """
    if _same_file(output_csv, coords_csv) or _same_file(output_csv, labels_npy):
        raise ArtifactError("the output must be a new file, not one of the inputs")
    refuse_repo_data(output_csv)
    require_files({"coordinates CSV": coords_csv, "cluster labels": labels_npy})
    with open(coords_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ArtifactError(f"{coords_csv}: empty file")
    header, body = rows[0], rows[1:]
    if "doc_id" not in header:
        raise ArtifactError(f"{coords_csv}: no doc_id column (header: {header})")
    if column in header:
        raise ArtifactError(f"{coords_csv}: already has a '{column}' column; "
                            "choose another --column or start from the pipeline's CSV")
    if any(len(r) != len(header) for r in body):
        raise ArtifactError(f"{coords_csv}: rows do not all have {len(header)} fields")
    if not body:
        raise ArtifactError(f"{coords_csv}: no data rows")
    j = header.index("doc_id")
    try:
        ids = [int(r[j]) for r in body]
    except ValueError as e:
        raise ArtifactError(f"{coords_csv}: non-integer doc_id ({e})") from None
    n_lab = len(np.load(labels_npy))
    if n_lab != len(body):
        raise ArtifactError(f"{labels_npy} has {n_lab} labels but {coords_csv} has {len(body)} rows")
    lab, _ = load_labels(labels_npy, n_lab)
    n = len(lab)
    counts = Counter(ids)
    dup = sorted(i for i, c in counts.items() if c > 1)
    missing = sorted(set(range(n)) - set(counts))
    extra = sorted(i for i in counts if not 0 <= i < n)
    if dup or missing or extra:
        raise ArtifactError(f"{coords_csv}: doc_id must be 0..{n - 1} exactly once each "
                            f"(duplicated {dup[:10]}, missing {missing[:10]}, out of range {extra[:10]})")
    checks = check_against_run(run, ids, body, header, lab, custom_labels) if run else None
    refuse_overwrite([output_csv], force)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header + [column])
        for r, i in zip(body, ids):
            w.writerow(r + [str(int(lab[i]))])
    return len(body), checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="パイプラインの出力ディレクトリ (KM_OUT)")
    ap.add_argument("--coords", help="座標 CSV (既定 <KM_OUT>/coordinates_2d.csv)")
    ap.add_argument("--labels", help="クラスタラベル .npy (既定 <KM_OUT>/work/cluster_labels.npy)")
    ap.add_argument("--output", help="書き出す CSV (既定 <KM_OUT>/coordinates_2d_clusters.csv)")
    ap.add_argument("--column", default="cluster", help="追加する列名 (既定 cluster)")
    ap.add_argument("--custom-labels", action="store_true",
                    help="この run の k-means 分割ではないラベルを意図して使う (分割の照合を省く)")
    ap.add_argument("--force", action="store_true", help="既存の出力 CSV を上書きする")
    a = ap.parse_args(argv)
    if not a.run and not (a.coords and a.labels and a.output):
        ap.error("give --run, or all of --coords, --labels and --output")
    coords = a.coords or os.path.join(a.run, "coordinates_2d.csv")
    labels = a.labels or os.path.join(a.run, "work", "cluster_labels.npy")
    output = a.output or os.path.join(a.run, "coordinates_2d_clusters.csv")
    n, checks = add_cluster_column(coords, labels, output, a.column, a.force, a.run, a.custom_labels)
    print(f"{n} rows, column '{a.column}' from {labels} -> {output}")
    if checks is None:
        print("note: not checked against a run (no --run)")
    else:
        for k, v in checks.items():
            print(f"check: {k}: {v}")


if __name__ == "__main__":
    run_cli(main)
