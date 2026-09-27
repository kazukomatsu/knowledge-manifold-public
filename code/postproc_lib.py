# -*- coding: utf-8 -*-
"""後処理ツール (add_cluster_to_csv / recluster / grid_scan / grid_viz) の共通部.

どのツールも完成した run の成果物を読むだけで、パイプライン本体の成果物
(coords.npy, cluster_labels.npy, coordinates_2d.csv など) は書き換えない。
入力の欠落・件数の不整合・別 run の成果物の混入は、推測で補わずにその場で失敗させる。
"""
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np


class ArtifactError(RuntimeError):
    """必要な成果物が無い・壊れている・互いに整合しない."""


def run_cli(main):
    """ArtifactError / ValueError を 1 行のエラーにして終了コード 2 で返す."""
    try:
        main()
    except (ArtifactError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_files(named_paths):
    """{名前: パス} のうち存在しないものを全部まとめて報告して失敗する."""
    missing = [f"{name}: {path}" for name, path in named_paths.items() if not os.path.isfile(path)]
    if missing:
        raise ArtifactError("missing required artifact(s) — produce them with the pipeline "
                            "(code/run_v50.sh); nothing is substituted from other runs:\n  "
                            + "\n  ".join(missing))


def refuse_overwrite(paths, force):
    """既存ファイルを黙って上書きしない. --force のときだけ上書きを許す."""
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not force:
        raise ArtifactError("refusing to overwrite existing output(s) (pass --force to replace them):\n  "
                            + "\n  ".join(existing))


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _inside(path, directory):
    """path が directory の中か (symlink と、大文字小文字を区別しないファイルシステムも考慮)."""
    if not os.path.isdir(directory):
        return False
    p = os.path.abspath(path)
    while not os.path.exists(p):              # まだ無い部分を落とし、存在する最も深い祖先へ
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    p = os.path.realpath(p)                   # symlink を実体へ解決してから祖先をたどる
    while True:
        if os.path.exists(p) and os.path.samefile(p, directory):
            return True
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent


def refuse_repo_data(path):
    """同梱データ <repo>/data の中には書かない (git 管理下で、.gitignore にも入っていない)."""
    data = os.path.join(REPO_ROOT, "data")
    if _inside(path, data):
        raise ArtifactError(f"refusing to write into the shipped data directory {data}; "
                            "give an output location outside it (--outdir / --output)")


def same_partition(a, b):
    """番号の付け方を除いて同じ分割か."""
    a, b = np.asarray(a).tolist(), np.asarray(b).tolist()
    pairs = set(zip(a, b))
    return len(pairs) == len(set(a)) == len(set(b))


COORD_CSV_ATOL = 1e-8    # coordinates_2d.csv は小数 8 桁 (丸め誤差 <= 5e-9)
SVD_GRAM_ATOL = 1e-10    # svd_scores は Gram の固有分解で Z Z^T = G (実測 ~6e-15)
XRAW_GRAM_ATOL = 1e-6    # Gram は X_raw から作る (BLAS の違いで相対 ~5e-9, 同一環境の実測 ~2e-13)
ZSCALE_RTOL = 1e-12
TOP_NGRAMS_CHECKED = 5


def check_run(out, P, Z=None, G=None, gpr=None, X=None, l2n=None, vocab=None, docs=None,
              meta=None, labels=None, custom_labels=False):
    """手元にある成果物の組合せで、同じ run のものかを検査する.

    つながり: coords.npy - coordinates_2d.csv - manifest.json (- corpus_metadata.csv)、
    coords.npy - gpr_model.pkl - svd_scores.npy - gram_l2.npy - X_raw.npy - docs_clean.json
    - corpus_metadata.csv (chars_cleaned = 本文の文字数)、
    ラベル - svd_scores.npy (標準ラベルも recluster.py のラベルも kmeans(svd[:, :10], K, seed=0) の分割)。
    食い違えば ArtifactError。戻り値は {検査: "ok" | "not available: …" | "skipped: …"}
    で、照合できなかった組合せもそのまま記録に残す。
    """
    res = {}
    N = len(P)
    csv_path = os.path.join(out, "coordinates_2d.csv")
    key = "coordinates_2d.csv matches coords.npy"
    if os.path.isfile(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ok = len(rows) == N and [r.get("doc_id") for r in rows] == [str(i) for i in range(N)]
        if ok:
            xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
            ok = bool(np.abs(xy - P).max() <= COORD_CSV_ATOL)
        if not ok:
            raise ArtifactError(f"{csv_path} does not describe coords.npy (artifacts of different runs?)")
        res[key] = "ok"
    else:
        res[key] = "not available: no coordinates_2d.csv"
    man = os.path.join(out, "manifest.json")
    key = "files recorded in manifest.json"
    if os.path.isfile(man):
        with open(man, encoding="utf-8") as f:
            hashes = json.load(f).get("artifact_hashes", {})
        checked = []
        for name in ("coordinates_2d.csv", "corpus_metadata.csv"):
            p = os.path.join(out, name)
            if name in hashes and os.path.isfile(p):
                if sha256_file(p) != hashes[name]:
                    raise ArtifactError(f"{p} is not the file recorded in {man} (sha256 differs): "
                                        "edited after the run, or taken from another run")
                checked.append(name)
        res[key] = ("ok: " + ", ".join(checked)) if checked else "not available: nothing to compare"
    else:
        res[key] = "not available: no manifest.json"
    key = "gpr_model.pkl fitted on coords.npy and svd_scores.npy"
    if gpr is not None:
        # 04_fields.py gpr は coords.npy そのものを学習点にし、svd_scores[:, :10] の std で割って学習する
        if not (isinstance(gpr.get("X"), np.ndarray) and np.array_equal(gpr["X"], P)):
            raise ArtifactError("gpr_model.pkl was not fitted on coords.npy (another run, or the map was "
                                "rebuilt without re-running 04_fields.py gpr)")
        if Z is not None and "zscale" in gpr:
            zs = float(Z[:, :10].std())
            if abs(gpr["zscale"] - zs) > ZSCALE_RTOL * abs(zs):
                raise ArtifactError(f"gpr_model.pkl zscale {gpr['zscale']} != std of svd_scores[:, :10] {zs} "
                                    "— the GPR belongs to another run")
        res[key] = "ok"
    else:
        res[key] = "not available: no gpr_model.pkl"
    key = "svd_scores.npy reproduces gram_l2.npy"
    if Z is not None and G is not None:
        if Z.shape[0] != G.shape[0] or np.abs(Z @ Z.T - G).max() > SVD_GRAM_ATOL:
            raise ArtifactError("svd_scores.npy is not the eigen-decomposition of gram_l2.npy (different runs?)")
        res[key] = "ok"
    else:
        res[key] = "not available"
    key = "X_raw.npy reproduces gram_l2.npy"
    if X is not None and G is not None:
        Xl2 = (X / l2n[:, None]).astype(np.float64)
        if np.abs(Xl2 @ Xl2.T - G).max() > XRAW_GRAM_ATOL:
            raise ArtifactError("gram_l2.npy was not computed from X_raw.npy (different runs?)")
        res[key] = "ok"
    else:
        res[key] = "not available"
    key = f"docs_clean.json contains each document's top {TOP_NGRAMS_CHECKED} n-grams of X_raw.npy"   # 語彙数まで
    if X is not None and docs is not None and vocab is not None:
        for i, d in enumerate(docs):
            text = " " + " ".join(d["text"].split()) + " "      # char_wb は語の前後を空白で埋めて切る
            m = min(TOP_NGRAMS_CHECKED, X.shape[1])
            top = np.argpartition(X[i], -m)[-m:]
            if any(X[i, j] > 0 and vocab[j] not in text for j in top):
                raise ArtifactError(f"docs_clean.json entry {i} does not contain the highest-weighted "
                                    f"n-grams of document {i} in X_raw.npy (text of another run?)")
        res[key] = "ok"
    else:
        res[key] = "not available"
    key = "corpus_metadata.csv chars_cleaned = length of each docs_clean.json text"
    if docs is not None and meta is not None and meta and "chars_cleaned" in meta[0]:
        bad = [i for i, (d, m) in enumerate(zip(docs, meta)) if m["chars_cleaned"] != str(len(d["text"]))]
        if bad:
            raise ArtifactError(f"corpus_metadata.csv does not describe docs_clean.json (chars_cleaned differs "
                                f"for {len(bad)} documents, first {bad[:5]}): metadata of another run?")
        res[key] = "ok"
    else:
        res[key] = "not available"
    key = "labels are this run's k-means partition"
    if labels is not None:
        if custom_labels:
            res[key] = "skipped: --custom-labels"
        elif Z is None:
            res[key] = "not available: no svd_scores.npy"
        else:
            from kmlib import kmeans
            raw, _ = kmeans(Z[:, :10], int(labels.max()) + 1, seed=0)
            if not same_partition(labels, raw):
                raise ArtifactError("the cluster labels are not the k-means partition of this run's "
                                    "svd_scores.npy (kmeans(svd[:, :10], K, seed=0)) — labels of another run? "
                                    "If they are deliberately different labels, pass --custom-labels")
            res[key] = "ok"
    return res


def load_labels(path, n):
    """クラスタラベル (長さ n の非負整数, 0..K-1 がすべて出現) を読み、(lab, K) を返す."""
    lab = np.load(path)
    if lab.ndim != 1 or len(lab) != n:
        raise ArtifactError(f"{path}: expected {n} labels (one per map document), got shape {lab.shape}")
    if not np.issubdtype(lab.dtype, np.integer):
        raise ArtifactError(f"{path}: labels must be integers, got dtype {lab.dtype}")
    if lab.min() < 0:
        raise ArtifactError(f"{path}: negative label {int(lab.min())}")
    K = int(lab.max()) + 1
    counts = np.bincount(lab, minlength=K)
    if (counts == 0).any():
        raise ArtifactError(f"{path}: labels are not contiguous 0..{K - 1}; "
                            f"empty label(s) {np.flatnonzero(counts == 0).tolist()}")
    return lab.astype(np.int64), K


def load_metadata(path, n):
    """corpus_metadata.csv を読む. 行 i が map index i (doc_id == i) であることを確かめる."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != n:
        raise ArtifactError(f"{path}: {len(rows)} rows, but the map has {n} documents")
    if rows and "doc_id" not in rows[0]:
        raise ArtifactError(f"{path}: no doc_id column")
    ids = [r["doc_id"] for r in rows]
    if ids != [str(i) for i in range(n)]:
        raise ArtifactError(f"{path}: doc_id must be 0..{n - 1} in row order "
                            "(row i describes map document i)")
    return rows


def load_docs_clean(path, meta):
    """corpus/docs_clean.json を読み、並びが corpus_metadata.csv と一致することを確かめる."""
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    if not isinstance(docs, list) or len(docs) != len(meta):
        raise ArtifactError(f"{path}: expected a list of {len(meta)} documents")
    for i, (d, m) in enumerate(zip(docs, meta)):
        if "text" not in d:
            raise ArtifactError(f"{path}: document {i} has no text")
        if "doc_id" in d and int(d["doc_id"]) != i:
            raise ArtifactError(f"{path}: entry {i} has doc_id {d['doc_id']}")
        if "filename" in d and "filename" in m and d["filename"] != m["filename"]:
            raise ArtifactError(f"{path}: entry {i} is {d['filename']} but corpus_metadata.csv "
                                f"row {i} is {m['filename']}")
    return docs


def load_vocab(path, n_features):
    """vocab.json を列番号順の n-gram リストで返す (dict 形式も受ける)."""
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    if isinstance(vocab, dict):
        inv = [None] * len(vocab)
        for t, i in vocab.items():
            inv[int(i)] = t
        if any(t is None for t in inv):
            raise ArtifactError(f"{path}: vocabulary indices are not 0..{len(vocab) - 1}")
        vocab = inv
    if len(vocab) != n_features:
        raise ArtifactError(f"{path}: {len(vocab)} terms, but X_raw.npy has {n_features} columns")
    return vocab


def provenance(code_dir, files):
    """解析結果に添える実行環境・コード版の記録 (実行時刻などの変動値は含めない)."""
    import scipy
    import sklearn
    rec = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    try:
        import matplotlib
        rec["matplotlib"] = matplotlib.__version__
    except ImportError:
        rec["matplotlib"] = None
    try:
        sha = subprocess.run(["git", "-C", code_dir, "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", code_dir, "status", "--porcelain", "--", "."],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        rec["code_git_sha"], rec["code_git_dirty"] = sha, dirty
    except (OSError, subprocess.CalledProcessError):
        rec["code_git_sha"], rec["code_git_dirty"] = None, None
    rec["code_sha256"] = {os.path.basename(p): sha256_file(p) for p in files}
    return rec


def rel_or_none(path, base):
    """base 配下なら base からの相対パス, そうでなければ None (絶対パスを記録に残さない)."""
    path, base = os.path.abspath(path), os.path.abspath(base)
    try:
        rel = os.path.relpath(path, base)
    except ValueError:
        return None
    return None if rel.startswith("..") else rel


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
