# -*- coding: utf-8 -*-
"""Tests for the post-processing tools: add_cluster_to_csv, recluster, grid_scan, grid_viz.

Synthetic runs only (tests/synthetic_run.py) plus the shipped data/derived for the one
check that the re-clustering reproduces the published k=5 partition. No corpus text.
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
DERIVED = os.path.join(ROOT, "data", "derived")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import add_cluster_to_csv  # noqa: E402
import grid_scan  # noqa: E402
import grid_viz  # noqa: E402
import recluster  # noqa: E402
from postproc_lib import ArtifactError  # noqa: E402
from synthetic_run import build_run  # noqa: E402


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def tree_hashes(root):
    return {os.path.relpath(os.path.join(d, f), root): sha(os.path.join(d, f))
            for d, _, fs in os.walk(root) for f in fs}


def run_tool(script, *args, env=None):
    e = dict(os.environ, **(env or {}))
    return subprocess.run([sys.executable, os.path.join(CODE, script), *args],
                          capture_output=True, text=True, env=e)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return build_run(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture
def fresh_run(tmp_path):
    """A private copy for tests that corrupt or add files."""
    return build_run(tmp_path)


@pytest.fixture(scope="module")
def scan01(run_dir, tmp_path_factory):
    """One words-mode scan at dx=0.1 with the evidence-package vocabulary size (15/15)."""
    od = str(tmp_path_factory.mktemp("scan01"))
    grid_scan.main(["--out", run_dir, "--outdir", od, "--topk-l2", "15", "--topk-l1", "15"])
    return od


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ============================================================ add_cluster_to_csv
class TestAddCluster:
    def test_joins_by_doc_id_and_leaves_inputs_untouched(self, fresh_run, tmp_path):
        before = tree_hashes(fresh_run)
        out = str(tmp_path / "with_clusters.csv")
        n, checks = add_cluster_to_csv.add_cluster_column(os.path.join(fresh_run, "coordinates_2d.csv"),
                                                          os.path.join(fresh_run, "work", "cluster_labels.npy"),
                                                          out, run=fresh_run)
        assert set(checks.values()) == {"ok"}
        lab = np.load(os.path.join(fresh_run, "work", "cluster_labels.npy"))
        rows = read_csv(out)
        assert n == len(rows) == len(lab)
        src = read_csv(os.path.join(fresh_run, "coordinates_2d.csv"))
        for r, s in zip(rows, src):
            assert {k: r[k] for k in s} == s                 # original fields verbatim
            assert int(r["cluster"]) == lab[int(r["doc_id"])]
        assert tree_hashes(fresh_run) == before

    def test_row_order_does_not_decide_the_label(self, fresh_run, tmp_path):
        src = os.path.join(fresh_run, "coordinates_2d.csv")
        lines = open(src).read().splitlines()
        shuffled = tmp_path / "shuffled.csv"
        shuffled.write_text("\n".join([lines[0]] + lines[1:][::-1]) + "\n")
        out = str(tmp_path / "out.csv")
        add_cluster_to_csv.add_cluster_column(str(shuffled), os.path.join(fresh_run, "work", "cluster_labels.npy"), out)
        lab = np.load(os.path.join(fresh_run, "work", "cluster_labels.npy"))
        rows = read_csv(out)
        assert [r["doc_id"] for r in rows] == [l.split(",")[0] for l in lines[1:][::-1]]
        assert all(int(r["cluster"]) == lab[int(r["doc_id"])] for r in rows)

    @pytest.mark.parametrize("edit", ["duplicate", "missing", "out_of_range", "non_integer"])
    def test_rejects_broken_doc_ids(self, fresh_run, tmp_path, edit):
        lines = open(os.path.join(fresh_run, "coordinates_2d.csv")).read().splitlines()
        body = lines[1:]
        if edit == "duplicate":
            body[3] = body[2].split(",")[0] + "," + ",".join(body[3].split(",")[1:])
        elif edit == "missing":
            body = body[:-1] + ["999" + body[-1][body[-1].index(","):]]
        elif edit == "out_of_range":
            body[0] = "-1" + body[0][body[0].index(","):]
        else:
            body[0] = "x" + body[0][body[0].index(","):]
        bad = tmp_path / "bad.csv"
        bad.write_text("\n".join([lines[0]] + body) + "\n")
        with pytest.raises(ArtifactError):
            add_cluster_to_csv.add_cluster_column(str(bad), os.path.join(fresh_run, "work", "cluster_labels.npy"),
                                                  str(tmp_path / "out.csv"))
        assert not (tmp_path / "out.csv").exists()

    def test_rejects_label_count_mismatch(self, fresh_run, tmp_path):
        short = tmp_path / "short.npy"
        np.save(short, np.load(os.path.join(fresh_run, "work", "cluster_labels.npy"))[:-1])
        with pytest.raises(ArtifactError, match="labels"):
            add_cluster_to_csv.add_cluster_column(os.path.join(fresh_run, "coordinates_2d.csv"), str(short),
                                                  str(tmp_path / "out.csv"))

    def test_rejects_existing_cluster_column_and_in_place_writes(self, fresh_run, tmp_path):
        lab = os.path.join(fresh_run, "work", "cluster_labels.npy")
        once = str(tmp_path / "once.csv")
        add_cluster_to_csv.add_cluster_column(os.path.join(fresh_run, "coordinates_2d.csv"), lab, once)
        with pytest.raises(ArtifactError, match="already has"):
            add_cluster_to_csv.add_cluster_column(once, lab, str(tmp_path / "twice.csv"))
        with pytest.raises(ArtifactError, match="new file"):
            add_cluster_to_csv.add_cluster_column(once, lab, once)

    def test_cli_defaults_and_overwrite_protection(self, fresh_run):
        p = run_tool("add_cluster_to_csv.py", "--run", fresh_run)
        assert p.returncode == 0, p.stderr
        assert os.path.exists(os.path.join(fresh_run, "coordinates_2d_clusters.csv"))
        p = run_tool("add_cluster_to_csv.py", "--run", fresh_run)
        assert p.returncode == 2 and "refusing to overwrite" in p.stderr
        assert run_tool("add_cluster_to_csv.py", "--run", fresh_run, "--force").returncode == 0


# ============================================================ recluster
class TestRecluster:
    def test_k5_on_the_shipped_map_is_the_published_partition(self, tmp_path):
        """The published labels are kmeans(svd[:, :10], 5, seed=0); re-clustering renumbers only."""
        if not os.path.exists(os.path.join(DERIVED, "svd_scores.npy")):
            pytest.skip("derived artifacts not present")
        before = tree_hashes(DERIVED)
        od = str(tmp_path / "rc")
        recluster.main(["--out", DERIVED, "--data", DERIVED, "--k", "4", "5", "6", "7",
                        "--no-terms", "--outdir", od])
        meta = json.load(open(os.path.join(od, "clusters_k5_meta.json")))
        assert meta["standard_partition"]["same_partition"] is True
        std = np.load(os.path.join(DERIVED, "cluster_labels.npy"))
        k5 = np.load(os.path.join(od, "cluster_labels_k5.npy"))
        mapping = {int(a): b for a, b in meta["standard_partition"]["standard_to_recluster"].items()}
        assert [mapping[int(s)] for s in std] == k5.tolist()
        for K in (4, 5, 6, 7):
            lab = np.load(os.path.join(od, f"cluster_labels_k{K}.npy"))
            sizes = np.bincount(lab, minlength=K)
            assert sorted(set(lab.tolist())) == list(range(K))
            assert (np.diff(sizes) <= 0).all()                     # numbered by descending size
            summ = json.load(open(os.path.join(od, f"clusters_k{K}.json")))
            assert [c["size"] for c in summ] == sizes.tolist()
            assert all(c["top_ngrams"] is None and c["representative_doc"] is None for c in summ)
        assert tree_hashes(DERIVED) == before                      # shipped artifacts untouched

    def test_equal_sizes_keep_the_raw_kmeans_order(self, monkeypatch):
        raw = np.array([2, 2, 0, 0, 1, 1, 1])                        # sizes 2, 3, 2
        monkeypatch.setattr(recluster, "kmeans", lambda Z, K, seed=0: (raw, None))
        lab, back = recluster.size_ranked_kmeans(np.zeros((7, 10)), 3)
        assert lab.tolist() == [2, 2, 1, 1, 0, 0, 0]                # raw 1 -> 0; raw 0 -> 1 before raw 2 -> 2
        assert back.tolist() == raw.tolist()

    def test_empty_cluster_is_an_error(self, monkeypatch):
        monkeypatch.setattr(recluster, "kmeans", lambda Z, K, seed=0: (np.array([0, 0, 2, 2]), None))
        with pytest.raises(ArtifactError, match="empty"):
            recluster.size_ranked_kmeans(np.zeros((4, 10)), 3)

    @pytest.mark.parametrize("ks", [[1], [17], [3, 3], [0]])
    def test_rejects_impossible_k(self, ks):
        Z = np.random.default_rng(0).normal(size=(16, 10))
        with pytest.raises(ValueError):
            recluster.validate_ks(ks, Z)

    def test_top_terms_on_a_vocabulary_smaller_than_the_pool(self):
        vocab = ["alpha", "alphabet", "betamax", "gamma", "delta", "epsilon"]
        vec = np.array([0.1, 0.9, 0.8, 0.3, 0.2, 0.5])
        assert recluster.top_terms(vec, vocab, [], n=3) == ["alphabet", "betamax", "epsilon"]

    def test_full_run_writes_per_k_outputs_and_touches_nothing_else(self, fresh_run):
        before = tree_hashes(fresh_run)
        recluster.main(["--out", fresh_run, "--k", "2", "3", "5"])
        od = os.path.join(fresh_run, "recluster")
        after = tree_hashes(fresh_run)
        assert {k: v for k, v in after.items() if not k.startswith("recluster" + os.sep)} == before
        N = len(np.load(os.path.join(fresh_run, "coords.npy")))
        for K in (2, 3, 5):
            summ = json.load(open(os.path.join(od, f"clusters_k{K}.json")))
            assert sum(c["size"] for c in summ) == N
            assert all(len(c["top_ngrams"]) > 0 and c["representative_doc"]["doc"] in range(N) for c in summ)
            rows = read_csv(os.path.join(od, f"coordinates_2d_k{K}.csv"))
            lab = np.load(os.path.join(od, f"cluster_labels_k{K}.npy"))
            assert [int(r["cluster"]) for r in rows] == lab.tolist()
            dat = [l.split() for l in open(os.path.join(od, f"fig_map_k{K}.dat")) if not l.startswith("#")]
            assert [int(d[3]) for d in dat] == lab.tolist()
            assert os.path.getsize(os.path.join(od, f"fig_map_k{K}.png")) > 0
        assert json.load(open(os.path.join(od, "clusters_k5_meta.json")))["standard_partition"]["same_partition"]
        with pytest.raises(ArtifactError, match="refusing to overwrite"):
            recluster.main(["--out", fresh_run, "--k", "3"])
        recluster.main(["--out", fresh_run, "--k", "3", "--force"])

    def test_missing_inputs_are_reported_together(self, fresh_run):
        os.remove(os.path.join(fresh_run, "work", "X_raw.npy"))
        os.remove(os.path.join(fresh_run, "work", "vocab.json"))
        with pytest.raises(ArtifactError) as e:
            recluster.main(["--out", fresh_run, "--k", "3"])
        assert "X_raw.npy" in str(e.value) and "vocab.json" in str(e.value)
        recluster.main(["--out", fresh_run, "--k", "3", "--no-terms"])   # numbers-only path still works


# ============================================================ grid definition
class TestGridAxis:
    def test_default_grid_is_the_distributed_one(self):
        xs, dec, spacing = grid_scan.grid_axis(0.1)
        assert len(xs) == 21 and dec == 1 and spacing == pytest.approx(0.1, abs=1e-15)
        # bit-identical to scan/k5_grid_scan.py: np.round(np.linspace(-1, 1, n), 10)
        assert np.array_equal(xs, np.round(np.linspace(-1, 1, 21), 10))
        Q, ix, iy = grid_scan.make_grid(xs)
        GX, GY = np.meshgrid(xs, xs, indexing="ij")
        assert np.array_equal(Q, np.column_stack([GX.ravel(), GY.ravel()]))
        assert np.array_equal(Q[:, 0], xs[ix]) and np.array_equal(Q[:, 1], xs[iy])

    @pytest.mark.parametrize("dx, n, dec", [(0.1, 21, 1), (0.05, 41, 2), (0.04, 51, 2), (0.025, 81, 3),
                                            (0.02, 101, 2), (0.5, 5, 1), (1.0, 3, 1), (2.0, 2, 1), (0.01, 201, 2)])
    def test_spacing_and_keys_are_exact_and_unique(self, dx, n, dec):
        xs, d, spacing = grid_scan.grid_axis(dx)
        assert (len(xs), d) == (n, dec)
        assert spacing == pytest.approx(dx, rel=1e-12)
        assert np.allclose(np.diff(xs), dx, atol=1e-12)
        Q, _, _ = grid_scan.make_grid(xs)
        keys = [grid_scan.point_key(x, y, d) for x, y in Q]
        assert len(set(keys)) == n * n
        assert not any(c.startswith("-") and float(c) == 0 for k in keys for c in k.split(","))  # no "-0.0"
        assert all(float(k.split(",")[0]) == x and float(k.split(",")[1]) == y for k, (x, y) in zip(keys, Q))

    @pytest.mark.parametrize("dx", [0.3, 0.07, 0.15, 0.0, -0.1, float("nan"), 2 / 3, 0.005, 3.0])
    def test_rejects_a_dx_the_grid_cannot_honour(self, dx):
        with pytest.raises(ValueError):
            grid_scan.grid_axis(dx)


# ============================================================ eigen analysis
class TestEigenAnalysis:
    def test_relations_and_sign_convention(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(200, 2, 3))
        g = A @ A.transpose(0, 2, 1)                                   # random PSD tensors
        E = grid_scan.eigen_analysis(g)
        lam, e1 = E["lam"], E["e1"]
        assert (lam[:, 0] >= lam[:, 1]).all() and (lam[:, 1] >= 0).all()
        np.testing.assert_allclose(lam.sum(1), g[:, 0, 0] + g[:, 1, 1], rtol=1e-12)
        np.testing.assert_allclose(lam.prod(1), np.linalg.det(g), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(np.einsum("ma,mab,mb->m", e1, g, e1), lam[:, 0], rtol=1e-12)
        np.testing.assert_allclose(E["dirgrad"] ** 2, lam, rtol=1e-12)
        assert (e1[:, 0] >= 0).all() and (np.abs(E["theta1"]) <= 90).all()

    def test_rounding_negative_is_clipped_but_a_real_negative_fails(self):
        g = np.array([[[1.0, 1.0], [1.0, 1.0 - 1e-15]]])                 # lam2 ~ -5e-16: rounding
        E = grid_scan.eigen_analysis(g)
        assert E["clipped"][0] and E["lam"][0, 1] == 0.0 and E["dirgrad"][0, 1] == 0.0
        assert np.isinf(E["aniso"][0])
        with pytest.raises(ArtifactError, match="negative"):
            grid_scan.eigen_analysis(np.array([[[1.0, 0.0], [0.0, -1e-6]]]))

    def test_isotropic_point_has_no_principal_direction(self):
        E = grid_scan.eigen_analysis(np.array([[[2.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [0.0, 1.0]],
                                               [[1.0, 0.0], [0.0, 1.0 - 1e-4]]]))
        assert E["ill"].tolist() == [True, False, True]


# ============================================================ grid_scan on a synthetic run
class TestGridScan:
    def test_441_points_one_to_one_between_csv_details_and_npz(self, scan01):
        rows = read_csv(os.path.join(scan01, "k5_grid_scan_raw.csv"))
        details = json.load(open(os.path.join(scan01, "k5_grid_scan_details.json")))
        V = np.load(os.path.join(scan01, "k5_grid_scan_values.npz"))
        assert len(rows) == len(details) == len(V["x"]) == 441
        keys = [f"{r['x']},{r['y']}" for r in rows]
        assert len(set(keys)) == 441 and keys == list(details)
        assert len({(r["ix"], r["iy"]) for r in rows}) == 441
        for r, (k, d) in zip(rows, details.items()):
            assert d["grid_index"] == [int(r["ix"]), int(r["iy"])]
        assert all(r["verbalization_50w"] == "" for r in rows)       # the script never writes prose

    def test_dx_005_gives_1681_points_one_to_one(self, run_dir, tmp_path):
        od = str(tmp_path / "s005")
        grid_scan.main(["--out", run_dir, "--dx", "0.05", "--outdir", od])
        rows = read_csv(os.path.join(od, "k5_grid_scan_raw.csv"))
        details = json.load(open(os.path.join(od, "k5_grid_scan_details.json")))
        keys = [f"{r['x']},{r['y']}" for r in rows]
        assert len(rows) == 1681 and len(set(keys)) == 1681 and keys == list(details)
        xs = sorted({float(r["x"]) for r in rows})
        assert len(xs) == 41 and np.allclose(np.diff(xs), 0.05, atol=1e-12)
        meta = json.load(open(os.path.join(od, "k5_grid_scan_meta.json")))
        assert meta["grid"]["dx"] == pytest.approx(0.05) and meta["grid"]["n_points"] == 1681
        assert meta["grid"]["coordinate_decimals"] == 2

    def test_probabilities_entropy_and_effective_support(self, scan01, run_dir):
        V = np.load(os.path.join(scan01, "k5_grid_scan_values.npz"))
        N = len(np.load(os.path.join(run_dir, "coords.npy")))
        p = V["p_cluster"]
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-12)
        assert (p >= 0).all()
        H = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0).sum(1) / np.log(5)
        np.testing.assert_allclose(V["H_cluster"], H, rtol=1e-12, atol=1e-15)
        assert ((V["H_cluster"] >= 0) & (V["H_cluster"] <= 1 + 1e-12)).all()
        np.testing.assert_allclose(V["N_eff"], N ** V["H_field"], rtol=1e-12)
        assert ((V["N_eff"] >= 1 - 1e-9) & (V["N_eff"] <= N + 1e-9)).all()
        dom = V["dominant_C"]
        assert (p[np.arange(len(p)), dom] == p.max(1)).all()

    def test_metric_relations_and_finite_distance(self, scan01, run_dir):
        from kmlib import L2Field
        V = np.load(os.path.join(scan01, "k5_grid_scan_values.npz"))
        g, lam, e1 = V["metric_g"], V["lam"], V["e1"]
        tr = g[:, 0, 0] + g[:, 1, 1]
        np.testing.assert_allclose(V["grad_norm"] ** 2, tr, rtol=1e-12)
        np.testing.assert_allclose(lam.sum(1), tr, rtol=1e-12)
        np.testing.assert_allclose(V["dirgrad"] ** 2, lam, rtol=1e-12)
        np.testing.assert_allclose(np.einsum("ma,mab,mb->m", e1, g, e1), lam[:, 0], rtol=1e-12)
        # ||v(P + s e) - v(P)||^2 = 2 (1 - cos) = s^2 e'ge + O(s^3). Away from the kink lines the
        # relative error is O(s): at most 2e-4 at s = 1e-4 on this map, against rtol 1e-3.
        P = np.load(os.path.join(run_dir, "coords.npy"))
        fld = L2Field(np.load(os.path.join(run_dir, "work", "gram_l2.npy")), P, h_mode="global")
        Q = np.column_stack([V["x"], V["y"]])
        smooth = np.flatnonzero(~V["metric_kink"])
        s = 1e-4
        for k, e in ((0, e1[smooth]), (1, np.column_stack([-e1[smooth, 1], e1[smooth, 0]]))):
            pred = 0.5 * s ** 2 * lam[smooth, k]
            got = 1 - fld.cos_between(Q[smooth], Q[smooth] + s * e)
            np.testing.assert_allclose(got, pred, rtol=1e-3)

    def test_kink_lines_are_flagged_and_are_where_delta_matters(self, scan01, run_dir):
        from kmlib import L2Field
        V = np.load(os.path.join(scan01, "k5_grid_scan_values.npz"))
        kink = V["metric_kink"].astype(bool)
        on_axes = (V["x"] == 0) | (V["y"] == 0)
        assert np.array_equal(kink, on_axes)                          # corner anchors => x = 0, y = 0
        P = np.load(os.path.join(run_dir, "coords.npy"))
        fld = L2Field(np.load(os.path.join(run_dir, "work", "gram_l2.npy")), P, h_mode="global")
        Q = np.column_stack([V["x"], V["y"]])
        tr = lambda d: np.einsum("mii->m", fld.metric(Q, delta=d))
        rel = np.abs(tr(1e-4) - tr(1e-3)) / tr(1e-3)
        # O(delta^2) off the lines (max 4e-6 on this map), O(delta) at every point on them (min 1e-4)
        assert rel[~kink].max() < 1e-5 < rel[kink].min()

    def test_words_and_numbers_only_runs_agree_on_every_number(self, run_dir, scan01, tmp_path):
        od = str(tmp_path / "nw")
        grid_scan.main(["--out", run_dir, "--no-words", "--outdir", od])
        assert not os.path.exists(os.path.join(od, "k5_grid_scan_details.json"))
        a = read_csv(os.path.join(scan01, "k5_grid_scan_raw.csv"))
        b = read_csv(os.path.join(od, "k5_grid_scan_raw.csv"))
        assert a == b
        assert json.load(open(os.path.join(od, "k5_grid_scan_meta.json")))["settings"]["words"] is False

    def test_rerun_in_another_process_is_byte_identical(self, run_dir, tmp_path):
        outs = []
        for seed in ("1", "2"):
            od = str(tmp_path / f"p{seed}")
            p = run_tool("grid_scan.py", "--out", run_dir, "--outdir", od, "--topk-l2", "15",
                         "--topk-l1", "15", env={"PYTHONHASHSEED": seed})
            assert p.returncode == 0, p.stderr
            outs.append(od)
        for name in ("k5_grid_scan_raw.csv", "k5_grid_scan_details.json", "k5_grid_scan_meta.json"):
            assert sha(os.path.join(outs[0], name)) == sha(os.path.join(outs[1], name)), name
        A, B = (np.load(os.path.join(o, "k5_grid_scan_values.npz")) for o in outs)
        assert A.files == B.files and all(np.array_equal(A[k], B[k]) for k in A.files)

    def test_matches_make_evidence_with_the_same_settings(self, run_dir, scan01, tmp_path):
        """make_evidence.py --readout global --topk 15 must see what the scan saw at that point.

        make_evidence.py keeps each document's words in a set, so when two words tie as the
        representative of a merged term (same df, same length) Python's per-process string hash
        picks one. It is therefore run under several PYTHONHASHSEED values: where it agrees with
        itself the scan must give the same word; where it does not, the scan (which orders words)
        must give a word of the same tie class, and df and source documents must agree throughout.
        """
        details = json.load(open(os.path.join(scan01, "k5_grid_scan_details.json")))
        V = np.load(os.path.join(scan01, "k5_grid_scan_values.npz"))
        row = {k: i for i, k in enumerate(details)}
        strip = lambda recs: [(e["df"], [s["doc"] for s in e.get("source_docs", [])]) for e in recs]
        corpus_words = {w for d in json.load(open(os.path.join(run_dir, "corpus", "docs_clean.json")))
                        for w in d["text"].split()}
        for x, y in ((-0.5, 0.0), (0.3, 0.7), (1.0, -1.0)):
            key = f"{x:.1f},{y:.1f}"
            d, i = details[key], row[key]
            evs = []
            for seed in ("0", "1", "2", "3"):
                ev_path = str(tmp_path / f"ev_{key}_{seed}.json")
                p = subprocess.run([sys.executable, os.path.join(CODE, "make_evidence.py"), "--x", str(x),
                                    "--y", str(y), "--out", run_dir, "--data", os.path.join(run_dir, "work"),
                                    "--code", CODE, "--readout", "global", "--topk", "15", "--outfile", ev_path],
                                   capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED=seed))
                assert p.returncode == 0, p.stderr
                evs.append(json.load(open(ev_path)))
            for lens in ("theme_lens_L2", "concentration_lens_L1"):
                assert all(strip(ev[lens]) == strip(d[lens]) for ev in evs), (key, lens)
                for pos, e in enumerate(d[lens]):
                    seen = {ev[lens][pos]["term"] for ev in evs}
                    if len(seen) == 1:
                        assert e["term"] in seen, (key, lens, pos)
                    else:
                        assert all(len(t) == len(e["term"]) for t in seen), (key, lens, pos, seen)
                        assert e["term"] in corpus_words, (key, lens, pos)
            ev = evs[0]
            assert [c["doc"] for c in d["contributing_documents_top10"][:3]] == \
                   [c["doc"] for c in ev["contributing_documents_top3"]]
            # make_evidence.py rounds to 3 decimals (N_eff to 1), so the full-precision value is within half a unit
            assert abs(V["H_field"][i] - ev["mixture_entropy_H"]) <= 5e-4 + 1e-12
            assert abs(V["N_eff"][i] - ev["effective_contributing_documents"]) <= 0.05 + 1e-9
            assert abs(V["u_RBF"][i] - ev["relative_uncertainty_u"]) <= 5e-4 + 1e-12

    def test_other_k_via_recluster_labels(self, fresh_run, tmp_path):
        recluster.main(["--out", fresh_run, "--k", "3", "--no-terms"])
        grid_scan.main(["--out", fresh_run, "--no-words", "--labels",
                        os.path.join(fresh_run, "recluster", "cluster_labels_k3.npy")])
        od = os.path.join(fresh_run, "grid_scan", "k3_dx0.1_nowords")
        rows = read_csv(os.path.join(od, "k3_grid_scan_raw.csv"))
        assert [c for c in rows[0] if c.startswith("p_C")] == ["p_C0", "p_C1", "p_C2"]
        meta = json.load(open(os.path.join(od, "k3_grid_scan_meta.json")))
        assert meta["K"] == 3 and meta["settings"]["cluster_labels"] == os.path.join("recluster", "cluster_labels_k3.npy")
        grid_viz.main(["--scan", od, "--out", fresh_run])
        assert os.path.exists(os.path.join(od, "k3_candidate_prelist.csv"))

    def test_default_outdir_is_inside_the_run_and_never_overwritten(self, fresh_run):
        before = tree_hashes(fresh_run)
        grid_scan.main(["--out", fresh_run, "--no-words"])
        od = os.path.join(fresh_run, "grid_scan", "k5_dx0.1_nowords")
        assert os.path.exists(os.path.join(od, "k5_grid_scan_raw.csv"))
        after = tree_hashes(fresh_run)
        assert {k: v for k, v in after.items() if not k.startswith("grid_scan" + os.sep)} == before
        with pytest.raises(ArtifactError, match="refusing to overwrite"):
            grid_scan.main(["--out", fresh_run, "--no-words"])
        grid_scan.main(["--out", fresh_run, "--no-words", "--force"])

    def test_missing_artifacts_fail_and_are_listed(self, fresh_run):
        os.remove(os.path.join(fresh_run, "work", "gpr_model.pkl"))
        os.remove(os.path.join(fresh_run, "corpus", "docs_clean.json"))
        with pytest.raises(ArtifactError) as e:
            grid_scan.main(["--out", fresh_run])
        assert "gpr_model.pkl" in str(e.value) and "docs_clean.json" in str(e.value)
        assert not os.path.exists(os.path.join(fresh_run, "grid_scan"))

    def test_gpr_of_another_map_is_rejected(self, fresh_run):
        os.remove(os.path.join(fresh_run, "coordinates_2d.csv"))       # else that check fires first
        P = np.load(os.path.join(fresh_run, "coords.npy"))
        P[12] += 0.01                                                   # the map changed, the GPR did not
        np.save(os.path.join(fresh_run, "coords.npy"), P)
        with pytest.raises(ArtifactError, match="not fitted on"):
            grid_scan.main(["--out", fresh_run, "--no-words"])

    def test_document_order_mismatch_is_rejected(self, fresh_run):
        path = os.path.join(fresh_run, "corpus_metadata.csv")
        lines = open(path).read().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        open(path, "w").write("\n".join(lines) + "\n")
        with pytest.raises(ArtifactError, match="doc_id"):
            grid_scan.main(["--out", fresh_run, "--no-words"])

    def test_docs_clean_order_mismatch_is_rejected(self, fresh_run):
        path = os.path.join(fresh_run, "corpus", "docs_clean.json")
        docs = json.load(open(path))
        docs[0]["filename"], docs[1]["filename"] = docs[1]["filename"], docs[0]["filename"]
        json.dump(docs, open(path, "w"))
        with pytest.raises(ArtifactError, match="docs_clean"):
            grid_scan.main(["--out", fresh_run])

    def test_too_few_documents(self, tmp_path):
        run = build_run(tmp_path, n_docs=8, n_clusters=2)
        with pytest.raises(ArtifactError, match="at least 9"):
            grid_scan.main(["--out", run, "--no-words"])

    def test_cli_rejects_bad_dx_before_reading_anything(self, tmp_path):
        p = run_tool("grid_scan.py", "--out", str(tmp_path / "nowhere"), "--dx", "0.3")
        assert p.returncode == 2 and "does not divide" in p.stderr


# ============================================================ grid_viz
class TestGridViz:
    @pytest.fixture
    def scan_copy(self, scan01, tmp_path):
        d = str(tmp_path / "scan")
        shutil.copytree(scan01, d)
        return d

    def test_outputs_and_order_independence(self, scan_copy, run_dir, tmp_path):
        grid_viz.main(["--scan", scan_copy, "--out", run_dir])
        ref = open(os.path.join(scan_copy, "k5_candidate_prelist.csv")).read()
        for f in ("heatmap_Hcluster.png", "heatmap_white_support_uncertainty.png", "k5_candidate_prelist_meta.json"):
            assert os.path.getsize(os.path.join(scan_copy, f)) > 0
        # the same scan with its rows shuffled must give the same candidate list
        path = os.path.join(scan_copy, "k5_grid_scan_raw.csv")
        lines = open(path).read().splitlines()
        body = lines[1:]
        np.random.default_rng(0).shuffle(body)
        open(path, "w").write("\n".join([lines[0]] + body) + "\n")
        grid_viz.main(["--scan", scan_copy, "--out", run_dir, "--force"])
        assert open(os.path.join(scan_copy, "k5_candidate_prelist.csv")).read() == ref

    @pytest.mark.parametrize("edit, msg", [("drop", "rows"), ("duplicate", "twice"), ("move", "does not match")])
    def test_rejects_a_damaged_grid(self, scan_copy, run_dir, edit, msg):
        path = os.path.join(scan_copy, "k5_grid_scan_raw.csv")
        rows = read_csv(path)
        if edit == "drop":
            rows = rows[:-1]
        elif edit == "duplicate":
            rows[5] = dict(rows[4])
        else:
            rows[7]["y"] = "0.95"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with pytest.raises(ArtifactError, match=msg):
            grid_viz.main(["--scan", scan_copy, "--out", run_dir])

    def test_rejects_coordinates_of_another_run(self, scan_copy, fresh_run):
        P = np.load(os.path.join(fresh_run, "coords.npy"))
        P[12] += 0.01
        np.save(os.path.join(fresh_run, "coords.npy"), P)
        with pytest.raises(ArtifactError, match="sha256"):
            grid_viz.main(["--scan", scan_copy, "--out", fresh_run])


# ============================================================ artifacts of two runs mixed together
@pytest.fixture(scope="module")
def other_run(tmp_path_factory):
    """A second synthetic run: different text and map, same file names (KM0001 ...)."""
    return build_run(tmp_path_factory.mktemp("other"), seed=1)


def mixed(run, other, tmp_path, *names):
    """A copy of run with the named files (relative paths) taken from other."""
    d = str(tmp_path / "mixed")
    shutil.copytree(run, d)
    for name in names:
        shutil.copyfile(os.path.join(other, name), os.path.join(d, name))
    return d


class TestMixedRuns:
    @pytest.mark.parametrize("names, words, msg", [
        (("corpus/docs_clean.json",), True, "docs_clean.json entry"),
        (("corpus_metadata.csv",), False, "manifest.json"),
        (("corpus_metadata.csv", "manifest.json", "coordinates_2d.csv"), False, "does not describe coords.npy"),
        (("work/cluster_labels.npy",), False, "k-means partition"),
        (("work/X_raw.npy", "work/l2_norms.npy"), True, "X_raw.npy"),
        (("work/svd_scores.npy",), False, "svd_scores|zscale"),
        (("work/gpr_model.pkl",), False, "not fitted on"),
    ])
    def test_grid_scan_refuses_artifacts_of_another_run(self, run_dir, other_run, tmp_path, names, words, msg):
        d = mixed(run_dir, other_run, tmp_path, *names)
        with pytest.raises(ArtifactError, match=msg):
            grid_scan.main(["--out", d, "--outdir", str(tmp_path / "o")] + ([] if words else ["--no-words"]))
        assert not os.path.exists(tmp_path / "o")

    def test_metadata_of_another_run_is_caught_without_a_manifest(self, run_dir, other_run, tmp_path):
        d = mixed(run_dir, other_run, tmp_path, "corpus_metadata.csv")
        os.remove(os.path.join(d, "manifest.json"))
        for tool, args in ((grid_scan, ["--out", d, "--outdir", str(tmp_path / "g")]),
                           (recluster, ["--out", d, "--k", "3", "--outdir", str(tmp_path / "r")])):
            with pytest.raises(ArtifactError, match="chars_cleaned"):
                tool.main(args)

    def test_custom_labels_are_accepted_only_when_declared(self, run_dir, other_run, tmp_path):
        foreign = os.path.join(other_run, "work", "cluster_labels.npy")
        with pytest.raises(ArtifactError, match="custom-labels"):
            grid_scan.main(["--out", run_dir, "--no-words", "--labels", foreign, "--outdir", str(tmp_path / "a")])
        grid_scan.main(["--out", run_dir, "--no-words", "--labels", foreign, "--custom-labels",
                        "--outdir", str(tmp_path / "b")])
        meta = json.load(open(tmp_path / "b" / "k5_grid_scan_meta.json"))
        assert meta["settings"]["custom_labels"] is True
        assert meta["consistency_checks"]["labels are this run's k-means partition"].startswith("skipped")

    def test_every_check_runs_on_a_complete_run(self, scan01):
        checks = json.load(open(os.path.join(scan01, "k5_grid_scan_meta.json")))["consistency_checks"]
        assert all(v.startswith("ok") for v in checks.values()), checks

    def test_add_cluster_refuses_labels_or_coordinates_of_another_run(self, run_dir, other_run, tmp_path):
        with pytest.raises(ArtifactError, match="k-means partition"):
            add_cluster_to_csv.add_cluster_column(os.path.join(run_dir, "coordinates_2d.csv"),
                                                  os.path.join(other_run, "work", "cluster_labels.npy"),
                                                  str(tmp_path / "a.csv"), run=run_dir)
        n, checks = add_cluster_to_csv.add_cluster_column(
            os.path.join(run_dir, "coordinates_2d.csv"), os.path.join(other_run, "work", "cluster_labels.npy"),
            str(tmp_path / "b.csv"), run=run_dir, custom_labels=True)
        assert checks["labels are this run's k-means partition"].startswith("skipped")
        with pytest.raises(ArtifactError, match="coords.npy"):
            add_cluster_to_csv.add_cluster_column(os.path.join(other_run, "coordinates_2d.csv"),
                                                  os.path.join(run_dir, "work", "cluster_labels.npy"),
                                                  str(tmp_path / "c.csv"), run=run_dir)
        assert not (tmp_path / "a.csv").exists() and not (tmp_path / "c.csv").exists()

    def test_recluster_refuses_a_foreign_svd(self, run_dir, other_run, tmp_path):
        d = mixed(run_dir, other_run, tmp_path, "work/svd_scores.npy")
        with pytest.raises(ArtifactError):
            recluster.main(["--out", d, "--k", "3", "--outdir", str(tmp_path / "o")])
        assert not os.path.exists(tmp_path / "o")


# ============================================================ determinism of the word resolution
TIE_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
from grid_scan import LensWords
docs = [{"text": "betaxyz alphxyz gammaword"}, {"text": "deltaword"}]
meta = [{"title": "a"}, {"title": "b"}]
print(LensWords(["xyz "], [], docs, meta).lens_words([("xyz", 0)])[0]["term"])
"""


class TestWordTies:
    def test_representative_word_tie_is_broken_alphabetically(self):
        # "alphxyz" and "betaxyz" end in "xyz", occur in the same single document and have the
        # same length: a tie that make_evidence.py leaves to Python's string-hash order
        docs = [{"text": "betaxyz alphxyz gammaword"}, {"text": "deltaword"}]
        lens = grid_scan.LensWords(["xyz "], [], docs, [{"title": "a"}, {"title": "b"}])
        assert lens.lens_words([("xyz", 0)]) == [{"term": "alphxyz", "df": 1,
                                                  "source_docs": [{"doc": 0, "title": "a"}]}]

    def test_tie_does_not_depend_on_the_hash_seed(self, tmp_path):
        got = set()
        for seed in range(8):
            p = subprocess.run([sys.executable, "-c", TIE_SCRIPT, CODE], capture_output=True, text=True,
                               env=dict(os.environ, PYTHONHASHSEED=str(seed)))
            assert p.returncode == 0, p.stderr
            got.add(p.stdout.strip())
        assert got == {"alphxyz"}


# ============================================================ the co-author's definitions, recomputed
def original_rows(run, dx=0.1, delta=1e-3):
    """The numeric CSV fields exactly as scan/k5_grid_scan.py computes and formats them."""
    import csv as _csv
    import pickle
    from kmlib import GPR, L2Field, sph_entropy, sph_weights
    P = np.load(os.path.join(run, "coords.npy")); N = len(P)
    lab = np.load(os.path.join(run, "work", "cluster_labels.npy")); K = int(lab.max()) + 1
    G = np.load(os.path.join(run, "work", "gram_l2.npy"))
    meta = list(_csv.DictReader(open(os.path.join(run, "corpus_metadata.csv"))))
    gm = pickle.load(open(os.path.join(run, "work", "gpr_model.pkl"), "rb"))
    gp = GPR(kernel="rbf")
    for k in ("theta", "X", "alpha", "L", "noise"):
        setattr(gp, k, gm[k])
    Z = np.load(os.path.join(run, "work", "svd_scores.npy"))[:, :10]; zs = Z.std()
    gp_m32 = GPR(kernel="matern32", seed=0, n_restarts=10).fit(P, Z / zs)
    n = int(round(2.0 / dx)) + 1
    xs = np.round(np.linspace(-1, 1, n), 10)
    GX, GY = np.meshgrid(xs, xs, indexing="ij")
    Q = np.column_stack([GX.ravel(), GY.ravel()]); M = len(Q)
    dmat = np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=2)
    near = dmat.argmin(1); dsort = np.sort(dmat, axis=1)
    d_nn, d_k5, d_k8 = dsort[:, 0], dsort[:, 4], dsort[:, 7]
    w = sph_weights(Q, P, h_mode="global")
    pc = np.zeros((M, K))
    for c in range(K):
        pc[:, c] = w[:, lab == c].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hc = -np.where(pc > 0, pc * np.log(pc), 0).sum(1) / np.log(K)
    ordc = np.argsort(-pc, axis=1); dom, sec = ordc[:, 0], ordc[:, 1]
    p_dom = pc[np.arange(M), dom]; p_sec = pc[np.arange(M), sec]
    Hf = sph_entropy(w, N); Neff = N ** Hf
    maxw = w.max(1); top3s = np.sort(w, axis=1)[:, -3:].sum(1)
    Hf_ad = sph_entropy(sph_weights(Q, P, h_mode="knn_adaptive", knn_k=8), N)
    u_rbf = gp.rel_uncertainty(Q); u_m32 = gp_m32.rel_uncertainty(Q)
    g = L2Field(G, P, h_mode="global").metric(Q, delta=delta)
    grad_norm = np.sqrt(g[:, 0, 0] + g[:, 1, 1])
    lam = np.empty((M, 2)); e1 = np.empty((M, 2))
    for i in range(M):
        vals, vecs = np.linalg.eigh(g[i]); lam[i] = vals[::-1]; v1 = vecs[:, 1]
        if v1[0] < 0 or (v1[0] == 0 and v1[1] < 0):
            v1 = -v1
        e1[i] = v1
    theta1 = np.degrees(np.arctan2(e1[:, 1], e1[:, 0]))
    aniso = lam[:, 0] / np.maximum(lam[:, 1], 1e-300)
    rows = []
    for i in range(M):
        rows.append([f"{Q[i,0]:.1f}", f"{Q[i,1]:.1f}", str(int(near[i])), meta[near[i]]["title"][:50],
                     f"{d_nn[i]:.4f}", f"{d_k5[i]:.4f}", f"{d_k8[i]:.4f}"]
                    + [f"{pc[i,c]:.4f}" for c in range(K)]
                    + [f"{Hc[i]:.4f}", str(int(dom[i])), f"{p_dom[i]:.4f}", str(int(sec[i])),
                       f"{p_sec[i]:.4f}", f"{p_dom[i]-p_sec[i]:.4f}",
                       f"{Hf[i]:.4f}", f"{Neff[i]:.1f}", f"{Hf_ad[i]:.4f}", f"{maxw[i]:.4f}", f"{top3s[i]:.4f}",
                       f"{u_rbf[i]:.4f}", f"{u_m32[i]:.4f}", f"{grad_norm[i]:.4f}", f"{lam[i,0]:.5f}",
                       f"{lam[i,1]:.5f}", f"{aniso[i]:.2f}", f"{theta1[i]:.1f}", f"{e1[i,0]:.4f}",
                       f"{e1[i,1]:.4f}", f"{np.sqrt(lam[i,0]):.4f}", f"{np.sqrt(lam[i,1]):.4f}"])
    return rows


class TestOriginalDefinitions:
    def test_numeric_columns_equal_the_original_formulas(self, scan01, run_dir):
        """Every column of the co-author's CSV, recomputed here independently, string for string."""
        rows = read_csv(os.path.join(scan01, "k5_grid_scan_raw.csv"))
        cols = list(rows[0])[:list(rows[0]).index("dirgrad2") + 1]
        V = np.load(os.path.join(scan01, "k5_grid_scan_values.npz"))
        assert not V["lam2_clipped"].any()                  # no degenerate point on this map
        assert [[r[c] for c in cols] for r in rows] == original_rows(run_dir)

    def test_size_ties_beyond_sixteen_clusters_keep_the_raw_order(self, monkeypatch):
        # numpy's default sort is only stable for short arrays; 20 clusters of alternating sizes
        raw = np.repeat(np.arange(20), [2, 3] * 10)
        monkeypatch.setattr(recluster, "kmeans", lambda Z, K, seed=0: (raw, None))
        lab, _ = recluster.size_ranked_kmeans(np.zeros((len(raw), 10)), 20)
        order = [int(lab[raw == c][0]) for c in range(20)]
        assert order == [10 + i // 2 if c % 2 == 0 else i // 2 for i, c in enumerate(range(20))]

    def test_eigen_gap_threshold_is_1e_3(self):
        g = np.array([[[1.0, 0.0], [0.0, 1.0 - 5e-4]], [[1.0, 0.0], [0.0, 1.0 - 2e-3]]])
        assert grid_scan.eigen_analysis(g)["ill"].tolist() == [True, False]
        assert grid_scan.EIG_GAP_RTOL == 1e-3


# ============================================================ output locations
class TestOutputGuards:
    def test_nothing_is_written_into_the_shipped_data(self, tmp_path):
        if not os.path.exists(os.path.join(DERIVED, "svd_scores.npy")):
            pytest.skip("derived artifacts not present")
        before = tree_hashes(DERIVED)
        with pytest.raises(ArtifactError, match="shipped data"):
            recluster.main(["--out", DERIVED, "--data", DERIVED, "--k", "3", "--no-terms"])
        with pytest.raises(ArtifactError, match="shipped data"):
            add_cluster_to_csv.add_cluster_column(os.path.join(DERIVED, "coordinates_2d.csv"),
                                                  os.path.join(DERIVED, "cluster_labels.npy"),
                                                  os.path.join(DERIVED, "coordinates_2d_clusters.csv"))
        assert tree_hashes(DERIVED) == before

    def test_a_symlink_into_the_shipped_data_does_not_get_around_the_guard(self, tmp_path):
        if not os.path.exists(os.path.join(DERIVED, "svd_scores.npy")):
            pytest.skip("derived artifacts not present")
        link = tmp_path / "link"
        os.symlink(DERIVED, link)
        before = tree_hashes(DERIVED)
        with pytest.raises(ArtifactError, match="shipped data"):
            recluster.main(["--out", DERIVED, "--data", DERIVED, "--k", "3", "--no-terms",
                            "--outdir", str(link / "sub")])
        with pytest.raises(ArtifactError, match="shipped data"):               # default outdir via the link
            recluster.main(["--out", str(link), "--data", str(link), "--k", "3", "--no-terms"])
        assert tree_hashes(DERIVED) == before

    def test_force_does_not_leave_files_of_an_earlier_scan(self, run_dir, tmp_path):
        od = str(tmp_path / "s")
        grid_scan.main(["--out", run_dir, "--outdir", od])
        grid_viz.main(["--scan", od, "--out", run_dir])
        with pytest.raises(ArtifactError, match="earlier scan") as e:
            grid_scan.main(["--out", run_dir, "--outdir", od, "--no-words", "--force"])
        assert "k5_grid_scan_details.json" in str(e.value) and "k5_candidate_prelist.csv" in str(e.value)

    def test_case_variant_of_the_input_is_the_input(self, fresh_run, tmp_path):
        src = os.path.join(fresh_run, "coordinates_2d.csv")
        variant = os.path.join(fresh_run, "Coordinates_2D.csv")
        if not os.path.exists(variant):
            pytest.skip("case-sensitive file system")
        before = sha(src)
        with pytest.raises(ArtifactError, match="new file"):
            add_cluster_to_csv.add_cluster_column(src, os.path.join(fresh_run, "work", "cluster_labels.npy"),
                                                  variant, force=True)
        assert sha(src) == before
