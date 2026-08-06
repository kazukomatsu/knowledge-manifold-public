# -*- coding: utf-8 -*-
"""Integration test: the published metrics must reproduce from the shipped artifacts.

This exercises the actual data/derived/ contents, which hold no corpus text —
only the 100x100 Gram matrix, the coordinates and the reference metric JSONs.
If this test fails, either the shipped artifacts and the shipped reference
values have drifted apart, or a metric implementation changed.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "derived")
sys.path.insert(0, os.path.join(ROOT, "code"))

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "gram_l2.npy")),
    reason="derived artifacts not present",
)


def test_verify_reference_exits_zero():
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "code", "verify_reference.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "METRICS REPRODUCED" in proc.stdout


def test_e1_recomputes_to_published_values():
    from verify_reference import compute_e1
    G = np.load(os.path.join(DATA, "gram_l2.npy"))
    P = np.load(os.path.join(DATA, "coords.npy"))
    audit = json.load(open(os.path.join(DATA, "map_audit.json")))
    ref = json.load(open(os.path.join(DATA, "e1_distance_preservation.json")))
    got = compute_e1(G, P, audit["kappa"], audit["alpha"])
    for key in ("spearman", "knn_preservation_k7", "trustworthiness_k7", "continuity_k7"):
        assert got[key] == pytest.approx(ref[key], abs=1e-12), key


@pytest.mark.parametrize("h_mode", ["global", "knn_adaptive"])
def test_e2_loo_recomputes_to_published_values(h_mode):
    from verify_reference import compute_e2
    G = np.load(os.path.join(DATA, "gram_l2.npy"))
    P = np.load(os.path.join(DATA, "coords.npy"))
    ref = json.load(open(os.path.join(DATA, f"e2_loo_{h_mode}.json")))
    got = compute_e2(G, P, h_mode)
    assert got["loo_cosine_mean"] == pytest.approx(ref["loo_cosine_mean"], abs=1e-12)
    assert got["nearest_paper_recovery_rate"] == pytest.approx(
        ref["nearest_paper_recovery_rate"], abs=1e-12)


def test_shipped_artifacts_are_self_consistent():
    """The shipped coordinates must be the ones the shipped audit describes."""
    P = np.load(os.path.join(DATA, "coords.npy"))
    audit = json.load(open(os.path.join(DATA, "map_audit.json")))
    assert len(P) == audit["N"]
    for slot, doc in enumerate(audit["anchor_docs"]):
        expected = audit["anchor_positions_after_finalization"][slot]
        np.testing.assert_allclose(P[doc], expected, atol=1e-9)


def test_gauge_fixing_was_applied_to_the_shipped_run():
    """Guards against re-publishing artifacts produced before the D4 fix."""
    audit = json.load(open(os.path.join(DATA, "map_audit.json")))
    assert "anchor_assignment_tie_count" in audit, (
        "shipped map_audit.json predates the anchor gauge fix — regenerate it"
    )
    assert audit["anchor_assignment_tie_count"] == 8


def test_no_corpus_text_is_shipped():
    """The derived directory must not contain the withheld reconstructive artifacts."""
    forbidden = {"X_raw.npy", "vocab.json", "tfidf_vectorizer.pkl"}
    present = forbidden & set(os.listdir(DATA))
    assert not present, f"must not ship: {present}"


def test_no_absolute_paths_leak_in_shipped_json():
    leaks = []
    for name in os.listdir(DATA):
        if not name.endswith(".json"):
            continue
        blob = open(os.path.join(DATA, name), encoding="utf-8").read()
        for needle in ("/Users/", "/private/tmp", "/sessions/", "/home/"):
            if needle in blob:
                leaks.append((name, needle))
    assert not leaks, f"absolute paths leaked: {leaks}"
