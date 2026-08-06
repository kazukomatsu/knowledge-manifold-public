# -*- coding: utf-8 -*-
"""Unit tests for kmlib. Synthetic data only — no corpus text, no real papers."""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))

from kmlib import (ANCHOR_TIE_RTOL, assign_anchors, cosine_dist_matrix,
                   knn_preservation, normalized_stress, spearman, sph_weights,
                   trustworthiness_continuity)

PERIM_9 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1],
                    [1, 0], [-1, 0], [0, 1], [0, -1]], float)
ANCHOR_POS_9 = np.vstack([PERIM_9, [[0.0, 0.0]]])


def d4_images(anchor_pos):
    """Index permutations induced by the 8 symmetries of the square."""
    out = []
    for k in range(4):
        c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
        rot = np.array([[c, -s], [s, c]], float)
        for mirror in (np.eye(2), np.array([[-1, 0], [0, 1]], float)):
            img = anchor_pos @ (rot @ mirror).T
            sigma = [int(np.argmin(np.linalg.norm(anchor_pos - img[i], axis=1)))
                     for i in range(len(anchor_pos))]
            out.append(sigma)
    return out


def degenerate_distance_matrix(scale=0.5):
    """A cos-distance matrix whose optimal anchor assignment is exactly D4-degenerate.

    Nine documents placed exactly at the anchor layout (scaled), so the layout's
    own symmetry group maps optimal assignments onto optimal assignments.
    """
    pts = ANCHOR_POS_9 * scale
    return np.linalg.norm(pts[:, None] - pts[None], axis=2)


class TestAnchorGaugeFixing:
    """Regression tests for the D4 gauge freedom in the anchor assignment.

    The 9-point anchor layout is symmetric under the dihedral group of the
    square, so the assignment optimum is exactly 8-fold degenerate. Picking the
    winner by naive `cost < best` lets floating-point noise decide, which
    mirrors or rotates the whole map when the BLAS/SciPy build changes — while
    leaving every pairwise-distance metric invariant, so it passes validation
    unnoticed. These tests pin the gauge.
    """

    def test_layout_has_eight_symmetries(self):
        sigmas = {tuple(s) for s in d4_images(ANCHOR_POS_9)}
        assert len(sigmas) == 8

    def test_optimum_is_eightfold_degenerate(self):
        D = degenerate_distance_matrix()
        _, _, n_tied = assign_anchors(D, np.arange(9), ANCHOR_POS_9, 8)
        assert n_tied == 8

    def test_gauge_is_stable_under_rounding_noise(self):
        """The whole point: 1e-14 perturbations must not change the assignment."""
        D0 = degenerate_distance_matrix()
        perm0, _, _ = assign_anchors(D0, np.arange(9), ANCHOR_POS_9, 8)
        rng = np.random.default_rng(0)
        for _ in range(12):
            noise = rng.normal(0, 1e-14, D0.shape)
            noise = (noise + noise.T) / 2
            np.fill_diagonal(noise, 0.0)
            perm, _, n_tied = assign_anchors(D0 + noise, np.arange(9), ANCHOR_POS_9, 8)
            assert perm == perm0
            assert n_tied == 8

    def test_returns_lexicographic_minimum_of_tied_orbit(self):
        D = degenerate_distance_matrix()
        reps = np.arange(9)
        perm, _, _ = assign_anchors(D, reps, ANCHOR_POS_9, 8)
        chosen = tuple(int(x) for x in reps[list(perm)])

        A2D = np.linalg.norm(ANCHOR_POS_9[:, None] - ANCHOR_POS_9[None], axis=2)
        iu_a, iu_p = np.triu_indices(9, 1), np.triu_indices(8, 1)
        mean2d = A2D[:8, :8][iu_p].mean()

        def cost_of(p):
            sub = D[np.ix_(reps[list(p)], reps[list(p)])]
            alpha = mean2d / sub[:8, :8][iu_p].mean()
            return float(np.sum((A2D[iu_a] - alpha * sub[iu_a]) ** 2))

        cmin = min(cost_of(p) for p in itertools.permutations(range(9)))
        tied = [tuple(int(x) for x in reps[list(p)])
                for p in itertools.permutations(range(9))
                if cost_of(p) - cmin <= ANCHOR_TIE_RTOL * abs(cmin) + 1e-12]
        assert chosen == min(tied)

    def test_gauge_invariant_under_mirroring_the_input(self):
        """Mirroring the document configuration must not change the chosen tuple.

        A mirrored corpus is the same corpus; only the winner of the degenerate
        orbit could differ, and the tie-break must absorb that.
        """
        pts = ANCHOR_POS_9 * 0.5
        D_mirror = np.linalg.norm((pts * [-1, 1])[:, None] - (pts * [-1, 1])[None], axis=2)
        perm_a, _, _ = assign_anchors(degenerate_distance_matrix(), np.arange(9), ANCHOR_POS_9, 8)
        perm_b, _, _ = assign_anchors(D_mirror, np.arange(9), ANCHOR_POS_9, 8)
        reps = np.arange(9)
        assert tuple(reps[list(perm_a)]) == tuple(reps[list(perm_b)])

    def test_four_anchor_layout_also_terminates(self):
        pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]], float)
        D = np.linalg.norm((pts * 0.5)[:, None] - (pts * 0.5)[None], axis=2)
        perm, alpha, n_tied = assign_anchors(D, np.arange(5), pts, 4)
        assert len(perm) == 5
        assert alpha > 0
        assert n_tied >= 1


class TestSphWeights:
    @pytest.mark.parametrize("h_mode", ["global", "knn_adaptive", "density_adaptive"])
    def test_rows_are_a_probability_distribution(self, h_mode):
        rng = np.random.default_rng(1)
        Y = rng.uniform(-1, 1, (40, 2))
        Q = rng.uniform(-0.9, 0.9, (7, 2))
        W = sph_weights(Q, Y, h_mode=h_mode)
        assert W.shape == (7, 40)
        assert np.all(W >= 0)
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-12)

    def test_fixed_mode_normalized(self):
        rng = np.random.default_rng(2)
        Y = rng.uniform(-1, 1, (30, 2))
        W = sph_weights(np.array([[0.0, 0.0]]), Y, h_mode="fixed", h_fixed=0.8)
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-12)

    def test_all_zero_kernel_falls_back_to_nearest_neighbour(self):
        Y = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        q = np.array([[10.0, 10.0]])
        W = sph_weights(q, Y, h_mode="fixed", h_fixed=1e-9)
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-12)
        assert W[0].argmax() in (1, 2)      # nearest of the three to (10, 10)
        assert np.count_nonzero(W[0]) == 1

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            sph_weights(np.zeros((1, 2)), np.ones((3, 2)), h_mode="nope")

    def test_weights_decrease_with_distance(self):
        Y = np.array([[0.0, 0.0], [0.3, 0.0], [0.9, 0.0]])
        W = sph_weights(np.array([[0.0, 0.0]]), Y, h_mode="global")[0]
        assert W[0] > W[1] > W[2]


class TestGramIdentities:
    """The identities that let the published metrics be reproduced without the corpus.

    verify_reference.py relies on these; data/README.md documents them.
    """

    @staticmethod
    def _normalized(rng, n=12, d=40):
        X = rng.uniform(0, 1, (n, d))
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    def test_cosine_dist_matrix_bounds_and_diagonal(self):
        rng = np.random.default_rng(3)
        X = self._normalized(rng)
        D = cosine_dist_matrix(X)
        assert D.shape == (12, 12)
        assert np.all(D >= 0) and np.all(D <= 2)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_loo_cosine_from_gram_matches_direct_computation(self):
        rng = np.random.default_rng(4)
        X = self._normalized(rng)
        G = X @ X.T
        n = len(X)
        for i in range(n):
            mask = np.arange(n) != i
            w = rng.uniform(0, 1, n - 1)
            w /= w.sum()

            v = w @ X[mask]
            direct = float(v @ X[i] / np.linalg.norm(v))

            nv = float(np.sqrt(w @ G[np.ix_(mask, mask)] @ w))
            from_gram = float((w @ G[mask, i]) / nv)

            assert direct == pytest.approx(from_gram, abs=1e-12)

    def test_nearest_neighbour_ranking_from_gram_matches_direct(self):
        rng = np.random.default_rng(5)
        X = self._normalized(rng)
        G = X @ X.T
        n = len(X)
        for i in range(n):
            mask = np.arange(n) != i
            w = rng.uniform(0, 1, n - 1)
            w /= w.sum()
            v = w @ X[mask]
            v = v / np.linalg.norm(v)

            direct_order = np.argsort(X[mask] @ v)[::-1]
            gram_order = np.argsort(G[np.ix_(mask, mask)] @ w)[::-1]
            np.testing.assert_array_equal(direct_order, gram_order)

            assert int(np.argmax(X[mask] @ X[i])) == int(np.argmax(G[mask, i]))


class TestMetrics:
    def test_spearman_perfect_monotone(self):
        x = np.array([0.1, 0.5, 0.2, 0.9, 0.4])
        assert spearman(x, 2 * x + 1) == pytest.approx(1.0, abs=1e-12)
        assert spearman(x, -x) == pytest.approx(-1.0, abs=1e-12)

    def test_spearman_handles_ties(self):
        a = np.array([1.0, 1.0, 2.0, 3.0])
        b = np.array([5.0, 5.0, 6.0, 7.0])
        assert spearman(a, b) == pytest.approx(1.0, abs=1e-12)

    def test_normalized_stress_zero_for_identical_matrices(self):
        rng = np.random.default_rng(6)
        P = rng.uniform(-1, 1, (15, 2))
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        assert normalized_stress(D, D) == pytest.approx(0.0, abs=1e-24)

    def test_knn_preservation_current_behaviour_is_pinned(self):
        """Pins the AS-SHIPPED behaviour so refactors cannot move published numbers.

        NOTE: this value is (k+1)/k, not 1.0 — see KNOWN_ISSUES.md #1. The
        function masks the self-distance to 1e9 AND takes [:k+1], so it compares
        k+1 neighbours while dividing by k. The published knn_preservation
        values were produced with this behaviour and are reproduced by it.
        """
        rng = np.random.default_rng(7)
        P = rng.uniform(-1, 1, (25, 2))
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        for k in (5, 7, 10):
            assert knn_preservation(D, D, k=k) == pytest.approx((k + 1) / k, abs=1e-12)

    @pytest.mark.xfail(strict=True, reason="KNOWN_ISSUES.md #1: off-by-one, "
                                          "not fixed because it would change published values")
    def test_knn_preservation_should_be_one_for_identical_geometry(self):
        """The mathematically correct behaviour. Fails until issue #1 is resolved."""
        rng = np.random.default_rng(7)
        P = rng.uniform(-1, 1, (25, 2))
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        assert knn_preservation(D, D, k=7) == pytest.approx(1.0, abs=1e-12)

    def test_knn_preservation_is_symmetric_in_its_arguments(self):
        rng = np.random.default_rng(9)
        A = np.linalg.norm(rng.uniform(-1, 1, (20, 2))[:, None]
                           - rng.uniform(-1, 1, (20, 2))[None], axis=2)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0.0)
        B = np.linalg.norm(rng.uniform(-1, 1, (20, 2))[:, None]
                           - rng.uniform(-1, 1, (20, 2))[None], axis=2)
        B = (B + B.T) / 2
        np.fill_diagonal(B, 0.0)
        assert knn_preservation(A, B, k=7) == pytest.approx(knn_preservation(B, A, k=7))

    def test_trustworthiness_continuity_is_one_for_identical_geometry(self):
        rng = np.random.default_rng(8)
        P = rng.uniform(-1, 1, (25, 2))
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        t, c = trustworthiness_continuity(D, D, k=7)
        assert t == pytest.approx(1.0, abs=1e-12)
        assert c == pytest.approx(1.0, abs=1e-12)
