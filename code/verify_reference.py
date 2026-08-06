# -*- coding: utf-8 -*-
"""Reproduce the published E1/E2 numbers from the shipped derived artifacts.

The full-text corpus cannot be redistributed, so ``X_raw.npy`` (the 100 x 250000
char n-gram TF-IDF matrix) is not part of this repository.  Everything that
depends only on *inner products* between documents can nevertheless be
recomputed exactly, because the 100 x 100 Gram matrix ``gram_l2.npy`` is
shipped.

E1 needs only the cosine distance matrix and the coordinates.

E2's leave-one-out reconstruction needs only the Gram matrix as well.  With
``v = sum_j w_j x_j`` over the held-out neighbourhood:

    v . x_i      = sum_j w_j (x_j . x_i) = w . G[mask, i]
    ||v||^2      = sum_jk w_j w_k (x_j . x_k) = w^T G[mask][:, mask] w
    X_mask . v   = G[mask][:, mask] @ w        (up to the positive factor 1/||v||)

so ``loo_cosine_*`` and the nearest-paper recovery rates are exactly
reproducible.  ``topk_recovery_*`` is NOT: it argsorts the 250000-dimensional
feature vectors, which requires the corpus itself.

usage: python3 code/verify_reference.py [--data DIR] [--tol 1e-9]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import (knn_preservation, normalized_stress, spearman, sph_weights,
                   trustworthiness_continuity)

NOT_REPRODUCIBLE = (
    "topk_recovery_mean",
    "topk_recovery_shared_df2_mean",
)


def compute_e1(G, P, kappa, alpha):
    N = len(P)
    # kmlib.cosine_dist_matrix(Xl2) is clip(1 - Xl2 @ Xl2.T); we already have the Gram.
    D_high = np.clip(1.0 - G, 0.0, 2.0)
    D_low = np.linalg.norm(P[:, None] - P[None], axis=2)
    iu = np.triu_indices(N, 1)
    res = {
        "stress_vs_target": normalized_stress(kappa * alpha * D_high, D_low),
        "stress_vs_rawcos_alpha_scaled": normalized_stress(alpha * D_high, D_low),
        "spearman": spearman(D_high[iu], D_low[iu]),
        "pearson": float(np.corrcoef(D_high[iu], D_low[iu])[0, 1]),
    }
    for k in (5, 7, 10):
        t, c = trustworthiness_continuity(D_high, D_low, k=k)
        res[f"trustworthiness_k{k}"] = t
        res[f"continuity_k{k}"] = c
        res[f"knn_preservation_k{k}"] = knn_preservation(D_high, D_low, k=k)
    nn = np.sort(D_low + np.eye(N) * 1e9, axis=1)[:, 0]
    res["nn_dist_cv"] = float(nn.std() / nn.mean())
    return res


def compute_e2(G, P, h_mode):
    N = len(P)
    cos_list, nn_list, nn5_list = [], [], []
    for i in range(N):
        mask = np.arange(N) != i
        w = sph_weights(P[i][None], P[mask], h_mode=h_mode)[0]
        g_col = G[mask, i]                      # X_mask . x_i
        Gmm = G[np.ix_(mask, mask)]             # X_mask . X_mask^T
        nv = float(np.sqrt(max(w @ Gmm @ w, 0.0)))
        cos_list.append(float((w @ g_col) / nv) if nv > 0 else 0.0)
        sim_v = Gmm @ w
        true_nn = int(np.argmax(g_col))
        nn_list.append(int(np.argmax(sim_v) == true_nn))
        nn5_list.append(int(true_nn in np.argsort(sim_v)[::-1][:5]))
    return {
        "h_mode": h_mode,
        "loo_cosine_mean": float(np.mean(cos_list)),
        "loo_cosine_std": float(np.std(cos_list)),
        "loo_cosine_min": float(np.min(cos_list)),
        "loo_cosine_max": float(np.max(cos_list)),
        "loo_cosine_median": float(np.median(cos_list)),
        "nearest_paper_recovery_rate": float(np.mean(nn_list)),
        "nearest_paper_in_top5_rate": float(np.mean(nn5_list)),
    }


def compare(label, got, ref, tol, skip=()):
    rows, ok = [], True
    for key, value in got.items():
        if not isinstance(value, (int, float)) or key in skip:
            continue
        if key not in ref:
            continue
        delta = abs(value - ref[key])
        scale = max(abs(ref[key]), 1.0)
        passed = delta <= tol * scale
        ok &= passed
        rows.append((key, value, ref[key], delta, passed))
    print(f"\n=== {label} ===")
    print(f"  {'metric':34s} {'recomputed':>13s} {'published':>13s} {'|diff|':>10s}   ")
    for key, value, refv, delta, passed in rows:
        print(f"  {key:34s} {value:13.6f} {refv:13.6f} {delta:10.2e}   "
              f"{'PASS' if passed else 'FAIL'}")
    return ok, len(rows)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "data", "derived"),
                    help="directory holding gram_l2.npy / coords.npy / reference JSONs")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="relative tolerance for the comparison (default 1e-9)")
    args = ap.parse_args()

    G = np.load(os.path.join(args.data, "gram_l2.npy"))
    P = np.load(os.path.join(args.data, "coords.npy"))
    audit = json.load(open(os.path.join(args.data, "map_audit.json")))
    print(f"data      : {args.data}")
    print(f"documents : {len(P)}   tolerance: {args.tol:g} (relative)")
    print(f"anchors   : {audit['anchor_docs']}")

    all_ok, total = True, 0

    ref = json.load(open(os.path.join(args.data, "e1_distance_preservation.json")))
    ok, n = compare("E1 distance preservation", compute_e1(G, P, audit["kappa"], audit["alpha"]),
                    ref, args.tol)
    all_ok &= ok
    total += n

    for h_mode in ("global", "knn_adaptive"):
        path = os.path.join(args.data, f"e2_loo_{h_mode}.json")
        if not os.path.exists(path):
            continue
        ok, n = compare(f"E2 leave-one-out ({h_mode})", compute_e2(G, P, h_mode),
                        json.load(open(path)), args.tol)
        all_ok &= ok
        total += n

    print(f"\nnot recomputable without the corpus (requires X_raw.npy): "
          f"{', '.join(NOT_REPRODUCIBLE)}")
    print(f"\n{'ALL ' + str(total) + ' METRICS REPRODUCED' if all_ok else 'MISMATCH DETECTED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
