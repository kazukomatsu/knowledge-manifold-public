# Known issues

Issues found while preparing this repository for release. Each is reproducible
and covered by a test. Neither is fixed here, because fixing them changes
numbers that appear in the manuscript — that is the authors' call, not a
packaging decision.

---

## 1. `knn_preservation` compares k+1 neighbours but divides by k

**Status:** open. Affects published values.

`kmlib.knn_preservation` masks the self-distance to `1e9` *and* takes the first
`k+1` entries of the argsort:

```python
nh = set(np.argsort(D_high[i] + np.eye(N)[i] * 1e9)[:k + 1]) - {i}
```

Because the self-distance is already pushed to last place, `- {i}` removes
nothing, so `nh` holds `k+1` neighbours while the score is normalised by `k`.
The result can exceed 1: for identical high- and low-dimensional geometry the
function returns `(k+1)/k` instead of `1.0`.

Measured on the shipped 100-document artifacts:

| k | published (as-shipped) | corrected `[:k]` |
|---|---|---|
| 5 | 0.492 | 0.388 |
| 7 | **0.504** | **0.427** |
| 10 | 0.513 | 0.463 |

Identical-geometry sanity check: as-shipped returns 1.200 / 1.143 / 1.100 for
k = 5 / 7 / 10; corrected returns 1.000 for all three.

**What this does and does not affect.** Every method in
`data/derived/e10_embedding_comparison.json` (v5.0, PCA, MDS, t-SNE, UMAP) is
scored with the same function, so the *comparison and the ranking are
unaffected* — only the absolute values are inflated. Trustworthiness,
continuity, Spearman, stress and all E2 metrics are computed by separate code
paths and are unaffected.

**Fix, when the authors decide to take it:** drop the `+ 1` and the `- {i}`
(the mask already excludes self), then regenerate the reference artifacts and
update the manuscript's kNN-preservation values. Tests
`test_knn_preservation_current_behaviour_is_pinned` and
`test_knn_preservation_should_be_one_for_identical_geometry` (currently
`xfail(strict=True)`) switch over together.

---

## 2. Anchor assignment had a D4 gauge freedom — fixed, but it re-orients the map

**Status:** fixed in `kmlib.assign_anchors`. Changes figure orientation relative
to previously circulated runs.

The nine anchor positions (four corners, four edge midpoints, centre) are
symmetric under the dihedral group of the square, so the exhaustive assignment
search in stage 2 has **exactly 8 optimal solutions**, degenerate to within
~5e-15 (0–3 ulp). The original code selected with a bare `cost < best`
comparison, letting floating-point rounding pick the winner. Upstream, the
sparse `X @ X.T` Gram product differs by ~5e-9 relative between BLAS/SciPy
builds, which is enough to flip the choice.

Consequence: the same corpus produced **mirror-image maps** on Python 3.9 vs
3.11 (`mean |A - flipX(B)| = 0.08`). Every pairwise-distance metric is invariant
under the mirror, so all 15 validation gates passed and E1/E2 matched — the
defect was invisible to the existing checks. What broke was anything addressed
by absolute coordinates: for a fixed query point `(x, y)`, the top-8
contributing documents agreed only 15.4% on average across the two
environments (median 0%, measured over a 19x19 grid).

The fix isolates the tied orbit with a relative tolerance and fixes the gauge by
taking the lexicographically smallest tuple of document indices. Margins are
wide: the tie spread is ~5e-15, cross-environment noise ~5e-9 relative, and the
gap to the next distinct cost level is 5.6e-3 relative — six orders of
magnitude either side of the 1e-9 threshold. After the fix the two environments
agree on the assignment exactly, and contributing-document agreement rises to
74.7%.

**Consequence for the manuscript:** the canonical gauge chosen by the
lexicographic rule is *a* valid one, not necessarily the one the previously
circulated figures used. All eight are equivalent up to reflection/rotation, so
no metric changes, but figures regenerated after the fix may be mirrored
relative to the earlier drafts. `map_audit.json` now records
`anchor_assignment_tie_count` and `anchor_assignment_tie_break` so the gauge is
auditable.

Residual, unrelated to the gauge: free-point coordinates still differ by up to
0.35 (mean 0.08) between Python 3.9 and 3.11 because the SMACOF initialisation
and L-BFGS-B land on different local optima of the same objective (J = 0.16435
vs 0.16385). Pinning the environment as `requirements.txt` specifies avoids
this; Python 3.9 is not supported.
