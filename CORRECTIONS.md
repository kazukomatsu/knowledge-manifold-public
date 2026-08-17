# Corrections

Three defects were found while preparing this repository for release. **All are
fixed here.** The first two are documented with the measurement that exposed
them, because one changes a value that appears in the manuscript and the other
changes the orientation of every figure. The third is a provenance string that
misdescribed how the code works; no value depends on it.

Defects 1 and 2 had both passed all 15 validation gates undetected. That is the
common lesson:
the gates check properties of pairwise distances and never checked either the
uniqueness of absolute coordinates or the range of a metric that is by
definition a fraction.

---

## 1. `knn_preservation` compared k+1 neighbours but divided by k

**Fixed. Changes published values — the manuscript must be updated.**

### The defect

Self-exclusion was implemented twice over. The diagonal was masked to `1e9`
*and* the slice took `k+1` entries with `- {i}` applied afterwards:

```python
nh = set(np.argsort(D_high[i] + np.eye(N)[i] * 1e9)[:k + 1]) - {i}
nl = set(np.argsort(D_low[i]  + np.eye(N)[i] * 1e9)[:k + 1]) - {i}
p += len(nh & nl) / k
```

Either mechanism alone is correct. `- {i}` after `[:k+1]` works when the
self-distance is 0 and therefore sorts first. The `1e9` mask works with `[:k]`.
Applied together, the mask has already pushed self to rank `N-1`, so `[:k+1]`
holds `k+1` genuine other documents and `- {i}` removes nothing. The sets have
`k+1` members, the intersection can reach `k+1`, and the score divides by `k`.

The tell-tale: for identical high- and low-dimensional geometry the function
returned `(k+1)/k` — 1.200, 1.143 and 1.100 for k = 5, 7, 10. A preservation
*fraction* above 1 is impossible.

### The fix

```python
nh = set(np.argsort(D_high[i] + np.eye(N)[i] * 1e9)[:k])
nl = set(np.argsort(D_low[i]  + np.eye(N)[i] * 1e9)[:k])
```

Identical geometry now returns exactly 1.000 for all k.

### Effect on reported values

E1, measured on the 100-document corpus:

| k | before (earlier drafts) | after (shipped) |
|---|---|---|
| 5 | 0.492 | **0.388** |
| 7 | **0.504** | **0.427** |
| 10 | 0.513 | **0.463** |

This is **not** a `k/(k+1)` rescale — `0.504 x 7/8 = 0.441`, not 0.427. The extra
neighbour changes *which* documents fall inside each set, not merely the
denominator, so the earlier numbers cannot be corrected arithmetically. They had
to be recomputed.

E10 method comparison, all five methods recomputed:

| method | before | after | rank before | rank after |
|---|---|---|---|---|
| t-SNE | 0.704 | 0.621 | 1 | 1 |
| UMAP | 0.676 | 0.587 | 2 | 2 |
| v5.0 (this work) | 0.504 | 0.427 | 3 | 3 |
| MDS | 0.411 | 0.361 | 4 | 4 |
| PCA | 0.371 | 0.301 | 5 | 5 |

Every method was scored by the same function, so **the ranking is unchanged**;
only the absolute values move. The E7 stability tables in
`report_*.md` also carry `knn_preservation_k7` and shifted accordingly.

### Not affected

`trustworthiness_continuity` uses a different and correct construction: it builds
rank matrices after the same `1e9` mask and then selects `np.where(rl[i] < k)`,
which yields exactly `k` neighbours because self has rank `N-1`. Trustworthiness,
continuity, Spearman, Pearson, stress and every E2 metric are computed on
separate code paths and are unchanged.

### Covered by

`test_knn_preservation_is_one_for_identical_geometry`,
`test_knn_preservation_never_exceeds_one`,
`test_knn_preservation_counts_exactly_k_neighbours`.

---

## 2. The anchor assignment had a D4 gauge freedom

**Fixed. No published value changes, but figures are mirrored relative to
earlier drafts.**

### The defect

Stage 2 picks nine representative documents, then searches all 9! = 362,880
assignments of those documents to the nine fixed anchor positions (four corners,
four edge midpoints, centre), minimising

```
cost(s) = sum_{i<j} ( A2D_ij - alpha(s) * d_{s(i)s(j)} )^2
```

That layout is invariant under the **dihedral group of the square, D4** (order 8:
identity, three rotations, two axis mirrors, two diagonal mirrors). Relabelling
positions by any element of D4 leaves `A2D` unchanged, so `cost(s o g) = cost(s)`
holds *exactly*, and the optimum is **8-fold degenerate**. The original code
selected with a bare comparison:

```python
if cost < best[0]:
    best = (cost, perm, alpha)
```

which hands the choice among the eight to floating-point rounding.

### Why the choice actually moved

Measured on the 100-document corpus:

```
cost magnitude               11.19
spread across the 8 optima    5.3e-15   (0-3 ulp — mathematically zero)
gap to the next cost level    6.3e-2    (relative 5.6e-3)
```

`cost` depends on `D = 1 - Gram`, and `Gram = Xl2 @ Xl2.T` is a product over a
100 x 250000 matrix whose accumulation order depends on the BLAS build:

```
cost_min, Python 3.9    11.192498032585483
cost_min, Python 3.11   11.192497975305898
difference               5.7e-8 absolute / 5.1e-9 relative
```

That noise is roughly **ten million times** the degeneracy spread, so it
reshuffles which of the eight rounds lowest. Even the number of exact ties
differed: four of the eight matched `cost_min` bit-for-bit on 3.9, two on 3.11.

### What it did

```
Python 3.9    anchor_docs = [ 5, 44, 48, 84, 25, 51, 26, 94, 64]
Python 3.11   anchor_docs = [44,  5, 84, 48, 51, 25, 26, 94, 64]
```

Document 5 sat at `(-1,-1)` under 3.9 and `(+1,-1)` under 3.11; 44 the reverse;
48 and 84, 25 and 51 likewise swapped. The fixed points of an x-mirror —
document 26 at `(0,1)`, 94 at `(0,-1)`, 64 at `(0,0)` — stayed put. Across all
100 documents `mean |A - flipX(B)| = 0.080`, while all seven other D4 transforms
gave a maximum deviation of 2.0. The same corpus produced **mirror-image maps**.

### Why validation could not see it

E1, E2, the geodesics and all 15 gates are functions of pairwise distances only.
Reflection is an isometry, so every one of them is exactly invariant. Both
environments returned 15/15 PASS and both reproduced the paper's E1/E2 values to
three decimals.

What broke was anything addressed by absolute coordinates. Over a 19 x 19 grid
of query points, the top-8 contributing-document sets agreed:

| | mean | median | points below 50% | identical |
|---|---|---|---|---|
| before | **0.154** | **0.000** | 81.7% | 0.3% |
| after | 0.747 | 0.750 | 0.6% | 5.0% |

Concretely, the worked example in the usage guide did not reproduce in either
environment: at `(-0.5, 0.0)` Python 3.11 reported a curing-reaction
neighbourhood (`cure-shrinkage`, `dgeba`, `gelation`) and Python 3.9 a damage-
mechanics one (`cfrtp`, `kink-band`, `paek`), while the draft that the guide was
written against showed a third region again.

This bears directly on the claimed advantages over t-SNE and UMAP recorded in
`data/derived/e10_embedding_comparison.json` — "固定枠(アンカー9点): 実行間・
コーパス間で座標系が比較可能" and "決定論性: seed0で座標が一意に再現". Determinism
held within one environment; the frame itself carried an unfixed reflection.

### The fix

`kmlib.assign_anchors()` collects all 362,880 candidates, isolates the tied set
with a threshold of `1e-9 * |cost_min| + 1e-12`, and fixes the gauge by taking
the **lexicographically smallest tuple of document indices**. The threshold sits
about six orders of magnitude above the degeneracy spread (5e-15) and six below
the gap to the next distinct cost level (5.6e-3), with the cross-environment
noise (5e-9) comfortably inside.

Verified: both environments now select `[5, 44, 48, 84, 25, 51, 26, 94, 64]`;
`map_audit.json` records `anchor_assignment_tie_count` (8) and the tie-break rule
so the gauge is auditable; no E1/E2/E10 numeric field changed by more than 0
against the pre-fix run; the new map is the exact mirror of the old one
(`max deviation 2.2e-16`).

**Consequence for the manuscript:** the canonical gauge is *a* valid member of
the orbit, not necessarily the one the earlier figures used. No metric changes,
but figures regenerated after the fix are re-oriented relative to pre-fix
drafts. `data/reference_figures/` holds the post-fix versions.

**The submitted manuscript is not affected.** In the version of 2026-08-17,
Fig. 1 places all nine anchor documents exactly where `data/derived/coords.npy`
puts them — 5 bottom-left, 44 bottom-right, 48 top-right, 84 top-left, 25 and 51
on the x-edge midpoints, 26 and 94 on the y-edge midpoints, 64 at the centre
(checked against the figure's own vector label coordinates). The paper and this
repository share one gauge, so no re-orientation statement applies between them.

Two comparisons do remain, and they land on different elements of D4, so keep
them apart. Against the **pre-fix run in this same environment** the canonical
map is the left-right mirror measured above (`flipX`, deviation 0.080 over the
100 documents; the canonical assignment is the one Python 3.9 produced). Against
the **earlier drafts** it is the diagonal mirror (x <-> y), which is what the
update note in `docs/submission_bundle_README.md` records.

### Residual, and why Python 3.11 is pinned

Fixing the gauge does not make coordinates identical across environments — that
is why the agreement above is 0.747 and not 1.0. The SMACOF initialisation and
the L-BFGS-B descent land on different local optima of the same objective
(Python 3.9: 42 iterations, J = 0.164346; Python 3.11: 20 iterations,
J = 0.163845), leaving free points up to 0.35 apart (mean 0.08). There is no
gauge fix for that; the environment has to be pinned, which is what
`requirements.txt` does.

### Covered by

`TestAnchorGaugeFixing` — six tests, including one that perturbs the distance
matrix by 1e-14 twelve times and asserts the assignment never moves.

---

## 3. The recorded provenance misdescribed two optimisers

**Fixed. No value changes; the shipped `manifest.json` is edited in place.**

`manifest.json` carried two claims that the code contradicts:

| field | recorded | actually used |
|---|---|---|
| `gpr.implementation` | custom NumPy GPR (sklearn unavailable); hyperparams by Nelder-Mead multi-restart | `sklearn.gaussian_process.GaussianProcessRegressor(n_restarts_optimizer=10, random_state=0)` in `kmlib.GPR.fit`; the posterior is then recomputed in NumPy from the fitted hyperparameters |
| `geodesic.optimizer` | custom L-BFGS (two-loop, Armijo backtracking; scipy L-BFGS-B unavailable) | `scipy.optimize.minimize(method="L-BFGS-B")` through `kmlib.lbfgs`, called at `05_geodesics.py:134` |

Both are leftovers from an earlier stage of the work, when those libraries really
were unavailable, and both were already contradicted inside the same file:
`environment.note` and `deviations_from_spec` state that the GPR is sklearn's and
the optimiser scipy's, and `map_audit.json` records the map optimiser correctly as
`scipy L-BFGS-B`. `kmlib.nelder_mead()` survives from that period and is now
called by nothing.

No number moves: the fields are descriptive, and no code reads them —
`validate.py` reads `n_restarts`, `length_scale_x` and `white_noise` from
`gpr_info.json`, never `implementation`. The sources are fixed too
(`04_fields.py` writes `gpr_info.json`, `09_manifest.py` writes `manifest.json`),
so regenerated runs record the corrected text.
