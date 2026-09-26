# Changelog

This file starts with the changes after commit `08013cd` (2026-08-18). Earlier
history is in the git log; the defects fixed before publication, and their effect
on the published numbers, are in [`CORRECTIONS.md`](CORRECTIONS.md).

## Unreleased

### Added

- Post-processing tools for a finished run, adapted from scripts contributed by a
  co-author in September 2026 (provenance, archive hashes and a file-by-file
  adoption record: [`docs/intake_2026-09_coauthor_tools.md`](docs/intake_2026-09_coauthor_tools.md);
  guide: [`docs/postprocessing_ja.md`](docs/postprocessing_ja.md)):
  - `code/add_cluster_to_csv.py` — writes a new coordinate CSV with a cluster
    column, joined by `doc_id` and checked for duplicate, missing and
    out-of-range ids and for a label count that does not match.
  - `code/recluster.py` — re-clusters a finished map at any k with the pipeline's
    own definition (k-means on the ten leading SVD scores, seed 0), numbering
    clusters by size; `--no-terms` runs on the shipped `data/derived`.
  - `code/grid_scan.py` — scans every point of a regular grid on [-1,1]²
    (default 441 points; `--dx 0.05` gives 1681) for gaps, cluster mixing,
    support, GPR uncertainty, metric-tensor eigenvalues, principal directions
    and directional gradients, with the lens words of the evidence package.
  - `code/grid_viz.py` — heat maps and a provisional candidate list from a scan.
  - `code/postproc_lib.py` — the input checks the four tools share, including a
    check that the artifacts handed to a tool belong to one run (coordinates,
    manifest hashes, GPR, SVD, feature matrix, text and labels chained together).
- `tests/test_postprocessing.py` and `tests/synthetic_run.py`: a synthetic run built from
  invented words, plus the shipped `data/derived` (no corpus text).
- `CHANGELOG.md`.

### Not changed

- No pipeline stage, no shipped artifact and no published number. `kmlib.py`,
  `04_fields.py`, `09_manifest.py` and `make_evidence.py` are untouched; the
  co-author's archive carried older copies of the first three without the fixes
  of `CORRECTIONS.md` #3 and #4, and those copies were not taken.
- `code/km_query.py` stays out of the tree (removed in `749708b`): the archive
  holds the same file, still without the `code/data/` bundle it loads.

### Known issues found while doing this (not fixed)

- `make_evidence.py` picks between representative words that tie on document
  frequency and length in Python's per-process string-hash order, so its
  `word_forms`, and occasionally a term, can change with `PYTHONHASHSEED`.
  `grid_scan.py` orders the words and is not affected.
- The order of character n-grams with exactly equal scores — common, because
  n-grams cut from the same word have identical TF-IDF columns — is left to
  numpy's partition and sort, so the lens words of `make_evidence.py` and
  `grid_scan.py` can differ between numpy versions or CPUs while every number
  agrees. Measured on the co-author's Polymer example: numpy 1.26.4 reproduces it
  exactly, numpy 2.4.6 differs in the concentration lens at 154 of 441 points.
  Documents at exactly equal weight (anchors equidistant from a grid point) are
  ordered the same way in the contributing-document lists.
- `make_evidence.py` fails on a vocabulary smaller than 1500 n-grams.
- The global SPH smoothing length h(P) = max distance / 1.98 has a kink on the
  lines x = 0 and y = 0 (the farthest document is a corner anchor and changes
  there). On those lines a central difference tends to the mean of the two
  one-sided derivatives, and a metric built from it is the inner product of that
  mean, which in general differs from the mean of the one-sided metrics.
  `grid_scan.py` flags such points (`metric_kink`); the same applies to the
  metric grid of `04_fields.py grid`. Because h there is set by the corner
  anchors, not by the corpus, the cross-shaped dip it leaves in `H_field` and
  `N_eff` has to be separated from corpus-specific structure when reading them.
