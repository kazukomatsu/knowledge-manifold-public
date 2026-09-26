# Changelog

This file starts with the changes after commit `08013cd` (2026-08-18). Earlier
history is in the git log; the defects fixed before publication, and their effect
on the published numbers, are in [`CORRECTIONS.md`](CORRECTIONS.md).

## Unreleased

### Fixed — evidence selection

- `make_evidence.py` could return different evidence packages for the same point:
  representative words and word forms tied on document count were chosen in
  Python's per-process string-hash order, and n-grams with exactly equal scores
  (common: n-grams cut from one word, and all words that occur once in the same
  document, have identical TF-IDF columns) and documents with exactly equal
  weights were ordered by numpy's partition and sort, which differ between numpy
  builds and CPUs — including which tied n-grams entered the 1500-candidate pool.
  The tie rules are now explicit in `code/evidence_lib.py`, and `grid_scan.py`
  uses the same code, so at matched settings the two return identical evidence:

  | what | order |
  |---|---|
  | n-grams | score descending; exactly equal scores by feature index ascending; the candidate pool is cut by the same order, so it is unique |
  | contributing documents | weight descending; exactly equal weights by `doc_id` ascending |
  | representative word | most documents (then shortest, for a merged term), then alphabetical |
  | word forms | most documents, then alphabetical, at most four |
  | source documents | `doc_id` ascending |

  Only exact equality counts as a tie; no tolerance or rounding is introduced.
- `make_evidence.py` failed on a vocabulary smaller than 1500 n-grams. The pool is
  now the smaller of 1500 and the vocabulary; fewer than `--topk` terms are
  returned when the candidates run out; an empty vocabulary, artifacts that
  disagree on the number of documents or n-grams, non-finite scores and
  `--topk` below 1 are errors with a reason. The fixed pools of `13_llm_export.py`
  (600) and of the post-processing consistency check (5) no longer index past a
  small vocabulary; their tie order is unchanged.

### Changed behaviour — evidence can differ from earlier packages

Wherever scores or weights tie exactly, words, word forms, df, source documents
and contributing documents can differ from evidence made before this fix. No
number changes. Measured on the two local journal runs (Python 3.11, numpy 2.4.6),
before (`origin/main` at e0eecc1) and after:

| | Polymer | J. Informetrics |
|---|---|---|
| `grid_scan.py`, 441 points (dx 0.1, 15 + 15 terms): points with any change | 389 | 328 |
| … L2 / L1 term lists changed | 251 / 241 | 139 / 187 |
| … top-10 contributing documents changed | 1 | 5 |
| … CSV and arrays | identical | identical |
| `make_evidence.py`, 81 points (9 x 9, `--topk 15`): terms changed | 58 | 43 |
| … word forms / df or source documents / top-3 documents changed | 81 / 68 / 0 | 81 / 58 / 0 |
| … points where the old version gave different packages for `PYTHONHASHSEED` 0 and 1 | 38 | 52 |
| … the same for the new version | 0 | 0 |

Every changed list was traced to an exact tie: the n-gram candidates of the old
and new order have the same score sequence, or the words differ only within a
tie class, or the document weights are equal. Reproducibility of the evidence
selection (the same scores give the same words and documents in every process
and numpy build) is now guaranteed; numerical reproducibility of the scores
themselves is unchanged and still depends on the pinned environment.

### Added — post-processing tools (PR #1)

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

### Not changed by PR #1

- No pipeline stage, no shipped artifact and no published number. `kmlib.py`,
  `04_fields.py`, `09_manifest.py` and `make_evidence.py` were untouched (the
  evidence fix above then changed `make_evidence.py`, and bounded the candidate
  pool of `13_llm_export.py`; no published number moves either way); the
  co-author's archive carried older copies of the first three without the fixes
  of `CORRECTIONS.md` #3 and #4, and those copies were not taken.
- `code/km_query.py` stays out of the tree (removed in `749708b`): the archive
  holds the same file, still without the `code/data/` bundle it loads.

### Known issues

Tracked, with decisions and next steps, in the table "残件の追跡" of
[`docs/intake_2026-09_coauthor_tools.md`](docs/intake_2026-09_coauthor_tools.md):
the licence of the derived data, the definition of `recluster.py`'s
`top_ngrams`, k recommendation and keyword-driven clustering, the grid scan on
the laboratory corpus, and the tie order still left to numpy in
`13_llm_export.py` and `recluster.py`.

- The global SPH smoothing length h(P) = max distance / 1.98 has a kink on the
  lines x = 0 and y = 0 (the farthest document is a corner anchor and changes
  there). On those lines a central difference tends to the mean of the two
  one-sided derivatives, and a metric built from it is the inner product of that
  mean, which in general differs from the mean of the one-sided metrics.
  `grid_scan.py` flags such points (`metric_kink`); the same applies to the
  metric grid of `04_fields.py grid`. Because h there is set by the corner
  anchors, not by the corpus, the cross-shaped dip it leaves in `H_field` and
  `N_eff` has to be separated from corpus-specific structure when reading them.
