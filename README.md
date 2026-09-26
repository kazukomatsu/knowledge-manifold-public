# Knowledge Manifold pipeline

A deterministic pipeline that embeds a document corpus into a **fixed-frame,
bounded 2-D knowledge manifold**, builds continuous fields over it (SPH density
and entropy, Gaussian-process mean and uncertainty), traces geodesics under
several metrics, and exports machine-computed **evidence packages** so that a
language model can describe a location on the map without being the thing that
decided what is there.

The pipeline itself calls no language model. `13_llm_export.py` and
`make_evidence.py` only write JSON; nothing in `code/` opens a network
connection. The division of labour is deliberate: the geometry, the
characteristic terms, the contributing documents and the uncertainty are all
computed deterministically, and verbalization is a separate, auditable step
governed by `docs/verbalization_protocol.md`.

This code accompanies a manuscript submitted to *Journal of Informetrics*. See
[`CITATION.cff`](CITATION.cff).

> **Read [`CORRECTIONS.md`](CORRECTIONS.md) before comparing against earlier
> drafts.** Four defects found while preparing and publishing this release are
> fixed here: one changes the reported kNN-preservation values (k=7:
> 0.504 → 0.427), one re-orients every figure relative to pre-fix drafts, one
> corrected a provenance string that misdescribed two optimisers, and one left
> two rank metrics at the mercy of an unstable sort, which moved `continuity_k10`
> by 1.2e-5 between CPU architectures. No published metric other than kNN
> preservation moves. The manuscript of 2026-08-17 already uses the
> post-fix gauge — its Fig. 1 places all nine anchor documents exactly where
> `data/derived/coords.npy` does.

## Requirements

**Python 3.11**, with the exact dependency versions in `requirements.txt`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional: adds the UMAP baseline
```

3.11 is the reference environment, not a lower bound. The map is sensitive to the
linear-algebra stack — the sparse Gram product `X @ X.T` varies by ~5e-9 relative
between BLAS builds, which is enough to change the anchor assignment
(`CORRECTIONS.md` #2) — so the dependency versions are pinned exactly to the
ones that produced the shipped artifacts, and 3.11.15 is the only interpreter on
which the published numbers were verified end to end. The code imports and its
unit tests pass on newer Pythons, but reproduction of the paper's values is not
claimed there, and `umap-learn` cannot be built on 3.14 at all.

### Python 3.10

The pinned versions do not resolve on 3.10 — `numpy==2.4.6` requires 3.11 or
newer, and the newest stack 3.10 accepts is numpy 2.2.6 / scipy 1.15.3 /
scikit-learn 1.7.2 / matplotlib 3.10.9. On that unpinned stack (verified
2026-08-18, macOS arm64, and `umap-learn==0.5.12` installs there too):

```bash
python3.10 -m venv .venv310 && source .venv310/bin/activate
pip install numpy scipy scikit-learn matplotlib pytest   # no pinning possible
python3 -m pytest tests/ -v            # 118 passed (2026-09-26; 35 before the post-processing tests)
python3 code/verify_reference.py       # ALL 28 METRICS REPRODUCED
```

So the corpus-free verification path is available on 3.10. It is insensitive to
the stack by construction: `verify_reference.py` reads the shipped Gram matrix
rather than recomputing `X @ X.T`, which is the step the pinning exists to
protect. What 3.10 cannot give you is a *rebuild* of the map from a corpus that
lands on the published coordinates; that needs the pinned 3.11.15 environment.

Do not run this on Python 3.9. It completes and still passes all 15 validation
gates, but the anchor assignment resolves differently and E1 drifts (Spearman
0.676 instead of 0.687).

## Reproducing the published results without the corpus

The corpus is 100 published papers on carbon-fibre composites. Their full text
is **not** in this repository and cannot be — it consists of publisher PDFs. What
ships instead is the derived representation, which is enough to recompute most
of the published numbers exactly:

```bash
python3 code/verify_reference.py
```

```
=== E1 distance preservation ===
  spearman                                0.686767      0.686767   0.00e+00   PASS
  knn_preservation_k7                     0.427143      0.427143   0.00e+00   PASS
  ...
ALL 28 METRICS REPRODUCED
```

This works because every E1 metric — and E2's `loo_cosine_*` and
`nearest_paper_*` rates — depends on the documents only through their inner
products, and the 100 x 100 Gram matrix is shipped. `data/README.md` gives the
identities and states precisely what is and is not reproducible this way
(`topk_recovery_*` is not: it needs the 250000-dimensional feature vectors).

`data/corpus_manifest.csv` lists all 100 DOIs with their map coordinates and
cluster, so each point in the published figures can be traced to a paper.

## Running the full pipeline on your own corpus

Stage 0 expects a single concatenated Markdown file, one document per section.

```bash
# 1. Expand the corpus into the layout the pipeline reads
python3 code/make_derived_input.py \
    --input ./your_corpus.md \
    --output inputs/derived/mycorpus

# 2. Run all stages. Relative paths are resolved against the directory you
#    call from, so either form works.
bash code/run_v50.sh \
    --derived-input inputs/derived/mycorpus \
    --run-id mycorpus \
    --outputs-root outputs

# 3. Validation gates (run_v50.sh runs these too and stops on failure)
KM_OUT="$PWD/outputs/mycorpus" \
KM_DATA="$PWD/outputs/mycorpus/work" \
python3 code/validate.py
```

`validate.py` must pass before any number is reported; the 15 gates and their
thresholds are tabulated in Table S1 of the supplementary material.

Expect roughly 12–20 minutes on one core; the 34 geodesics dominate. `--quick`
skips the E7/E8 stability and sensitivity sweeps, which is most of the time —
but some validation gates then fail, because they read E8 products. `--fullfig`
adds three auxiliary figures not used in the paper.

Nine figures land in `outputs/<run-id>/figures/`. Intermediate products (Gram
matrix, GPR model, vocabulary, paths) stay in `outputs/<run-id>/work/` and are
needed by the evidence-package step, so keep them.

### Verbalizing a location

```bash
python3 code/make_evidence.py --x -0.5 --y 0.0 \
    --out outputs/mycorpus --data outputs/mycorpus/work --code code
```

The readout smoothing length follows the selection protocol of the manuscript
(Sec. 3.3): `kmlib.select_readout()` scores both rules by leave-one-out
reconstruction — `e2_loo_global.json` against `e2_loo_knn_adaptive.json`, both
written by stage 5 — and adopts the better one. Only the readout tier is
selected; the geometry always uses the global rule. On the laboratory corpus the
selection lands on adaptive (LOO cosine 0.4708 against 0.4663), and on both
journal corpora of E9 on global (Table 5 of the manuscript).
`--readout global|adaptive` forces the choice, `13_llm_export.py` prints the tier
it selected, and with neither LOO file present the adaptive rule is used.

This prints the location's uncertainty, mixture entropy and both term lenses,
and writes `evidence_point_-0.5_0.0.json`. Hand that file **together with**
`docs/verbalization_protocol.md` to a language model. Passing the evidence
without the protocol destroys auditability: there is then no way to separate
what the model read off the evidence from what it supplied out of general
knowledge.

Terms in the package are whole words, deterministically reconstructed from the
corpus text rather than raw character n-grams, so `compton` arrives as a word
and not as the fragment `ompto`.

### Post-processing a finished run

Four tools work on a completed run without re-running the pipeline or touching
any of its artifacts; none calls a language model. They were contributed by a
co-author and adapted here (provenance and adoption decisions:
[`docs/intake_2026-09_coauthor_tools.md`](docs/intake_2026-09_coauthor_tools.md);
full guide in Japanese: [`docs/postprocessing_ja.md`](docs/postprocessing_ja.md)).

```bash
# cluster column joined onto the map coordinates (by doc_id) -> coordinates_2d_clusters.csv
python3 code/add_cluster_to_csv.py --run outputs/mycorpus

# the same fixed map clustered at other k -> outputs/mycorpus/recluster/
python3 code/recluster.py --out outputs/mycorpus --k 4 6 7

# every grid point of [-1,1]^2: gaps, cluster mixing, support, GPR uncertainty,
# metric tensor eigen-analysis, lens words -> outputs/mycorpus/grid_scan/k5_dx0.1_l2-15_l1-15/
python3 code/grid_scan.py --out outputs/mycorpus --topk-l2 15 --topk-l1 15
# 41 x 41 = 1681 points -> outputs/mycorpus/grid_scan/k5_dx0.05_l2-15_l1-15/
python3 code/grid_scan.py --out outputs/mycorpus --dx 0.05 --topk-l2 15 --topk-l1 15

# heat maps and a provisional candidate list from a scan
python3 code/grid_viz.py --scan outputs/mycorpus/grid_scan/k5_dx0.1_l2-15_l1-15 --out outputs/mycorpus
```

`recluster.py` uses the pipeline's own definition (k-means on the ten leading
SVD scores, seed 0) and numbers clusters by size; at k=5 it returns the standard
partition renumbered, and with `--no-terms` it runs on the shipped
`data/derived` alone (give `--outdir`: nothing is written inside `data/`).
Cluster numbers are specific to a run and a k.

All four tools refuse to overwrite earlier output without `--force` and list
every missing input instead of looking for it elsewhere. `grid_scan.py` and
`recluster.py` check that the artifacts they are given belong to one run —
coordinates against `coordinates_2d.csv` and the hashes in `manifest.json`, the
GPR's training points and scale, the SVD and the feature matrix against the Gram
matrix, each document's top n-grams against its text, the metadata's character
counts against the text, and (for the scan) the cluster labels against the run's
own k-means partition, with `--custom-labels` to use other labels deliberately —
and record what could be checked in their metadata. `add_cluster_to_csv.py`
checks coordinates and labels against the run when given `--run` and prints the
result; `grid_viz.py` checks the coordinates and labels it overlays against the
hashes the scan recorded.

`grid_scan.py` keeps the co-author's definitions of every quantity. It records
the grid, settings, tolerances, input hashes and library versions in
`k5_grid_scan_meta.json` and keeps timing in a separate `*_runinfo.json`; rerun
in another process, its CSV, details and meta are byte-identical and its arrays
equal. Three flags mark where a number should not be
over-read: `lam2_clipped` (a rounding-level negative eigenvalue set to 0),
`e1_ill_defined` (near-isotropic metric, no principal direction) and
`metric_kink` (the lines x = 0 and y = 0, where the global SPH smoothing length
h(P) = max distance / 1.98 has a kink, so the finite-difference metric is an
average of one-sided derivatives). The `verbalization_50w` column is left empty:
prose is written afterwards, outside the pipeline, by handing
`make_evidence.py` output for a chosen coordinate to a model together with
`docs/verbalization_protocol.md`. The scan's fields all use the global rule, so
compare its lens words with `make_evidence.py --readout global --topk 15`. The
candidate list is a screening under provisional percentile thresholds, not an
optimum, a finding or a Discovery Score.

## Repository layout

```
code/                 the pipeline; numbered scripts run in order, kmlib.py is shared
  run_v50.sh          driver for all stages
  validate.py         15 validation gates
  verify_reference.py recompute the published metrics from data/derived/
  make_derived_input.py     package a concatenated Markdown corpus for stage 0
  subset_derived_input.py   cut N documents out of a packaged corpus (the N=100/200/400 probe)
  add_cluster_to_csv.py     post-processing: cluster column on the coordinate CSV
  recluster.py              post-processing: re-cluster a finished run at other k
  grid_scan.py, grid_viz.py post-processing: full grid scan of a finished map, heat maps
  postproc_lib.py           shared input checks for the four post-processing tools
data/
  corpus_manifest.csv 100 DOIs, titles, years, clusters, map coordinates
  derived/            Gram matrix, coordinates, SVD scores, reference metric JSONs
  reference_figures/  the nine published figures
docs/
  USAGE_ja.md         detailed walkthrough (Japanese)
  verbalization_protocol.md      the binding protocol for the verbalization step
  example_evidence_point.json    what make_evidence.py produces, for reference
  postprocessing_ja.md           guide to the post-processing tools (Japanese)
  intake_2026-09_coauthor_tools.md  what was taken from the co-author's archives, and why
tests/                pytest suite; synthetic data only, no corpus text
```

Script names carry historical version numbers (`run_v50.sh`, `12_map_v30.py`)
that no longer match the spec version they implement, which is v5.2
(`KAPPA=0.80` and the other v5.2 values). They were left alone so that existing
artifacts and log files stay traceable.

## Correspondence with the manuscript

The manuscript numbers its evaluations in order of presentation; the file names
carry the internal numbering the suite was developed under. The two do not
agree — E1 and E2 are transposed, and the paper's E8 is the internal `e7_*`.

| Paper | Content | Artifacts | Here? |
|---|---|---|---|
| E1 | leave-one-out prediction | `e2_loo_global.json`, `e2_loo_knn_adaptive.json` | shipped in `data/derived/` |
| E2 | distance preservation | `e1_distance_preservation.json` | shipped in `data/derived/` |
| E3 | embedding baselines | `e10_embedding_comparison.json` | shipped in `data/derived/` |
| E4 | geodesic family | `geodesic_results.csv`, `e4_lambda_path_deviation.json`, `path_*.npy` | produced by a run |
| E5 | lenses and readout | `e9_verbalization_L1_L2.json` | produced by a run; the two-tier appendix analysis is not |
| E6 | evidence budget and vocabulary | — | bundle only |
| E7 | cross-model verbalization | — | bundle only: the six verbatim model responses |
| E8 | robustness suite | `e7_*.json`, `r1_N*.json` | produced by a run, skipped under `--quick` |
| E9 | independent journal corpora | — | bundle only: this same pipeline on two further corpora of 100 papers each, sampled from *Polymer* and the *Journal of Informetrics* (2020–2026, open access) |

This repository is the code-and-derived-data release. The submission bundle
additionally carries `config/`, the full `results/` tree (metrics, coordinates,
paths, figures, plot data, appendix analyses), `evidence/` (the packages handed
to the language models and their six verbatim responses),
`results_external/` (the journal-corpus replication) and the supplementary
material as submitted. That bundle's own README is kept here as
[`docs/submission_bundle_README.md`](docs/submission_bundle_README.md).

## Tests

```bash
pip install pytest && python3 -m pytest tests/ -v
```

Unit coverage of the metric functions and the SPH kernel, regression tests for
both corrections in `CORRECTIONS.md` (including one that perturbs the distance
matrix by 1e-14 twelve times and asserts the anchor assignment never moves),
verification of the Gram-matrix identities that `verify_reference.py` relies on,
and integration tests that the shipped artifacts still reproduce the shipped
reference values and that no corpus-derived file has crept into `data/`.

`tests/test_postprocessing.py` covers the post-processing tools on a complete
but tiny run built from invented words (`tests/synthetic_run.py`, same TF-IDF,
clustering and GPR code as the pipeline): doc_id joins and their failure modes,
re-clustering at several k with the inputs left byte-identical, the published
k=5 partition recovered from `data/derived`, the 441- and 1681-point grids
one-to-one across CSV, details and arrays, the metric-tensor identities and a
finite-distance check, the kink flag, byte-identical reruns under different
`PYTHONHASHSEED`, and agreement with `make_evidence.py` at matched settings.

CI runs the suite on Python 3.11 and additionally asserts that no corpus-derived
file has been committed.

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Kazuhiko Komatsu, Yoshiaki
Kawagoe, Shigeru Obayashi and Tomonaga Okabe, the authors of the accompanying
manuscript.

One licence covers the whole repository, the derived artifacts in `data/`
included; there is no separate data licence. It does not extend to the corpus
itself, which is not here to license: the 100 papers are publisher-copyrighted,
are not redistributed, and each user assembles them under their own
subscriptions (see [`data/README.md`](data/README.md)).

## Citation

`CITATION.cff` carries placeholders for the author list and title, deliberately
not guessed. Fill them in before publishing.
