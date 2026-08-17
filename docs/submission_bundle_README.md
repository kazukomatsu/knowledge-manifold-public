# Knowledge Manifold — code, configuration and evaluation artifacts

> **This file describes the submission bundle, not this repository.** It is kept
> verbatim (received 2026-08-14) because it records the paper-to-artifact
> correspondence and the contents of the bundle — `config/`, `results/`,
> `evidence/`, `results_external/`, `supplementary/` — none of which are part of
> this tree. Paths below resolve inside the bundle. Two statements have since been
> reconciled with this repository: the Python version — 3.10 is confirmed for the
> corpus-free verification path, while the pinned 3.11.15 stack stays necessary to
> rebuild the published map (see "Python 3.10" in the top-level README) — and the
> direction of the post-fix mirroring, where the diagonal mirror stated below is
> now what `CORRECTIONS.md` #2 records for the comparison against the earlier
> drafts. Against the manuscript of 2026-08-17 there is no mirroring at all: its
> Fig. 1 is in the same gauge as `data/derived/coords.npy`, verified anchor by
> anchor. Note also that this file predates the manuscript of 2026-08-17, which
> carries a different title (*Knowledge Manifold: A Geometric Framework for
> Science Mapping and Knowledge Navigation*) and evaluates **three** corpora —
> the laboratory one plus journal samples from *Polymer* and the *Journal of
> Informetrics* — where the layout below names only one external corpus.
> `CITATION.cff` follows the 2026-08-17 manuscript.

Reproduction material for

> Komatsu, Kawagoe, Obayashi, Okabe,
> *Knowledge Manifold: An Information-Geometric Framework for Scientific
> Knowledge Exploration*, submitted to the Journal of Informetrics.

Everything reported in the paper is produced by the code in `code/` from the
configuration in `config/`, and is written to machine-readable files in
`results/`.  Every number in the manuscript and the supplementary material can
be traced to one of these files.

## What is and is not included

The corpus consists of 100 published papers of the authors' laboratory.  Their
full texts are copyrighted and are **not** redistributed here.  The release
contains the bibliographic list (`results/corpus_table_S1.csv`, with titles,
years and DOIs), all derived quantities, and the code that produces them from a
corpus of text files.  To reproduce the pipeline end to end, obtain the 100
documents from their publishers and place their extracted text in the input
directory expected by `code/00_derived_input_corpus.py`.

## Layout

```
code/        pipeline (numbered by execution order) and the shared library
config/      fixed configuration (spec v5.2), run manifest and hashes
results/
  metrics/     evaluation outputs, one JSON or CSV per evaluation
  coordinates/ the map, and every variant used in the robustness suite
  paths/       geodesics (coordinates and optimizer records)
  figures/     the figures of the paper and of the supplementary material
  plot_data/   the same figures as plain-text data, for replotting
  appendix/    supporting analyses (vocabulary cap, two-tier readout, E7)
evidence/    the machine-computed evidence packages given to language models,
             the cross-model kit and the six verbatim responses
results_external/  the same pipeline on a second, unrelated corpus
             (100 papers sampled from the journal Polymer, 2020-2026;
             bibliographic list corpus_table_S2.csv; full texts not included)
supplementary/ the supplementary material as submitted
```

## Reproducing a run

```
python3 code/make_derived_input.py --input papers.md --output derived/   # corpus packaging
bash code/run_v50.sh --derived-input derived/ --run-id run1 --outputs-root outputs/
python3 code/validate.py       # 15 validation gates; non-zero exit on failure
python3 code/make_evidence.py --x -0.5 --y 0.0 --out outputs/run1 \
    --data outputs/run1/work --code code   # evidence package for any location
```

The readout smoothing length follows the selection protocol of the paper
(Sec. 3.3): both rules are scored by leave-one-out reconstruction and the
better one is adopted automatically (`kmlib.select_readout`); on the
laboratory corpus this selects the adaptive rule, on the journal corpus the
global one.

`code/validate.py` must pass before any number is reported; the gates and their
thresholds are listed in Table S1 of the supplementary material.

## Determinism

The geometry is produced by deterministic numerical code on a CPU; no GPU and
no neural model are involved in constructing the map, the field, the metrics or
the geodesics.  A fixed corpus and configuration always yield the same output.
A language model is used only after `code/13_llm_export.py` has written the
evidence package, and only to render that evidence as prose; its output is not
deterministic, which is why the six responses in
`evidence/E7_cross_model_kit/responses/` are released verbatim.

## Environment

Python 3.10, NumPy, SciPy, scikit-learn.  `umap-learn` is required only for the
baseline comparison in `code/15_e10_embeddings.py` and is optional elsewhere.
Runtimes reported in the supplementary material were measured on a single CPU
core.

## Correspondence between the paper and the files

| Paper | Content | Principal artifacts |
|---|---|---|
| E1 | leave-one-out prediction | `results/metrics/e2_loo_*.json` |
| E2 | distance preservation | `results/metrics/e1_distance_preservation.json` |
| E3 | embedding baselines | `results/metrics/e10_embedding_comparison.json` |
| E4 | geodesic family | `results/metrics/geodesic_results.csv`, `e4_lambda_path_deviation.json`, `results/paths/` |
| E5 | lenses and readout | `results/metrics/e9_verbalization_L1_L2.json`, `results/appendix/add_analysis_E9_twotier/` |
| E6 | evidence budget and vocabulary | `results/appendix/add_analysis_E12_vocabulary/` |
| E7 | cross-model verbalization | `evidence/E7_cross_model_kit/` |
| E8 | robustness suite | `results/metrics/e7_*.json`, `r1_N*.json` |
| E9 | external corpus | `results_external/` |

File names carry the internal numbering under which the suite was developed;
the paper numbers the evaluations in order of presentation.

## Update 2026-08-06

Two code corrections (see the public repository's CORRECTIONS.md): the
`knn_preservation` metric previously compared k+1 neighbours while dividing by k
(all kNN values decrease uniformly; method rankings unchanged), and the anchor
assignment now fixes the D4 gauge degeneracy deterministically (tolerance-based
tie set, lexicographic minimum). The canonical laboratory-corpus map is the
diagonal mirror (x<->y) of earlier drafts; all field content is unchanged.
`results/` holds the regenerated laboratory-corpus artifacts; `results_external/`
(journal corpus) is unchanged in both orientation and values. Embedding-comparison
values for t-SNE/UMAP are taken from the pinned CI environment of the public
repository. The E7 kit evidence file is relabelled to the canonical gauge
(content verified identical).
