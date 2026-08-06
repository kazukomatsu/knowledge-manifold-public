# Data in this repository

## What is here, and what is not

The corpus analysed in the paper is 100 published papers on carbon-fibre
composite materials. **Their full text is not in this repository and cannot be**,
because it consists of publisher PDFs converted to Markdown. Redistributing it
would infringe the publishers' copyright.

What ships instead is the *derived* representation — everything needed to
reproduce the published map, its evaluation metrics and its figures, without
containing the text.

| Path | Contents | Reconstructs text? |
|---|---|---|
| `corpus_manifest.csv` | one row per document: DOI, title, year, cluster, map coordinates | No — bibliographic facts |
| `derived/gram_l2.npy` | 100 x 100 L2-normalised Gram matrix (document-document cosine similarity) | No — 100 scalars per document |
| `derived/svd_scores.npy` | 100 x 100 truncated-SVD latent scores | No |
| `derived/coords.npy`, `derived/coordinates_2d.csv` | the published 2-D map | No |
| `derived/cluster_labels.npy` | k-means cluster assignment | No |
| `derived/map_audit.json` | anchor selection, alpha, kappa, objective terms, optimiser trace | No |
| `derived/e1_*.json`, `derived/e2_*.json`, `derived/e10_*.json` | published metric values, used as the reference for `verify_reference.py` | No |
| `derived/manifest.json`, `derived/tfidf_config.json` | run provenance and TF-IDF settings | No |
| `reference_figures/*.png` | the nine published figures | No |

Deliberately **absent**: `X_raw.npy` (the 100 x 250000 char 4–7-gram TF-IDF
matrix) and `vocab.json` (the n-gram vocabulary). Overlapping character n-grams
plus their vocabulary can be reassembled into running text, so both are
withheld — quite apart from `X_raw.npy` being ~200 MB.

## Reproducing without the corpus

```bash
python3 code/verify_reference.py
```

recomputes 28 published metrics from `derived/gram_l2.npy` + `derived/coords.npy`
and checks them against the shipped reference JSONs. This works because every E1
metric, and E2's `loo_cosine_*` and `nearest_paper_*` rates, depend on the
documents only through their inner products. With `v = sum_j w_j x_j`:

```
v . x_i    = w . G[mask, i]
||v||^2    = w^T G[mask][:, mask] w
X_mask . v = G[mask][:, mask] @ w      (up to the positive factor 1/||v||)
```

Verified: all 28 metrics reproduce, most at exactly zero difference and the rest
at machine epsilon (<= 1e-15).

`E2 topk_recovery_mean` and `topk_recovery_shared_df2_mean` are **not**
reproducible this way — they argsort the 250000-dimensional feature vectors, so
they need the corpus.

## Reproducing with your own corpus

Stages 0 and 1 (`make_derived_input.py`, `02_tfidf_sklearn.py`) rebuild
`X_raw.npy` from full text you hold yourself. `corpus_manifest.csv` lists the
100 DOIs so you can assemble the same corpus under your own subscriptions;
`derived/manifest.json` records `source_sha256` for the concatenated Markdown
the published run used, so you can tell whether your rebuild matched it.

Note that stage 1's Gram matrix is sensitive to the BLAS/SciPy build at the
1e-8 level. That is harmless for the metrics but see the note on anchor gauge
fixing in the top-level `README.md`.

## Provenance of the shipped artifacts

Produced by the run recorded in `derived/manifest.json`: Python 3.11.15,
NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0, matplotlib 3.11.1,
umap-learn 0.5.12, all 15 validation gates passing. The absolute input path in
`manifest.json` has been redacted; `source_sha256` is unchanged.
