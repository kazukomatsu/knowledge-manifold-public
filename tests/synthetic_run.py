# -*- coding: utf-8 -*-
"""Build a complete, tiny pipeline run from invented text, for the post-processing tests.

The layout is the one run_v50.sh leaves behind (KM_OUT with work/ and corpus/), and each
artifact is made the way the pipeline makes it: TF-IDF exactly as 02_tfidf_sklearn.py, the
standard clusters as 05_geodesics.py (kmeans on 10 SVD dims, k=5, seed 0) and the GPR as
04_fields.py gpr. Only the map is not optimised: nine anchors sit on the fixed anchor
lattice and the other documents at seeded random positions inside the box. No corpus
text is involved — the words are made up.
"""
import csv
import hashlib
import json
import os
import pickle
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
sys.path.insert(0, CODE)

from kmlib import GPR, kmeans  # noqa: E402

TOPICS = [
    ["fibrolux", "carbonate", "stretchon", "tensilor", "filamex", "weavora"],
    ["curatone", "gelatrix", "resinox", "thermavex", "exotherm", "viscora"],
    ["lamellix", "crystora", "nucleon", "spherulite", "annealix", "meltora"],
    ["voidrix", "crackon", "fractora", "delamix", "fatiguon", "notchera"],
    ["solvenox", "swellora", "diffusix", "permeon", "membrix", "sorptone"],
]
COMMON = ["sample", "results", "method", "analysis", "measured", "increase", "specimen", "surface"]
ANCHOR_POS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]], float)


SYLLABLES = ["ka", "lo", "mi", "ra", "tu", "vex", "zor", "pil", "dan", "qui", "sel", "fen", "bro", "gat", "hum", "jor"]


def _texts(topic_of, rng, seed):
    # a few made-up words private to each document keep the n-gram vocabulary above the
    # 1500-candidate pool that make_evidence.py assumes
    texts = []
    for i, t in enumerate(topic_of):
        topic = TOPICS[t]
        other = TOPICS[(t + 1) % len(TOPICS)]
        own = ["".join(rng.choice(SYLLABLES, 4)) for _ in range(6)]
        rare = f"rareword{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}x" + ("" if seed == 0 else chr(97 + seed % 26) * 3)
        words = (list(rng.choice(topic, 40)) + list(rng.choice(other, 8)) + list(rng.choice(COMMON, 25))
                 + own + [rare])
        rng.shuffle(words)
        texts.append(" ".join(words))
    return texts


def build_run(root, n_docs=16, seed=0, n_clusters=5):
    """Write a synthetic run under root/run and return its KM_OUT path."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    rng = np.random.default_rng(seed)
    out = os.path.join(str(root), "run")
    work = os.path.join(out, "work")
    os.makedirs(os.path.join(out, "corpus"))
    os.makedirs(work)
    # seed 0 puts document i in topic i % 5; other seeds shuffle that, so two runs built with
    # different seeds differ in grouping, text, titles and map, but share the file names
    topic_of = np.arange(n_docs) % len(TOPICS)
    if seed != 0:
        topic_of = rng.permutation(topic_of)
    texts = _texts(topic_of, rng, seed)
    docs = [{"doc_id": i, "filename": f"KM{i + 1:04d}", "text": t} for i, t in enumerate(texts)]
    with open(os.path.join(out, "corpus", "docs_clean.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f)
    with open(os.path.join(out, "corpus_metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "filename", "title", "year", "doi", "chars_cleaned"])
        for i in range(n_docs):
            title = f"Synthetic study {i} of {TOPICS[topic_of[i]][0]}" + ("" if seed == 0 else f" (set {seed})")
            w.writerow([i, f"KM{i + 1:04d}", title, 2020, "", len(texts[i])])

    # stage 1, as 02_tfidf_sklearn.py
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 7), sublinear_tf=True,
                          max_features=250000, min_df=1, norm=None, lowercase=False)
    X = np.asarray(vec.fit_transform(texts).todense(), dtype=np.float32)
    np.save(os.path.join(work, "X_raw.npy"), X)
    l2n = np.linalg.norm(X, axis=1)
    np.save(os.path.join(work, "l2_norms.npy"), l2n)
    np.save(os.path.join(work, "l1_sums.npy"), X.sum(axis=1) + 1e-10 * X.shape[1])
    Xl2 = (X / l2n[:, None]).astype(np.float64)
    G = Xl2 @ Xl2.T
    np.save(os.path.join(work, "gram_l2.npy"), G)
    wv, U = np.linalg.eigh(G)
    o = np.argsort(wv)[::-1]
    wv, U = np.clip(wv[o], 0, None), U[:, o]
    np.save(os.path.join(work, "svd_scores.npy"), U * np.sqrt(wv)[None, :])
    with open(os.path.join(work, "vocab.json"), "w") as f:
        json.dump(vec.get_feature_names_out().tolist(), f)

    # the map: anchors on the lattice, the rest at seeded positions
    n_anchor = min(9, n_docs)
    anchors = list(range(n_anchor))
    P = np.empty((n_docs, 2))
    P[:n_anchor] = ANCHOR_POS[:n_anchor]
    centres = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.0, 0.1]])
    free = np.arange(n_anchor, n_docs)
    P[free] = np.clip(centres[topic_of[free]] + rng.normal(0, 0.15, (len(free), 2)), -0.9, 0.9)
    np.save(os.path.join(out, "coords.npy"), P)
    with open(os.path.join(out, "coordinates_2d.csv"), "w") as f:
        f.write("doc_id,x,y,is_anchor,anchor_slot\n")
        for i in range(n_docs):
            f.write(f"{i},{P[i, 0]:.8f},{P[i, 1]:.8f},{int(i in anchors)},{anchors.index(i) if i in anchors else -1}\n")
    with open(os.path.join(out, "map_audit.json"), "w") as f:
        json.dump({"N": n_docs, "anchor_docs": anchors}, f)

    # standard clusters, as 05_geodesics.py pairs
    Z = np.load(os.path.join(work, "svd_scores.npy"))[:, :10]
    lab, _ = kmeans(Z, n_clusters, seed=0)
    np.save(os.path.join(work, "cluster_labels.npy"), lab)

    # GPR, as 04_fields.py gpr (fewer restarts: this is a fixture, not a result)
    zs = Z.std()
    with warnings.catch_warnings():             # tiny invented corpora can push a length scale to its bound
        warnings.simplefilter("ignore")
        gp = GPR(kernel="rbf", seed=0, n_restarts=2).fit(P, Z / zs)
    with open(os.path.join(work, "gpr_model.pkl"), "wb") as f:
        pickle.dump({"theta": gp.theta, "X": gp.X, "alpha": gp.alpha, "L": gp.L, "noise": gp.noise,
                     "lml": gp.lml_, "zscale": zs, "kernel": "rbf"}, f)

    # the part of 09_manifest.py the post-processing tools compare against
    sha = lambda name: hashlib.sha256(open(os.path.join(out, name), "rb").read()).hexdigest()
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"N": n_docs, "artifact_hashes": {n: sha(n) for n in ("coordinates_2d.csv", "corpus_metadata.csv")}}, f)
    return out
