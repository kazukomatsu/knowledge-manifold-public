# -*- coding: utf-8 -*-
"""§1.3/§2 TF-IDF構築 (sklearn TfidfVectorizer 純正実装).
char_wb 4-7gram, sublinear_tf, max_features=250000, min_df=1, norm=None (§1.3)。
L2版・L1版(ε=1e-10)の係数, Gram行列, SVDスコアを KM_DATA (/tmp/km_sk) に保存。
usage: python3 02_tfidf_sklearn.py
"""
import json, time, pickle, os, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import OUT, DATA

os.makedirs(DATA, exist_ok=True)
EPS_L1 = 1e-10
t0 = time.time()

docs = json.load(open(OUT + "/corpus/docs_clean.json"))
texts = [d["text"] for d in docs]

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 7), sublinear_tf=True,
                      max_features=250000, min_df=1, norm=None, lowercase=False)
Xs = vec.fit_transform(texts)                       # 前処理§1.1で小文字化済みのため lowercase=False
X = np.asarray(Xs.todense(), dtype=np.float32)
np.save(DATA + "/X_raw.npy", X)

l2n = np.linalg.norm(X, axis=1)
np.save(DATA + "/l2_norms.npy", l2n)
V = X.shape[1]
l1s = X.sum(axis=1) + EPS_L1 * V
np.save(DATA + "/l1_sums.npy", l1s)

Xl2 = (X / l2n[:, None]).astype(np.float64)
G = Xl2 @ Xl2.T                                     # Gram行列 (SPH計量の厳密内積用)
np.save(DATA + "/gram_l2.npy", G)
w, U = np.linalg.eigh(G)
o = np.argsort(w)[::-1]
w, U = np.clip(w[o], 0, None), U[:, o]
np.save(DATA + "/svd_scores.npy", U * np.sqrt(w)[None, :])   # TruncatedSVD相当 (GPR潜在用)
np.save(DATA + "/svd_singular_values.npy", np.sqrt(w))
json.dump(vec.get_feature_names_out().tolist(), open(DATA + "/vocab.json", "w"))
pickle.dump(vec, open(DATA + "/tfidf_vectorizer.pkl", "wb"))   # §9.3 再現性成果物

# 確率単体検証 (§2.2) + 設定ファイル (§9.3)
rows = ((X + EPS_L1) / l1s[:, None]).sum(axis=1)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer="char_wb", ngram_range=(4, 7), lowercase=False)
cv.fit(texts)
V_full = len(cv.vocabulary_)
p = (X[0].astype(np.float64) + EPS_L1); p /= p.sum()
q = (X[1].astype(np.float64) + EPS_L1); q /= q.sum()
m_ = 0.5 * (p + q)
kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log(np.maximum(a, 1e-300) / np.maximum(b, 1e-300)), 0)))
js = 0.5 * kl(p, m_) + 0.5 * kl(q, m_)
hel = float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2))
fr = float(2 * np.arccos(min(1.0, float(np.sum(np.sqrt(p * q))))))
w2 = np.load(DATA + "/svd_singular_values.npy") ** 2
cfg = {"analyzer": "char_wb", "ngram_range": [4, 7], "sublinear_tf": True, "smooth_idf": True,
       "max_features": 250000, "min_df": 1, "norm": None,
       "vocab_size_full": V_full, "vocab_size": V, "nnz": int((X > 0).sum()),
       "epsilon_l1_smoothing": EPS_L1,
       "implementation": "sklearn TfidfVectorizer (analyzer=char_wb)",
       "l1_simplex_rowsum_min": float(rows.min()), "l1_simplex_rowsum_max": float(rows.max()),
       "example_doc0_doc1": {"js_divergence": js, "js_distance": float(np.sqrt(js)),
                              "hellinger": hel, "fisher_rao": fr},
       "svd_top10_evr": (w2[:10] / w2.sum()).round(5).tolist()}
json.dump(cfg, open(OUT + "/tfidf_config.json", "w"), indent=1)
print("done: nnz=%d V_full=%d rowsum=[%.9f, %.9f]  %.1fs"
      % (int((X > 0).sum()), V_full, rows.min(), rows.max(), time.time() - t0))
