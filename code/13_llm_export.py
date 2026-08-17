# -*- coding: utf-8 -*-
"""LLM解釈用エクスポート: ベクトル演算を要する「意味の材料」をすべて前計算し、
LLM(GPT/Gemini/Claude)にそのまま渡せる自己完結テキスト(JSON)にする。
ベクトル本体(100MB)は不要になる。出力: llm_context.json (~数百KB)

含むもの:
 A. コーパス概要: 文書ID・タイトル・年・クラスタ
 B. クラスタ特徴: 各クラスタのtop特徴語・代表文書
 C. 経路トレース: 全測地線について31点ごとの
    - 位置, SPHエントロピー, GPR不確かさ u
    - 特徴的n-gram top12 (コーパス平均方向との差分, 意味軸を表す)
    - 寄与上位文書3件 (SPH重みとタイトル)
    - 前ステップからのcosine変化
 D. 4象限勾配の言語化材料 (knowledge_gradient.jsonを統合)
 E. research gap候補: u(r)×最近文書距離の内部極大点とその周辺意味
 F. 経路指標表 (results_metrics) と読み方の注記
usage: python3 13_llm_export.py
"""
import json, os, sys, csv, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import OUT, DATA, load_tfidf, sph_weights, sph_entropy, GPR, kmeans, term_ok, load_stoplist, select_readout
import pickle

TOPK_FEAT = 15
# 読み出し層の選択 (Sec.3.3): LOO で global / adaptive を比較して良い方を使う
READOUT_MODE, READOUT_KW = None, None
TOPK_DOCS = 3
t_out = {}

P = np.load(OUT + "/coords.npy")
READOUT_MODE, READOUT_KW = select_readout(OUT)
print(f"readout tier selected: {READOUT_MODE}", flush=True)
X, l2n, l1s = load_tfidf()
Xl2 = (X / l2n[:, None]).astype(np.float64)
vocab = json.load(open(DATA + "/vocab.json"))
meta = list(csv.DictReader(open(OUT + "/corpus_metadata.csv")))
lab = np.load(DATA + "/cluster_labels.npy")
mean_dir = Xl2.mean(0); mean_dir /= np.linalg.norm(mean_dir)
Xf1 = X.astype(np.float64) + 1e-10
Xl1 = Xf1 / Xf1.sum(1, keepdims=True)
del Xf1
mean_l1 = Xl1.mean(0)
DF = (X > 0).sum(0)                     # document frequency (証拠の重さの注記用; フィルタには使わない)
md = pickle.load(open(DATA + "/gpr_model.pkl", "rb"))
gp = GPR(kernel="rbf"); gp.theta = md["theta"]; gp.X = md["X"]; gp.alpha = md["alpha"]
gp.L = md["L"]; gp.noise = md["noise"]

_STOP = load_stoplist()
def _pick_idx(scores, k):
    """上位候補から TeX/数式断片・著者名断片(term_stoplist.txt)と重複n-gramを除いて返す."""
    cand = np.argpartition(scores, -600)[-600:]
    cand = cand[np.argsort(scores[cand])[::-1]]
    words, idxs = [], []
    for i in cand:
        t = vocab[i].strip()
        if not term_ok(t, _STOP): continue
        if any(t in x or x in t for x in words): continue
        words.append(t); idxs.append(int(i))
        if len(words) >= k:
            break
    return words, idxs

def top_features(v, k=TOPK_FEAT):
    """コーパス平均方向との差分でその位置に特徴的なn-gramを返す(L2レンズ)."""
    return _pick_idx(v - mean_dir, k)[0]

def l1_kl_features(pi, k=TOPK_FEAT):
    """L1レンズ: KL寄与 pi_k*log(pi_k/mean_k)。df と df<=3 の出典文書を注記する。"""
    kl = pi * np.log(np.maximum(pi, 1e-300) / np.maximum(mean_l1, 1e-300))
    words, idxs = _pick_idx(kl, k)
    out = []
    for t, i in zip(words, idxs):
        e = {"ngram": t, "df": int(DF[i])}
        if DF[i] <= 3:
            srcs = np.where(X[:, i] > 0)[0][:3]
            e["source_docs"] = [{"doc": int(s), "title": meta[s]["title"][:60]} for s in srcs]
        out.append(e)
    return out

def point_semantics(pt):
    w = sph_weights(np.asarray(pt)[None], P, h_mode=READOUT_MODE, **READOUT_KW)[0]
    v = w @ Xl2
    v /= np.linalg.norm(v)
    pi = w @ Xl1
    pi = pi / pi.sum()
    docs = np.argsort(w)[::-1][:TOPK_DOCS]
    return v, {
        "xy": [round(float(pt[0]), 3), round(float(pt[1]), 3)],
        "entropy": round(float(sph_entropy(w[None], len(P))[0]), 3),
        "gpr_uncertainty": round(float(gp.rel_uncertainty(np.asarray(pt)[None])[0]), 3),
        "top_ngrams": top_features(v),
        "l1_kl_ngrams": l1_kl_features(pi),
        "contributing_docs": [{"doc": int(i), "weight": round(float(w[i]), 3),
                               "title": meta[i]["title"][:70]} for i in docs if w[i] > 0.01],
    }

# ---- A. コーパス概要 ----
t_out["corpus"] = {
    "N": len(P), "field": "carbon fiber composites / polymer / damage mechanics (char_wb 4-7gram TF-IDF)",
    "documents": [{"doc": i, "title": meta[i]["title"][:80], "year": meta[i]["year"],
                   "cluster": int(lab[i]), "xy": [round(float(P[i, 0]), 3), round(float(P[i, 1]), 3)]}
                  for i in range(len(P))]}

# ---- B. クラスタ特徴 ----
clusters = []
for c in sorted(set(lab.tolist())):
    idx = np.where(lab == c)[0]
    cv = Xl2[idx].mean(0); cv /= np.linalg.norm(cv)
    cen = P[idx].mean(0)
    rep = idx[np.argmin(np.linalg.norm(P[idx] - cen, axis=1))]
    clusters.append({"cluster": int(c), "size": int(len(idx)),
                     "top_ngrams": top_features(cv),
                     "representative_doc": {"doc": int(rep), "title": meta[rep]["title"][:70]},
                     "centroid_xy": [round(float(cen[0]), 3), round(float(cen[1]), 3)]})
t_out["clusters"] = clusters

# ---- C. 経路トレース ----
pairs = json.load(open(DATA + "/pairs.json"))
traces = []
for f in sorted(glob.glob(DATA + "/path_*.npy")):
    name = os.path.basename(f)[5:-4]          # {pair}_{method}
    pair, method = name.rsplit("_", 1)
    if pair not in pairs:
        continue
    path = np.load(f)
    i, j = pairs[pair]
    steps = []
    prev_v = None
    for k in range(len(path)):
        v, sem = point_semantics(path[k])
        sem["t"] = round(k / (len(path) - 1), 3)
        if prev_v is not None:
            sem["cos_change_from_prev"] = round(float(1 - v @ prev_v), 6)
        prev_v = v
        steps.append(sem)
    # 全点は冗長なので 0,3,6,...,30 の11点に間引き (端点は必ず含む)
    keep = sorted(set(list(range(0, len(path), 3)) + [len(path) - 1]))
    traces.append({"pair": pair, "method": method,
                   "from": {"doc": i, "title": meta[i]["title"][:70]},
                   "to": {"doc": j, "title": meta[j]["title"][:70]},
                   "waypoints": [steps[k] for k in keep]})
    print("traced", pair, method, flush=True)
t_out["path_traces"] = traces

# ---- D. 勾配言語化材料 ----
t_out["knowledge_gradient_quadrants"] = json.load(open(OUT + "/knowledge_gradient.json"))

# ---- E. research gap候補 (内部, u×最近文書距離の複合スコア極大) ----
n = 61
xs = np.linspace(-0.8, 0.8, n)
gx, gy = np.meshgrid(xs, xs)
GR = np.stack([gx.ravel(), gy.ravel()], 1)
u = gp.rel_uncertainty(GR)
nd = np.linalg.norm(GR[:, None] - P[None], axis=2).min(axis=1)
un = (u - u.min()) / (u.max() - u.min()); ndn = (nd - nd.min()) / (nd.max() - nd.min())
score = un * ndn
spots = []
for iidx in np.argsort(score)[::-1]:
    ppt = GR[iidx]
    if all(np.linalg.norm(ppt - np.array(s["xy"])) > 0.3 for s in spots):
        _, sem = point_semantics(ppt)
        sem["gap_score"] = round(float(score[iidx]), 3)
        spots.append(sem)
    if len(spots) >= 3:
        break
t_out["research_gap_candidates"] = spots

# ---- F. 指標表と注記 ----
t_out["metrics_table"] = list(csv.DictReader(open(OUT + "/results_metrics.csv")))
t_out["how_to_read"] = {
    "top_ngrams": "char_wb 4-7文字n-gram。コーパス平均に対して当該位置で過剰な特徴。断片から語を復元して解釈すること (例: '3d-pri'→3D printing, 'cfrtp'→carbon fiber reinforced thermoplastic)",
    "entropy": "0=少数文献支配(明確な局所知識), 1=多文献混合(橋渡し/曖昧領域)",
    "gpr_uncertainty": "u=σ_post/σ_prior。高いほど文献支持が弱い(research gap指標)。本コーパスの範囲は約0.17-0.42",
    "cos_change_from_prev": "前ステップからの意味ベクトル変化(1-cos)。大きい箇所が概念の切り替わり",
    "methods": {"line": "直線基準", "graph": "実在文献の飛び石(Dijkstra)", "sph": "SPH計量測地線(滑らかさ最適)",
                "gpr1/4/9": "不確かさ回避測地線(λ=1/4/9)", "fr": "Fisher-Rao情報幾何測地線(特徴配分の急変回避)"},
    "l1_kl_ngrams": "L1レンズ(KL寄与 pi*log(pi/mean))。その位置で統計的に過剰濃縮した特徴。dfは出現文書数(証拠の重さ)。df<=3にはsource_docsを付す",
    "caution": "言語化はhypothesis suggestionであり、最終妥当性は専門家評価(E8)が担う (仕様§7.4)"}

# ---- G. 言語化プロトコル (LLM向け規定; E9深掘りで確立した2部構成) ----
t_out["verbalization_protocol"] = {
    "purpose": "任意のLLMが本JSONのみから、証拠の重さを偽らない言語化を再現するための規定",
    "structure": [
        "第1部(主題): top_ngrams(L2方向差分)と contributing_docs から、その位置/経路区間の合意的な主題を構成する。領域の知識として断定してよいのはここまで。",
        "第2部(セレンディピティ): l1_kl_ngrams のうち top_ngrams と重複しない語を検討する。df>=4 は『少数派だが共有された萌芽テーマ』として提示可。df<=3 は必ず『単一源・未検証の手法移転仮説』と明示し、source_docs のタイトルを出典として引用した上で『こんな方法もあり得る』の提案形でのみ述べる。"],
    "prohibitions": [
        "df<=3 の語を領域の性質・傾向として記述すること",
        "本JSONに存在しない事実・数値で補完すること",
        "著者名・誌名・書誌情報由来のn-gram断片(人名の部分文字列等)を内容語として解釈すること (判断に迷う場合は無視する)",
        "数式・記号断片(例: v'/e')を内容語として解釈すること"],
    "audit": "生成した言語化の末尾に、根拠として採用した語(top_ngrams / l1_kl_ngrams)とそのdfを列挙し、本JSONと突合可能にすること",
    "established_by": "E9深掘り(2026-07): L2=語り/L1=検出の役割分担 + df注記による証拠重み表示。(-0.5,0)仮想論文の実演で確立"}

out_path = OUT + "/llm_context.json"
json.dump(t_out, open(out_path, "w"), ensure_ascii=False, indent=1)
print("written:", out_path, f"{os.path.getsize(out_path)/1024:.0f} KB")
