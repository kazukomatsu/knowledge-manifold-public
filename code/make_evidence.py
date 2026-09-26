# -*- coding: utf-8 -*-
"""任意地点の証拠パッケージ生成（語レベル）

特徴語の選択そのものはパイプラインと同一（char n-gram の順位づけ）だが、
選ばれた n-gram を出典テキスト中の「語」へ決定論的に復元してから出力する。
LLM には断片ではなく語が渡るため、compton を人名断片と誤認するような
取り違えが起きない。

同じスコアの n-gram、同じ重みの寄与文献、同じ文献数の語の並べ方は evidence_lib.py に
明示してあり (特徴番号・doc_id・辞書順の昇順)、Python の文字列 hash や numpy のソート実装に
よらない。語彙が 1500 n-gram より少なければ候補は語彙数まで。

usage:
  python3 make_evidence.py --x -0.5 --y 0.0 \
      --out outputs/papers100_local --data /tmp/km_papers100_local \
      --code code
"""
import argparse, json, os, sys, re, csv, pickle
import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--x", type=float, required=True, help="問い合わせ点の x 座標")
ap.add_argument("--y", type=float, required=True, help="問い合わせ点の y 座標")
ap.add_argument("--out", required=True, help="パイプラインの出力ディレクトリ (KM_OUT)")
ap.add_argument("--data", required=True, help="中間ファイルのディレクトリ (KM_DATA)")
ap.add_argument("--code", default="code", help="kmlib.py のあるディレクトリ")
ap.add_argument("--topk", type=int, default=15, help="各レンズの特徴語数")
ap.add_argument("--readout", choices=["auto", "global", "adaptive"], default="auto",
                help="読み出し層の h。auto は論文 Sec.3.3 の選択則"
                     "（LOO で global/adaptive を比較し良い方を採用）")
ap.add_argument("--outfile", default=None, help="出力する JSON のパス")
args = ap.parse_args()
if args.topk < 1:
    ap.error(f"--topk must be at least 1, got {args.topk}")

sys.path.insert(0, os.path.abspath(args.code))
from kmlib import sph_weights, sph_entropy, load_stoplist, GPR, select_readout
from evidence_lib import WordResolver, pick_ngrams, rank_documents
O = os.path.abspath(args.out); D = os.path.abspath(args.data)
P=np.load(O+"/coords.npy"); X=np.load(D+"/X_raw.npy"); l2n=np.load(D+"/l2_norms.npy")
vocab=json.load(open(D+"/vocab.json")); ST=load_stoplist(); DF=(X>0).sum(0)
meta=list(csv.DictReader(open(O+"/corpus_metadata.csv")))
docs=json.load(open(O+"/corpus/docs_clean.json"))
if not vocab or X.ndim != 2 or X.shape[1] != len(vocab):
    sys.exit(f"error: vocab.json has {len(vocab)} n-grams but X_raw.npy has shape {X.shape}")
if not (len(P) == X.shape[0] == len(l2n) == len(meta) == len(docs)):
    sys.exit(f"error: the run's artifacts disagree on the number of documents: coords.npy {len(P)}, "
             f"X_raw.npy {X.shape[0]}, l2_norms.npy {len(l2n)}, corpus_metadata.csv {len(meta)}, "
             f"docs_clean.json {len(docs)}")
Xl2=(X/l2n[:,None]).astype(np.float64); md=Xl2.mean(0); md/=np.linalg.norm(md)
Xf=X.astype(np.float64)+1e-10; Xl1=Xf/Xf.sum(1,keepdims=True); ml1=Xl1.mean(0); del Xf
gm=pickle.load(open(D+"/gpr_model.pkl","rb")); gp=GPR(kernel="rbf")
for k in ("theta","X","alpha","L","noise"): setattr(gp,k,gm[k])

resolver=WordResolver(vocab,docs)

def lens_words(picks):
    """n-gram picks -> word-level entries (evidence_lib.WordResolver), in the package's format."""
    out=[]
    for g in resolver.resolve(picks):
        rec={"term":g["term"],"word_forms":g["word_forms"],"df":len(g["docs"])}
        if rec["df"]<=3:
            rec["source_docs"]=[{"doc":int(d),"title":meta[d]["title"][:70]} for d in g["docs"]]
        out.append(rec)
    return out

pt=np.array([[args.x,args.y]])
if args.readout == "auto":
    _mode, _kw = select_readout(O)
elif args.readout == "global":
    _mode, _kw = "global", {}
else:
    _mode, _kw = "knn_adaptive", {"knn_k": 8}
print(f"readout tier: {_mode}" + (" (selected by LOO)" if args.readout == "auto" else " (forced)"))
w=sph_weights(pt,P,h_mode=_mode,**_kw)[0]
v=w@Xl2; v/=np.linalg.norm(v); pi=w@Xl1; pi/=pi.sum()
kl=pi*np.log(np.maximum(pi,1e-300)/np.maximum(ml1,1e-300))
L2=lens_words(pick_ngrams(v-md,vocab,ST,args.topk)); L1=lens_words(pick_ngrams(kl,vocab,ST,args.topk))
E={"query_location":[args.x,args.y],
   "corpus":"100 papers, carbon-fiber composite research (single laboratory)",
   "how_produced":"deterministic pipeline; feature selection on character n-grams, then each selected n-gram resolved to the words containing it in the corpus text; terms below are WORDS, df = number of corpus documents containing the word",
   "theme_lens_L2":L2,"concentration_lens_L1":L1,
   "contributing_documents_top3":[{"doc":int(i),"weight":round(float(w[i]),3),"title":meta[i]["title"][:70]}
        for i in rank_documents(w,3)],
   "mixture_entropy_H":round(float(sph_entropy(w[None],len(P))[0]),3),
   "effective_contributing_documents":round(float(len(P)**sph_entropy(w[None],len(P))[0]),1),
   "relative_uncertainty_u":round(float(gp.rel_uncertainty(pt)[0]),3)}
outfile = args.outfile or f"evidence_point_{args.x}_{args.y}.json"
json.dump(E, open(outfile, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"query      : ({args.x}, {args.y})")
print(f"u          : {E['relative_uncertainty_u']}")
print(f"entropy H  : {E['mixture_entropy_H']}  (effective {E['effective_contributing_documents']} documents)")
print(f"L2 lens    : {', '.join(e['term'] for e in E['theme_lens_L2'])}")
print(f"L1 lens    : {', '.join(e['term'] for e in E['concentration_lens_L1'])}")
print(f"written    : {outfile}")
