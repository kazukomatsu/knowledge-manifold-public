# -*- coding: utf-8 -*-
"""任意地点の証拠パッケージ生成（語レベル）

特徴語の選択そのものはパイプラインと同一（char n-gram の順位づけ）だが、
選ばれた n-gram を出典テキスト中の「語」へ決定論的に復元してから出力する。
LLM には断片ではなく語が渡るため、compton を人名断片と誤認するような
取り違えが起きない。

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
ap.add_argument("--outfile", default=None, help="出力する JSON のパス")
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.code))
from kmlib import sph_weights, sph_entropy, term_ok, load_stoplist, GPR
O = os.path.abspath(args.out); D = os.path.abspath(args.data)
P=np.load(O+"/coords.npy"); X=np.load(D+"/X_raw.npy"); l2n=np.load(D+"/l2_norms.npy")
vocab=json.load(open(D+"/vocab.json")); ST=load_stoplist(); DF=(X>0).sum(0)
meta=list(csv.DictReader(open(O+"/corpus_metadata.csv")))
docs=json.load(open(O+"/corpus/docs_clean.json"))
tok=[set(re.findall(r"[a-z][a-z\-]{2,}",d["text"].lower())) for d in docs]
Xl2=(X/l2n[:,None]).astype(np.float64); md=Xl2.mean(0); md/=np.linalg.norm(md)
Xf=X.astype(np.float64)+1e-10; Xl1=Xf/Xf.sum(1,keepdims=True); ml1=Xl1.mean(0); del Xf
gm=pickle.load(open(D+"/gpr_model.pkl","rb")); gp=GPR(kernel="rbf")
for k in ("theta","X","alpha","L","noise"): setattr(gp,k,gm[k])

def words_for(frag_raw):
    """padded fragment -> containing words, respecting word-boundary padding."""
    f=frag_raw
    lead=f.startswith(" "); trail=f.endswith(" "); s=f.strip()
    hits={}
    for di,ts in enumerate(tok):
        for w in ts:
            ok=(w==s) if (lead and trail) else (w.startswith(s) if lead else (w.endswith(s) if trail else (s in w)))
            if ok: hits.setdefault(w,set()).add(di)
    return hits

def pick_idx(sc,k=None):
    k = k or args.topk
    cand=np.argpartition(sc,-1500)[-1500:]; cand=cand[np.argsort(sc[cand])[::-1]]
    out=[]
    for c in cand:
        s=vocab[c].strip()
        if not term_ok(s,ST): continue
        if any(s in x or x in s for x,_ in out): continue
        out.append((s,int(c)))
        if len(out)>=k: break
    return out

def lens_words(idx_list):
    """fragment picks -> word-level entries, merged by shared containing word."""
    entries=[]
    for s,c in idx_list:
        hits=words_for(vocab[c])
        if not hits:   # no containing word (should not happen)
            entries.append({"frags":[s],"raw":[vocab[c]],"words":{}})
            continue
        entries.append({"frags":[s],"raw":[vocab[c]],"words":hits})
    # union-find: merge entries sharing any containing word
    n=len(entries); par=list(range(n))
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    def rep_of(e):
        return max(e["words"],key=lambda w2:len(e["words"][w2])) if e["words"] else e["frags"][0]
    for i in range(n):
        for j in range(i+1,n):
            a,b=entries[i],entries[j]
            share=set(a["words"])&set(b["words"])
            ra,rb=rep_of(a),rep_of(b)
            morph=(len(ra)>=4 and len(rb)>=4 and (ra.startswith(rb) or rb.startswith(ra)))
            if share or morph: par[find(i)]=find(j)
    groups={}
    for i in range(n): groups.setdefault(find(i),[]).append(entries[i])
    merged={}
    for g in groups.values():
        allw={}
        frs=[]
        for e in g:
            frs+=e["frags"]
            for w2,ds in e["words"].items(): allw.setdefault(w2,set()).update(ds)
        if not allw:
            merged["?"+frs[0]]={"frags":frs,"word_forms":[],"docs":set()}; continue
        # keep only forms supported by at least half of the group's raw fragments
        rawfr=[]
        for e in g: rawfr+=e.get("raw",[])
        def nsup(w2):
            c=0
            for f2 in rawfr:
                lead=f2.startswith(" "); trail=f2.endswith(" "); st=f2.strip()
                ok=(w2==st) if (lead and trail) else (w2.startswith(st) if lead else (w2.endswith(st) if trail else (st in w2)))
                c+=ok
            return c
        thr=max(1,(len(rawfr)+1)//2)
        keep={w2:d2 for w2,d2 in allw.items() if nsup(w2)>=thr}
        if not keep: keep=allw
        rep=max(keep,key=lambda w2:(len(keep[w2]),-len(w2)))
        docs=set().union(*keep.values())
        forms=sorted(keep,key=lambda w2:-len(keep[w2]))[:4]
        merged[rep]={"frags":frs,"word_forms":forms,"docs":docs}
    out=[]
    for key,v in merged.items():
        df=len(v["docs"])
        rec={"term":key.lstrip("?"),"word_forms":v["word_forms"],"df":df}
        if df<=3:
            rec["source_docs"]=[{"doc":int(d),"title":meta[d]["title"][:70]} for d in sorted(v["docs"])]
        out.append(rec)
    return out

pt=np.array([[args.x,args.y]])
w=sph_weights(pt,P,h_mode="knn_adaptive",knn_k=8)[0]
v=w@Xl2; v/=np.linalg.norm(v); pi=w@Xl1; pi/=pi.sum()
kl=pi*np.log(np.maximum(pi,1e-300)/np.maximum(ml1,1e-300))
L2=lens_words(pick_idx(v-md)); L1=lens_words(pick_idx(kl))
E={"query_location":[args.x,args.y],
   "corpus":"100 papers, carbon-fiber composite research (single laboratory)",
   "how_produced":"deterministic pipeline; feature selection on character n-grams, then each selected n-gram resolved to the words containing it in the corpus text; terms below are WORDS, df = number of corpus documents containing the word",
   "theme_lens_L2":L2,"concentration_lens_L1":L1,
   "contributing_documents_top3":[{"doc":int(i),"weight":round(float(w[i]),3),"title":meta[i]["title"][:70]}
        for i in np.argsort(w)[::-1][:3]],
   "mixture_entropy_H":round(float(sph_entropy(w[None],100)[0]),3),
   "effective_contributing_documents":round(float(100**sph_entropy(w[None],100)[0]),1),
   "relative_uncertainty_u":round(float(gp.rel_uncertainty(pt)[0]),3)}
outfile = args.outfile or f"evidence_point_{args.x}_{args.y}.json"
json.dump(E, open(outfile, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"query      : ({args.x}, {args.y})")
print(f"u          : {E['relative_uncertainty_u']}")
print(f"entropy H  : {E['mixture_entropy_H']}  (effective {E['effective_contributing_documents']} documents)")
print(f"L2 lens    : {', '.join(e['term'] for e in E['theme_lens_L2'])}")
print(f"L1 lens    : {', '.join(e['term'] for e in E['concentration_lens_L1'])}")
print(f"written    : {outfile}")
