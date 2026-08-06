# -*- coding: utf-8 -*-
"""§4 連続知識場: SPH補間(L2/L1)・知識勾配・SPHエントロピー・GPR不確かさ.
usage: python3 04_fields.py gpr | grid | grad
"""
import json, os, sys, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmlib import (OUT, DATA, load_tfidf, sph_weights, sph_entropy, L2Field, GPR)

phase = sys.argv[1]
t0 = time.time()
P = np.load(OUT + "/coords.npy")
N = len(P)

if phase == "gpr":
    Z = np.load(DATA + "/svd_scores.npy")[:, :10]     # TruncatedSVD 10次元潜在 (§4.4)
    # 潜在をスケール正規化(数値安定; 逆変換情報を保存)
    zs = Z.std()
    gp = GPR(kernel="rbf", seed=0, n_restarts=10).fit(P, Z / zs)
    pickle.dump({"theta": gp.theta, "X": gp.X, "alpha": gp.alpha, "L": gp.L,
                 "noise": gp.noise, "lml": gp.lml_, "zscale": zs, "kernel": "rbf"},
                open(DATA + "/gpr_model.pkl", "wb"))
    info = {"kernel": "ConstantKernel*RBF(ard lx,ly)+WhiteKernel", "n_restarts": 10,
            "random_state": 0, "svd_dim": 10,
            "constant": float(np.exp(gp.theta[0])), "length_scale_x": float(np.exp(gp.theta[1])),
            "length_scale_y": float(np.exp(gp.theta[2])), "white_noise": float(np.exp(gp.theta[3])),
            "log_marginal_likelihood": gp.lml_,
            "implementation": "custom NumPy GPR (sklearn unavailable); hyperparams by Nelder-Mead multi-restart"}
    json.dump(info, open(OUT + "/gpr_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))

elif phase == "grid":
    X, l2n, l1s = load_tfidf()
    G = np.load(DATA + "/gram_l2.npy")
    fld = L2Field(G, P, h_mode="global")
    n = 41
    xs = np.linspace(-1.05, 1.05, n)
    gx, gy = np.meshgrid(xs, xs)
    GR = np.stack([gx.ravel(), gy.ravel()], 1)
    # SPHエントロピー
    w = sph_weights(GR, P, h_mode="global")
    H = sph_entropy(w, N)
    # GPR不確かさ
    md = pickle.load(open(DATA + "/gpr_model.pkl", "rb"))
    gp = GPR(kernel="rbf"); gp.theta = md["theta"]; gp.X = md["X"]; gp.alpha = md["alpha"]
    gp.L = md["L"]; gp.noise = md["noise"]
    u = gp.rel_uncertainty(GR)
    # 計量テンソル (chunked)
    gt = np.empty((len(GR), 2, 2))
    for i in range(0, len(GR), 200):
        gt[i:i + 200] = fld.metric(GR[i:i + 200], delta=1e-3)
    np.savez(DATA + "/grid_fields.npz", xs=xs, entropy=H.reshape(n, n),
             uncertainty=u.reshape(n, n), g=gt.reshape(n, n, 2, 2))
    print("grid done: H[min,max]=", H.min().round(3), H.max().round(3),
          "u[min,max]=", u.min().round(3), u.max().round(3),
          "tr(g)[min,max]=", (gt[:, 0, 0] + gt[:, 1, 1]).min().round(4),
          (gt[:, 0, 0] + gt[:, 1, 1]).max().round(4), f"{time.time()-t0:.1f}s")

elif phase == "grad":
    # 知識勾配 (4象限±0.5, δ=1e-3): L2場・L1場, top特徴の言語化用データ
    X, l2n, l1s = load_tfidf()
    Xl2 = X / l2n[:, None]
    EPS = 1e-10
    Xl1 = (X + EPS) / l1s[:, None]
    vocab = json.load(open(DATA + "/vocab.json"))
    pts = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])
    delta = 1e-3
    res = []
    for pt in pts:
        entry = {"point": pt.tolist()}
        for name, Xn in [("L2", Xl2), ("L1", Xl1)]:
            dirs = {}
            for ax, e in [("x", np.array([delta, 0])), ("y", np.array([0, delta]))]:
                wp = sph_weights((pt + e)[None], P, h_mode="global")[0]
                wm = sph_weights((pt - e)[None], P, h_mode="global")[0]
                if name == "L2":
                    vp = wp @ Xn; vp /= np.linalg.norm(vp)
                    vm = wm @ Xn; vm /= np.linalg.norm(vm)
                else:
                    vp = wp @ Xn; vm = wm @ Xn
                dv = (vp - vm) / (2 * delta)
                top_pos = np.argsort(dv)[::-1][:10]
                top_neg = np.argsort(dv)[:10]
                dirs[ax] = {"norm": float(np.linalg.norm(dv)),
                            "top_pos": [[vocab[i], float(dv[i])] for i in top_pos],
                            "top_neg": [[vocab[i], float(dv[i])] for i in top_neg],
                            "vec3": None}
                dirs[ax]["_dv"] = dv
            gxv, gyv = dirs["x"].pop("_dv"), dirs["y"].pop("_dv")
            ip = float(gxv @ gyv)
            cs = ip / (np.linalg.norm(gxv) * np.linalg.norm(gyv) + 1e-300)
            entry[name] = {"x": dirs["x"], "y": dirs["y"],
                           "inner_product_xy": ip, "cosine_similarity_xy": float(cs)}
        res.append(entry)
        print("point", pt, "done", f"{time.time()-t0:.1f}s", flush=True)
    json.dump(res, open(OUT + "/knowledge_gradient.json", "w"), indent=1, ensure_ascii=False)

print("elapsed", round(time.time() - t0, 1))
