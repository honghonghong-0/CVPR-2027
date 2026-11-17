"""Specificity controls for lighting-subspace removal (reviewer: is it the *lighting* subspace, or any k-dim removal?).
usage: python controls_proj.py <feat_xfer_dir> <render_xfer_dir> <out_json>
Same cross-shape protocol and same k selection (leave-one-light-out over matching lights 0..7) as analyze_xfer_proj.py.
U sources (all exclude query shape and held-out lights 8..11):
  light   : deviation from mean over lights (per mat, level, shape)   [the method]
  random  : random orthonormal k-dim subspace (5 seeds, averaged)
  pca_all : top-k PCs of all features (generic whitening / dominant-direction removal)
  material: deviation from mean over levels (per mat, light, shape)   [should hurt]
"""
import sys, json, glob, os, numpy as np
fx, rx, outp = sys.argv[1:4]
M = [json.loads(l) for f in sorted(glob.glob(os.path.join(rx, "meta_*.jsonl"))) for l in open(f)]
key = {(m["mat"], m["geom"], m["light_id"], m["level"]): m["file"] for m in M}
mats = sorted({m["mat"] for m in M}); geoms = sorted({m["geom"] for m in M})
KS = [0, 1, 2, 4, 8, 16, 32, 64]
nz = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
def topk(D, k): D = D - D.mean(0); return np.linalg.svd(D, full_matrices=False)[2][:k].T
def proj(X, U): return X if U is None else nz(X - (X @ U) @ U.T)
def evaluate(F, idx, Ufn, k, qL, cl):
    e = []
    for mt in mats:
        for g in geoms:
            oth = [h for h in geoms if h != g]; U = None if k == 0 else Ufn(g, k)
            f = lambda gg, l, j: F[idx[key[(mt, gg, l, j)]]]
            for L in qL:
                C = proj(nz(np.stack([np.mean([f(h, l, j) for h in oth for l in cl(L)], 0) for j in range(5)])), U)
                for kk in range(5): e.append(abs(int(np.argmax(C @ proj(f(g, L, kk)[None], U)[0])) - kk))
    return np.array(e)
res = {}
for fp in sorted(glob.glob(os.path.join(fx, "*.npz"))):
    name = os.path.basename(fp)[:-4]; z = np.load(fp); F = nz(z["feat"]); idx = {n: i for i, n in enumerate(z["files"])}; Dm = F.shape[1]
    cache = {}
    def build(src, g, lights, seed=0):
        c = (src, g, lights, seed)
        if c in cache: return cache[c]
        R = []
        for mt in mats:
            for h in geoms:
                if h == g: continue
                if src == "light":
                    for j in range(5): X = np.stack([F[idx[key[(mt, h, l, j)]]] for l in lights]); R.append(X - X.mean(0))
                elif src == "material":
                    for l in lights: X = np.stack([F[idx[key[(mt, h, l, j)]]] for j in range(5)]); R.append(X - X.mean(0))
                elif src == "pca_all":
                    R.append(np.stack([F[idx[key[(mt, h, l, j)]]] for l in lights for j in range(5)]))
        if src == "random":
            U = np.linalg.qr(np.random.default_rng(seed).standard_normal((Dm, max(KS))))[0]
        else: U = topk(np.concatenate(R), max(KS))
        cache[c] = U; return U
    out = {}
    for src in ["light", "random", "pca_all", "material"]:
        seeds = range(5) if src == "random" else [0]
        for mode, cl in [("single", lambda L: [0]), ("marginal", lambda L: range(8))]:
            per_seed = []
            for s in seeds:
                sel = {}
                for k in KS:
                    v = []
                    for lo in range(1, 8):
                        rest = tuple(l for l in range(8) if l != lo)
                        Uf = lambda g, k, rest=rest: build(src, g, rest, s)[:, :k]
                        v.append(evaluate(F, idx, Uf, k, [lo], (lambda L: [0]) if mode == "single" else (lambda L, rest=rest: rest)).mean())
                    sel[k] = float(np.mean(v))
                kb = min(sel, key=sel.get)
                Uf = lambda g, k: build(src, g, tuple(range(8)), s)[:, :k]
                e = evaluate(F, idx, Uf, kb, range(8, 12), cl); e0 = evaluate(F, idx, Uf, 0, range(8, 12), cl)
                per_seed.append({"k": kb, "mae": float(e.mean()), "mae_k0": float(e0.mean()), "errs": e.tolist(), "errs_k0": e0.tolist()})
            out[f"{src}_{mode}"] = per_seed
            print(name, src, mode, "k*", [p["k"] for p in per_seed], "mae %.3f -> %.3f" % (per_seed[0]["mae_k0"], np.mean([p["mae"] for p in per_seed])), flush=True)
    res[name] = out
json.dump(res, open(outp, "w"))
