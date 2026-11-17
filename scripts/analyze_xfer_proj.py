"""Diagnosis-driven method: remove the lighting subspace from features before recovering material level.

usage: python analyze_xfer_proj.py <feat_xfer_dir> <render_xfer_dir> <feat_diag_dir> <pairs_diag.json> <out_json>
f' = (I - U U^T) f, U = top-k principal directions of lighting-induced feature change.
Two sources of U (neither touches held-out lights 8..11 nor the query's shape):
  diag : differences of lighting-only minimal pairs from the diagnosis set (E2)
  xfer : per (mat, level, other-shape) deviation from the mean over matching lights 0..7
Evaluation is the cross-shape protocol of analyze_xfer.py (query shape never in candidates or in U).
k is chosen on matching lights only (leave-one-light-out over 0..7), never on held-out lights.
"""
import sys, json, glob, os, numpy as np
fx, rx, fd, pj, outp = sys.argv[1:6]
M = [json.loads(l) for f in sorted(glob.glob(os.path.join(rx, "meta_*.jsonl"))) for l in open(f)]
key = {(m["mat"], m["geom"], m["light_id"], m["level"]): m["file"] for m in M}
mats = sorted({m["mat"] for m in M}); geoms = sorted({m["geom"] for m in M})
P = json.load(open(pj)); KS = [0, 1, 2, 4, 8, 16, 32, 64]
nz = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)

def basis(D, k):
    if k == 0: return None
    D = D - D.mean(0); _, _, Vt = np.linalg.svd(D, full_matrices=False); return Vt[:k].T

def proj(X, U): return X if U is None else nz(X - (X @ U) @ U.T)

def evaluate(F, idx, Ufn, k, mode, query_lights, cand_lights_fn):
    errs = []
    for mt in mats:
        for g in geoms:
            others = [h for h in geoms if h != g]; U = Ufn(g, k)
            f = lambda gg, l, j: F[idx[key[(mt, gg, l, j)]]]
            for L in query_lights:
                C = np.stack([np.mean([f(h, l, j) for h in others for l in cand_lights_fn(L)], 0) for j in range(5)])
                C = proj(nz(C), U)
                for kk in range(5):
                    q = proj(f(g, L, kk)[None], U)[0]
                    errs.append(abs(int(np.argmax(C @ q)) - kk))
    e = np.array(errs); return {"acc": float((e == 0).mean()), "mae": float(e.mean()), "errs": e.tolist()}

res = {}
zd = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(os.path.join(fd, "*.npz"))}
for f in sorted(glob.glob(os.path.join(fx, "*.npz"))):
    name = os.path.basename(f)[:-4]; z = np.load(f)
    F = nz(z["feat"]); idx = {n: i for i, n in enumerate(z["files"])}
    # U from diagnosis lighting pairs (different renders entirely)
    Z = zd[name]; Fd = nz(Z["feat"]); di = {n: i for i, n in enumerate(Z["files"])}
    Dd = np.stack([Fd[di[p["file"]]] - Fd[di[p["base_file"]]] for p in P if p["changed"] != "material"])
    Ud = {k: basis(Dd, k) for k in KS}
    # U from xfer matching lights, excluding query shape (and, during k-selection, the left-out light)
    def Ux_build(g, lights):
        R = []
        for mt in mats:
            for h in geoms:
                if h == g: continue
                for j in range(5):
                    X = np.stack([F[idx[key[(mt, h, l, j)]]] for l in lights]); R.append(X - X.mean(0))
        return np.concatenate(R)
    cache = {}
    def Ux(g, k, lights=tuple(range(8))):
        if k == 0: return None
        c = (g, lights)
        if c not in cache: cache[c] = basis(Ux_build(g, list(lights)), max(KS))
        return cache[c][:, :k]
    out = {}
    for src in ["diag", "xfer"]:
        for mode, cl in [("single", lambda L: [0]), ("marginal", lambda L: range(8))]:
            # k selection on matching lights: query light l in 1..7, candidates exclude it
            sel = {}
            for k in KS:
                e = []
                for lo in range(1, 8):
                    rest = tuple(l for l in range(8) if l != lo)
                    Uf = (lambda g, k: Ud[k]) if src == "diag" else (lambda g, k, rest=rest: Ux(g, k, rest))
                    cand = (lambda L: [0]) if mode == "single" else (lambda L, rest=rest: rest)
                    e.append(evaluate(F, idx, Uf, k, mode, [lo], cand)["mae"])
                sel[k] = float(np.mean(e))
            kbest = min(sel, key=sel.get)
            Uf = (lambda g, k: Ud[k]) if src == "diag" else (lambda g, k: Ux(g, k))
            curve = {k: evaluate(F, idx, Uf, k, mode, range(8, 12), cl) for k in KS}
            out[f"{src}_{mode}"] = {"k_selected": kbest, "val_mae": sel, "heldout_at_k": curve[kbest], "heldout_curve": curve}
            print(name, src, mode, "k*", kbest, "k0 mae %.3f" % curve[0]["mae"], "k* mae %.3f acc %.3f" % (curve[kbest]["mae"], curve[kbest]["acc"]),
                  "curve", " ".join("%d:%.2f" % (k, curve[k]["mae"]) for k in KS), flush=True)
    res[name] = out
json.dump(res, open(outp, "w"), indent=1)
