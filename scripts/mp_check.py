"""Material-preserving projection check on retrieval (ViT-H + other encoders).
U_L: lighting deviations (lights 0..7, shapes != g), k_L fixed to the k selected earlier for 'light single'.
U_M: material-level deviations (levels 0..4 per (mat, shape, light), shapes != g, lights 0..7).
Ut = orth((I - U_M U_M^T) U_L); remove Ut. k_M chosen by leave-one-light-out on lights 1..7 (held-out 8..11 untouched).
usage: python mp_check.py <feat_dir> <render_xfer_dir> <controls_json_for_kL> <out_json>"""
import sys, json, glob, os, numpy as np
fx, rx, cj, outp = sys.argv[1:5]
M = [json.loads(l) for f in sorted(glob.glob(os.path.join(rx, "meta_*.jsonl"))) for l in open(f)]
key = {(m["mat"], m["geom"], m["light_id"], m["level"]): m["file"] for m in M}
mats = sorted({m["mat"] for m in M}); geoms = sorted({m["geom"] for m in M})
nz = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True); KM = [0, 2, 4, 8, 16, 32]
def svdb(R, k): R = R - R.mean(0); return np.linalg.svd(R, full_matrices=False)[2][:k].T
res = {}
for f in sorted(glob.glob(os.path.join(fx, "*.npz"))):
    name = os.path.basename(f)[:-4]; z = np.load(f); F = nz(z["feat"]); idx = {n: i for i, n in enumerate(z["files"])}
    fe = lambda m, g, l, j: F[idx[key[(m, g, l, j)]]]
    kL = 32
    def bases(g, lights):
        RL, RM = [], []
        for m in mats:
            for h in geoms:
                if h == g: continue
                for j in range(5):
                    X = np.stack([fe(m, h, l, j) for l in lights]); RL.append(X - X.mean(0))
                for l in lights:
                    X = np.stack([fe(m, h, l, j) for j in range(5)]); RM.append(X - X.mean(0))
        return svdb(np.concatenate(RL), kL), svdb(np.concatenate(RM), max(KM))
    def mk(UL, UM, km):
        if km == 0: return UL
        Q = UL - UM[:, :km] @ (UM[:, :km].T @ UL); return np.linalg.qr(Q)[0]
    def ev(Uf, qlights, cl):
        e = []
        for m in mats:
            for g in geoms:
                U = Uf(g); oth = [h for h in geoms if h != g]; pj = lambda X: nz(X - (X @ U) @ U.T) if U is not None else X
                for L in qlights:
                    C = pj(nz(np.stack([np.mean([fe(m, h, l, j) for h in oth for l in cl(L)], 0) for j in range(5)])))
                    for kk in range(5): e.append(abs(int(np.argmax(C @ pj(fe(m, g, L, kk)[None])[0])) - kk))
        return np.array(e)
    out = {}
    for mode, cl in [("single", lambda L: [0])]:
        sel = {}
        for km in KM:
            v = []
            for lo in range(1, 8):
                rest = [l for l in range(8) if l != lo]; B = {g: bases(g, rest) for g in geoms}
                v.append(ev(lambda g: mk(*B[g], km), [lo], cl).mean())
            sel[km] = float(np.mean(v))
        kb = min(sel, key=sel.get); B = {g: bases(g, list(range(8))) for g in geoms}
        base = ev(lambda g: None, range(8, 12), cl); curve = {km: float(ev(lambda g: mk(*B[g], km), range(8, 12), cl).mean()) for km in KM}
        # how much material energy remains in removed subspace
        out[mode] = {"kL": kL, "val": sel, "kM_selected": kb, "none": float(base.mean()), "heldout_by_kM": curve}
    res[name] = out; print(name, out, flush=True)
json.dump(res, open(outp, "w"), indent=1)
