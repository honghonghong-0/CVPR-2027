"""Application: recover the material parameter level of a reference render under unseen lighting.

usage: python analyze_xfer.py <feat_dir> <render_dir> <out_json>
For each (material, geometry): references = renders at level k under held-out lights (8..11).
Candidates for each level j:
  single   : feature of level j rendered under one matching light (L0)  -- naive, lighting baked in
  marginal : mean feature of level j over the 8 matching lights           -- lighting-marginalized
  oracle   : level j rendered under the SAME light as the reference      -- upper bound
All candidates come from the OTHER geometries (cross-shape), so the query render is never in the
candidate pool (asserted). Earlier same-shape oracle was a self-match (acc 1.0 by construction).
Prediction = argmax cosine similarity. Metrics: exact accuracy, mean |level error| (levels 0..4).
Also pixel-space (downsampled RGB) as a feature-free reference.
"""
import sys, json, glob, os, numpy as np
from PIL import Image

fd, rd, outp = sys.argv[1:4]
M = [json.loads(l) for f in sorted(glob.glob(os.path.join(rd, "meta_*.jsonl"))) for l in open(f)]
key = {(m["mat"], m["geom"], m["light_id"], m["level"]): m["file"] for m in M}
mats = sorted({m["mat"] for m in M}); geoms = sorted({m["geom"] for m in M})

def run(F, idx):
    out = {}
    for mode in ["single", "marginal", "oracle"]:
        errs = []; per_mat = {}
        for mt in mats:
            e_m = []
            for g in geoms:
                others = [h for h in geoms if h != g]  # candidates never come from the query's own shape
                def f(gg, l, k): return F[idx[key[(mt, gg, l, k)]]]
                def cand(ls, j): return np.mean([f(h, l, j) for h in others for l in ls], 0)
                for L in range(8, 12):
                    for k in range(5):
                        q = f(g, L, k)
                        used = {key[(mt, h, l, j)] for h in others for l in range(12) for j in range(5)}
                        assert key[(mt, g, L, k)] not in used  # no self-match: query image cannot be a candidate
                        if mode == "single": C = np.stack([cand([0], j) for j in range(5)])
                        elif mode == "marginal": C = np.stack([cand(range(8), j) for j in range(5)])
                        else: C = np.stack([cand([L], j) for j in range(5)])
                        C = C / np.linalg.norm(C, axis=1, keepdims=True)
                        e_m.append(abs(int(np.argmax(C @ q)) - k))
            per_mat[mt] = float(np.mean(e_m)); errs += e_m
        errs = np.array(errs)
        out[mode] = {"acc": float((errs == 0).mean()), "mae": float(errs.mean()), "mae_by_material": per_mat}
    return out

res = {}
files = sorted(os.path.basename(m["file"]) for m in M)
pix = np.stack([np.asarray(Image.open(os.path.join(rd, f)).convert("RGB").resize((64, 64)), np.float32).ravel() for f in files])
pix = pix - pix.mean(1, keepdims=True); pix /= np.linalg.norm(pix, axis=1, keepdims=True)
res["pixels"] = run(pix, {f: i for i, f in enumerate(files)})
for f in sorted(glob.glob(os.path.join(fd, "*.npz"))):
    z = np.load(f); F = z["feat"] / np.linalg.norm(z["feat"], axis=1, keepdims=True)
    res[os.path.basename(f)[:-4]] = run(F, {n: i for i, n in enumerate(z["files"])})
for k, v in res.items():
    print(k.ljust(14), "  ".join("%s acc %.3f mae %.3f" % (m, v[m]["acc"], v[m]["mae"]) for m in v))
json.dump(res, open(outp, "w"), indent=1)
