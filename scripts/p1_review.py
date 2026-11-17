"""P1 analyses (CPU, reuse features/results).
 1. leave-one-material-family-out binary attribution (diag feats).
 2. per-material / per-shape breakdown of projection gains (from controls_proj.json light_single errs if present, else recompute via xfer_proj.json).
 3. ZeST: share of ViT-H material-level variation removed by the lighting subspace (k=32) vs random; per-material ZeST deltas.
usage: python p1_review.py <out_json>"""
import sys, json, glob, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
outp = sys.argv[1]; res = {}
P = json.load(open("results/pairs_diag.json")); ch = np.array([p["changed"] for p in P]); mat = np.array([p["mat"] for p in P]); geom = np.array([p["geom"] for p in P])
y = (ch != "material").astype(int); mats = sorted(set(mat))
res["lomo"] = {}
for f in sorted(glob.glob("feats_diag/*.npz")):
    z = np.load(f); F = z["feat"] / np.linalg.norm(z["feat"], axis=1, keepdims=True); idx = {n: i for i, n in enumerate(z["files"])}
    D = np.stack([F[idx[p["file"]]] - F[idx[p["base_file"]]] for p in P]); name = os.path.basename(f)[:-4]; r = {}
    for m in mats:
        # held-out family m; train on other families; test on m (material pairs of m + lighting pairs of m), also shape-disjoint
        pred = np.full(len(P), -1)
        for g in np.unique(geom):
            tr = (mat != m) & (geom != g); te = (mat == m) & (geom == g)
            sc = StandardScaler().fit(D[tr]); c = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(D[tr]), y[tr])
            pred[te] = c.predict(sc.transform(D[te]))
        t = mat == m
        r[m] = {"bal_acc": float(balanced_accuracy_score(y[t], pred[t])), "M_to_L": float((pred[t & (y == 0)] == 1).mean()), "L_to_M": float((pred[t & (y == 1)] == 0).mean())}
    r["mean"] = {k: float(np.mean([r[m][k] for m in mats])) for k in ["bal_acc", "M_to_L", "L_to_M"]}
    res["lomo"][name] = r; print("lomo", name, {k: round(v, 3) for k, v in r["mean"].items()}, flush=True)
# 2. breakdown
xp = json.load(open("results/xfer_proj_withqwen.json")) if os.path.exists("results/xfer_proj_withqwen.json") else json.load(open("results/xfer_proj.json"))
M = [json.loads(l) for f in sorted(glob.glob("renders_xfer/meta_*.jsonl")) for l in open(f)]
xm = sorted({m["mat"] for m in M}); xg = sorted({m["geom"] for m in M})
order = [(mt, g, L, kk) for mt in xm for g in xg for L in range(8, 12) for kk in range(5)]
res["breakdown"] = {}
for enc, d in xp.items():
    c = d["xfer_single"]; e0 = np.array(c["heldout_curve"]["0"]["errs"] if "0" in c["heldout_curve"] else c["heldout_curve"][0]["errs"])
    ks = str(c["k_selected"]); e1 = np.array(c["heldout_curve"][ks]["errs"] if ks in c["heldout_curve"] else c["heldout_curve"][int(ks)]["errs"])
    assert len(e0) == len(order)
    bm = {mt: [float(e0[[i for i, o in enumerate(order) if o[0] == mt]].mean()), float(e1[[i for i, o in enumerate(order) if o[0] == mt]].mean())] for mt in xm}
    bg = {g: [float(e0[[i for i, o in enumerate(order) if o[1] == g]].mean()), float(e1[[i for i, o in enumerate(order) if o[1] == g]].mean())] for g in xg}
    res["breakdown"][enc] = {"by_material": bm, "by_shape": bg, "n_mat_improved": sum(v[1] < v[0] for v in bm.values()), "n_shape_improved": sum(v[1] < v[0] for v in bg.values())}
    print("breakdown", enc, {k: [round(a, 2) for a in v] for k, v in bm.items()}, {k: [round(a, 2) for a in v] for k, v in bg.items()}, flush=True)
# 3. ZeST: material info removed
z = np.load("feats_vith_only/openclip_vith.npz"); F = z["feat"] / np.linalg.norm(z["feat"], axis=1, keepdims=True); idx = {n: i for i, n in enumerate(z["files"])}
key = {(m["mat"], m["geom"], m["light_id"], m["level"]): m["file"] for m in M}
fe = lambda m, g, l, j: F[idx[key[(m, g, l, j)]]]
rows = {"mat_level": [], "mat_level_rand": [], "light": []}; rng = np.random.default_rng(0)
for g in xg:
    R = []
    for m in xm:
        for h in xg:
            if h == g: continue
            for j in range(5):
                X = np.stack([fe(m, h, l, j) for l in range(8)]); R.append(X - X.mean(0))
    Dd = np.concatenate(R); Dd -= Dd.mean(0); U = np.linalg.svd(Dd, full_matrices=False)[2][:32].T
    UR = np.linalg.qr(rng.standard_normal((F.shape[1], 32)))[0]
    for m in xm:
        for L in range(8, 12):
            X = np.stack([fe(m, g, L, j) for j in range(5)]); X = X - X.mean(0); tot = (X ** 2).sum()
            rows["mat_level"].append(float(((X @ U) ** 2).sum() / tot)); rows["mat_level_rand"].append(float(((X @ UR) ** 2).sum() / tot))
        for j in range(5):
            X = np.stack([fe(m, g, L, j) for L in range(8, 12)]); X = X - X.mean(0); rows["light"].append(float(((X @ U) ** 2).sum() / (X ** 2).sum()))
res["zest_vith_energy_k32"] = {k: float(np.mean(v)) for k, v in rows.items()}
print("vith energy", res["zest_vith_energy_k32"], flush=True)
ze = json.load(open("results/zest_eval.json")); rr = ze["rows"]; zb = {}
for mt in xm:
    s = [r for r in rr if r["tag"].startswith(mt + "_")]
    zb[mt] = {k: float(np.mean([r["light"][k] - r["raw"][k] for r in s])) for k in ["lpips_gt", "color_gt", "level_err", "leak"]}
res["zest_by_material"] = zb; print("zest by mat", {m: {k: round(v, 3) for k, v in d.items()} for m, d in zb.items()})
json.dump(res, open(outp, "w"), indent=1)
