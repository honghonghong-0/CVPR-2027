"""E2: change attribution from feature differences, with magnitude-only baseline and stratification.

usage: python analyze_e2.py <feat_dir> <pairs.json> <out_json>
Label (5-way): material | size | ratio | dir | kelvin.  Also binary: material vs lighting.
Split: leave-one-geometry-out (train on 2 shapes, test on the 3rd), so shape cannot leak.
Features: d = normalize(f_variant) - normalize(f_base).
Baseline: logistic regression on [pix, lpips] only (how much changed, not what).
Stratified: test pairs binned by LPIPS tercile (bins fixed on all pairs); balanced accuracy per bin.
"""
import sys, json, glob, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

fd, pj, outp = sys.argv[1:4]
P = json.load(open(pj))
CLS = ["material", "size", "ratio", "dir", "kelvin"]
y5 = np.array([CLS.index(p["changed"]) for p in P])
y2 = (y5 > 0).astype(int)  # 0 material, 1 lighting
geom = np.array([p["geom"] for p in P])
mag = np.array([[p["pix"], p["lpips"]] for p in P])
edges = np.quantile(mag[:, 1], [1 / 3, 2 / 3])
mbin = np.digitize(mag[:, 1], edges)

def loo(X, y, n_cls):
    pred = np.zeros_like(y)
    for g in np.unique(geom):
        tr, te = geom != g, geom == g
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return pred

def summarize(y, pred, n_cls):
    cm = confusion_matrix(y, pred, labels=list(range(n_cls)), normalize="true")
    r = {"bal_acc": float(balanced_accuracy_score(y, pred)), "cm": cm.round(3).tolist(),
         "by_mag_bin": [float(balanced_accuracy_score(y[mbin == b], pred[mbin == b]))
                        if len(np.unique(y[mbin == b])) > 1 else None for b in range(3)]}
    return r

res = {"n_pairs": len(P), "lpips_tercile_edges": edges.tolist(),
       "class_counts": {c: int((y5 == i).sum()) for i, c in enumerate(CLS)}}
# material vs lighting share per magnitude bin (how confounded is each bin)
res["material_share_by_bin"] = [float((y2[mbin == b] == 0).mean()) for b in range(3)]
for tag, X in [("magnitude_only", mag)]:
    res[tag] = {"5way": summarize(y5, loo(X, y5, 5), 5), "binary": summarize(y2, loo(X, y2, 2), 2)}
for f in sorted(glob.glob(os.path.join(fd, "*.npz"))):
    z = np.load(f); idx = {n: i for i, n in enumerate(z["files"])}
    F = z["feat"] / np.linalg.norm(z["feat"], axis=1, keepdims=True)
    D = np.stack([F[idx[p["file"]]] - F[idx[p["base_file"]]] for p in P])
    name = os.path.basename(f)[:-4]
    r5, r2 = summarize(y5, loo(D, y5, 5), 5), summarize(y2, p2 := loo(D, y2, 2), 2)
    # asymmetry: P(lighting|material) vs P(material|lighting)
    cm2 = np.array(r2["cm"]); r2["P_light_given_mat"] = cm2[0, 1]; r2["P_mat_given_light"] = cm2[1, 0]
    # per-material breakdown of material->lighting error
    r2["mat_to_light_by_material"] = {m: float((p2[(y2 == 0) & (np.array([p["mat"] for p in P]) == m)] == 1).mean())
                                      for m in sorted({p["mat"] for p in P})}
    res[name] = {"5way": r5, "binary": r2}
    print(name, "5way %.3f" % r5["bal_acc"], "bin %.3f" % r2["bal_acc"], "bins", r2["by_mag_bin"])
print("magnitude_only 5way %.3f bin %.3f" % (res["magnitude_only"]["5way"]["bal_acc"], res["magnitude_only"]["binary"]["bal_acc"]))
json.dump(res, open(outp, "w"), indent=1)
