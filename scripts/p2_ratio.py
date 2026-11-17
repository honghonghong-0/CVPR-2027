"""(a) material vs ratio only: 1:1 magnitude matching (same procedure as p0_review, caliper 0.2 SD), probe = shape-disjoint
    material-vs-ratio probe trained on all 720 pairs (matching uses magnitudes only, no model selection), plus retrained-on-matched.
(b) cluster bootstrap CI of directional error difference (M->F minus F->M) per factor, resampling material x shape clusters.
usage: python p2_ratio.py <out_json>"""
import sys, json, glob, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
P = json.load(open("results/pairs_diag.json")); ch = np.array([p["changed"] for p in P]); geom = np.array([p["geom"] for p in P]); mat = np.array([p["mat"] for p in P])
mag = np.log(np.array([[p["pix"], p["lpips"]] for p in P]) + 1e-6); y = (ch != "material").astype(int)
def loo(X, m):
    pr = np.full(len(P), -1)
    for g in np.unique(geom):
        tr, te = m & (geom != g), m & (geom == g); sc = StandardScaler().fit(X[tr])
        pr[te] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return pr
def rates(pr, m): return {"bal_acc": float(balanced_accuracy_score(y[m], pr[m])), "M_to_L": float((pr[m & (y == 0)] == 1).mean()), "L_to_M": float((pr[m & (y == 1)] == 0).mean()), "n": int(m.sum())}
res = {}; rng = np.random.default_rng(0)
sub = (ch == "material") | (ch == "ratio"); Z = (mag - mag[sub].mean(0)) / mag[sub].std(0); matched = np.zeros(len(P), bool)
for g in np.unique(geom):
    mi = np.where(sub & (geom == g) & (y == 0))[0]; li = list(np.where(sub & (geom == g) & (y == 1))[0])
    for i in rng.permutation(mi):
        if not li: break
        d = np.linalg.norm(Z[li] - Z[i], axis=1); j = int(np.argmin(d))
        if d[j] < 0.2: matched[i] = True; matched[li.pop(j)] = True
smd = lambda m: ((Z[m & (y == 0)].mean(0) - Z[m & (y == 1)].mean(0)) / np.sqrt((Z[m & (y == 0)].var(0) + Z[m & (y == 1)].var(0)) / 2)).round(3).tolist()
res["ratio_matching"] = {"n_pairs": int(matched.sum() // 2), "smd_before": smd(sub), "smd_after": smd(matched),
    "median_lpips_material": float(np.median([P[i]["lpips"] for i in np.where(ch == "material")[0]])), "median_lpips_ratio": float(np.median([P[i]["lpips"] for i in np.where(ch == "ratio")[0]])),
    "matched_by_material": {m: int((matched & (mat == m) & (y == 0)).sum()) for m in sorted(set(mat))}}
res["ratio_matching"]["magnitude_only"] = rates(loo(mag, sub), matched)
LF = ["size", "ratio", "dir", "kelvin"]; cl = sorted({(a, b) for a, b in zip(mat, geom)})
for f in sorted(glob.glob("feats_diag/*.npz")):
    z = np.load(f); F = z["feat"] / np.linalg.norm(z["feat"], axis=1, keepdims=True); idx = {n: i for i, n in enumerate(z["files"])}
    D = np.stack([F[idx[p["file"]]] - F[idx[p["base_file"]]] for p in P]); name = os.path.basename(f)[:-4]; r = {}
    pr = loo(D, sub); r["ratio_all"] = rates(pr, sub); r["ratio_matched"] = rates(pr, matched); r["ratio_matched_retrained"] = rates(loo(D, matched), matched)
    r["ci"] = {}
    for fac in LF:
        m = (ch == "material") | (ch == fac); p2 = loo(D, m)
        ci_idx = {c: np.where(m & (mat == c[0]) & (geom == c[1]))[0] for c in cl}
        def diff(ids):
            a = ids[y[ids] == 0]; b = ids[y[ids] == 1]; return (p2[a] == 1).mean() - (p2[b] == 0).mean()
        bs = [diff(np.concatenate([ci_idx[cl[k]] for k in rng.integers(0, len(cl), len(cl))])) for _ in range(5000)]
        r["ci"][fac] = {"diff": float(diff(np.where(m)[0])), "cl95": np.percentile(bs, [2.5, 97.5]).round(3).tolist()}
    res[name] = r
    print(name, "ratio matched", {k: round(v, 3) for k, v in r["ratio_matched"].items()}, "retr", {k: round(v, 3) for k, v in r["ratio_matched_retrained"].items()}, "CI", {k: (round(v["diff"], 3), v["cl95"]) for k, v in r["ci"].items()}, flush=True)
print(res["ratio_matching"]); json.dump(res, open(sys.argv[1], "w"), indent=1)
