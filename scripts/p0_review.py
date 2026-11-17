"""Reviewer P0 checks on the diagnostic set (reuses E2 features).
 A. factor-wise binary: material vs each lighting factor (360 vs 360), leave-one-shape-out.
 B. magnitude-matched evaluation: 1:1 nearest-neighbour matching of material to lighting pairs on
    (log pix, log lpips) within each test shape, caliper 0.2 SD; probe predictions from the standard binary task.
 C. overlap controls: energy of material diffs in lighting subspace vs global-PCA subspace vs
    view-change (nuisance) subspace, and principal angles between material and lighting subspaces.
usage: python p0_review.py <feat_dir> <pairs.json> <out_json>"""
import sys, json, glob, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
fd, pj, outp = sys.argv[1:4]
P = json.load(open(pj)); rng = np.random.default_rng(0)
ch = np.array([p["changed"] for p in P]); geom = np.array([p["geom"] for p in P])
mag = np.log(np.array([[p["pix"], p["lpips"]] for p in P]) + 1e-6)
LF = ["size", "ratio", "dir", "kelvin"]

def loo(X, y, mask):
    pred = np.full(len(y), -1)
    for g in np.unique(geom):
        tr, te = mask & (geom != g), mask & (geom == g)
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return pred

def rates(y, pred, m):
    yy, pp = y[m], pred[m]
    return {"bal_acc": float(balanced_accuracy_score(yy, pp)),
            "M_to_L": float((pp[yy == 0] == 1).mean()), "L_to_M": float((pp[yy == 1] == 0).mean()), "n": int(m.sum())}

y2 = (ch != "material").astype(int)
# matching (shared across encoders)
Z = (mag - mag.mean(0)) / mag.std(0); matched = np.zeros(len(P), bool)
for g in np.unique(geom):
    mi = np.where((geom == g) & (y2 == 0))[0]; li = list(np.where((geom == g) & (y2 == 1))[0])
    for i in rng.permutation(mi):
        if not li: break
        d = np.linalg.norm(Z[li] - Z[i], axis=1); j = int(np.argmin(d))
        if d[j] < 0.2: matched[i] = True; matched[li.pop(j)] = True
def smd(m):
    a, b = Z[m & (y2 == 0)], Z[m & (y2 == 1)]
    return ((a.mean(0) - b.mean(0)) / np.sqrt((a.var(0) + b.var(0)) / 2)).round(3).tolist()
res = {"matching": {"n_pairs_matched": int(matched.sum() // 2), "smd_before": smd(np.ones(len(P), bool)), "smd_after": smd(matched),
       "matched_by_factor": {f: int((matched & (ch == f)).sum()) for f in LF}}}

def run(name, X):
    r = {"factorwise": {}}
    for f in LF:
        m = (ch == "material") | (ch == f); pr = loo(X, y2, m); r["factorwise"][f] = rates(y2, pr, m)
    allm = np.ones(len(P), bool); pr = loo(X, y2, allm)
    r["binary_all"] = rates(y2, pr, allm); r["binary_matched"] = rates(y2, pr, matched)
    # also retrain on matched only
    pr2 = loo(X, y2, matched); r["binary_matched_retrained"] = rates(y2, pr2, matched)
    return r

res["magnitude_only"] = run("mag", mag)
for f in sorted(glob.glob(os.path.join(fd, "*.npz"))):
    name = os.path.basename(f)[:-4]; z = np.load(f)
    F = z["feat"] / np.linalg.norm(z["feat"], axis=1, keepdims=True); idx = {n: i for i, n in enumerate(z["files"])}
    D = np.stack([F[idx[p["file"]]] - F[idx[p["base_file"]]] for p in P])
    r = run(name, D)
    # C. overlap controls, k=64, leave-one-shape-out
    k = 64; E = {s: [] for s in ["light", "global", "view", "material_self"]}; EL = {s: [] for s in ["global", "view"]}; ang = {"mat_light": [], "mat_global": [], "light_global": []}
    base = {}
    for p in P: base[(p["mat"], p["geom"], p["color"], p["view"])] = p["base_file"]
    def B(X):
        X = X - X.mean(0); return np.linalg.svd(X, full_matrices=False)[2][:k].T
    for g in np.unique(geom):
        tr = geom != g; te = geom == g
        UL = B(D[tr & (y2 == 1)]); UM = B(D[tr & (y2 == 0)])
        files = sorted({base[key] for key in base if key[1] != g}); UG = B(F[[idx[x] for x in files]])
        # view-change differences of base images (same mat, shape, color; different views)
        V = []
        keys = [key for key in base if key[1] != g]
        grp = {}
        for key in keys: grp.setdefault(key[:3], []).append(key)
        for ks in grp.values():
            ks = sorted(ks, key=lambda t: t[3])
            for a in range(len(ks)):
                for b in range(a + 1, len(ks)): V.append(F[idx[base[ks[b]]]] - F[idx[base[ks[a]]]])
        UV = B(np.stack(V))
        en = lambda Dm, U: (((Dm @ U) ** 2).sum(1) / (Dm ** 2).sum(1)).tolist()
        Dm, Dl = D[te & (y2 == 0)], D[te & (y2 == 1)]
        E["light"] += en(Dm, UL); E["global"] += en(Dm, UG); E["view"] += en(Dm, UV); E["material_self"] += en(Dm, UM)
        EL["global"] += en(Dl, UG); EL["view"] += en(Dl, UV)
        cos = lambda A, Bm: np.linalg.svd(A.T @ Bm, compute_uv=False)
        ang["mat_light"].append(cos(UM, UL).mean()); ang["mat_global"].append(cos(UM, UG).mean()); ang["light_global"].append(cos(UL, UG).mean())
    r["overlap_k64"] = {"material_energy_in": {s: float(np.mean(v)) for s, v in E.items()},
                        "lighting_energy_in": {s: float(np.mean(v)) for s, v in EL.items()},
                        "mean_cos_principal_angles": {s: float(np.mean(v)) for s, v in ang.items()},
                        "random": k / D.shape[1], "n_view_diffs": len(V)}
    res[name] = r
    print(name, json.dumps({f: (round(v["M_to_L"], 3), round(v["L_to_M"], 3), round(v["bal_acc"], 3)) for f, v in r["factorwise"].items()}),
          "matched", {k2: round(v, 3) for k2, v in r["binary_matched"].items()}, "retr", {k2: round(v, 3) for k2, v in r["binary_matched_retrained"].items()},
          "overlap", json.dumps(r["overlap_k64"], default=float)[:400], flush=True)
print("mag", json.dumps(res["magnitude_only"], default=float)[:900]); print("matching", res["matching"])
json.dump(res, open(outp, "w"), indent=1, default=float)
