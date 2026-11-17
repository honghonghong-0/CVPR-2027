"""Does the shared lighting subspace *cause* the asymmetry? Intervention on the E2 probe.
usage: python asym_intervention.py <feat_dir> <pairs.json> <out_json>
For each held-out geometry: U_k = top-k PCs of lighting-pair diffs from the 2 training geometries only.
Project all diffs d -> (I-UU^T)d, retrain the binary probe, measure P(light|mat) and P(mat|light).
Control: random k-dim U (5 seeds). Also per-pair link: energy share of material diffs in U_64 vs probe error.
"""
import sys, json, glob, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
fd, pj, outp = sys.argv[1:4]
P = json.load(open(pj)); y = np.array([0 if p["changed"] == "material" else 1 for p in P]); geom = np.array([p["geom"] for p in P])
KS = [0, 4, 16, 64]
def run(D, Ufn):
    pred = np.zeros_like(y); share = np.zeros(len(y))
    for g in np.unique(geom):
        tr, te = geom != g, geom == g; U = Ufn(D, tr)
        X = D if U is None else D - (D @ U) @ U.T
        if U is not None: share[te] = ((D[te] @ U) ** 2).sum(1) / (D[te] ** 2).sum(1)
        sc = StandardScaler().fit(X[tr]); pred[te] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return pred, share
def rates(pred): return {"P_light_given_mat": float((pred[y == 0] == 1).mean()), "P_mat_given_light": float((pred[y == 1] == 0).mean())}
res = {}
for f in sorted(glob.glob(os.path.join(fd, "*.npz"))):
    name = os.path.basename(f)[:-4]; z = np.load(f); idx = {n: i for i, n in enumerate(z["files"])}
    F = z["feat"] / np.linalg.norm(z["feat"], axis=1, keepdims=True)
    D = np.stack([F[idx[p["file"]]] - F[idx[p["base_file"]]] for p in P]); out = {}
    for k in KS:
        def Ul(D, tr, k=k):
            if k == 0: return None
            L = D[tr & (y == 1)]; L = L - L.mean(0); return np.linalg.svd(L, full_matrices=False)[2][:k].T
        pred, share = run(D, Ul); out[f"light_k{k}"] = rates(pred)
        if k == 64:
            m = y == 0; wrong = pred[m] == 1
            out["link_k64"] = {"share_wrong": float(share[m][wrong].mean()), "share_right": float(share[m][~wrong].mean()),
                               "auc_share_predicts_error": float(roc_auc_score(wrong, share[m]))}
        if k > 0:
            rr = [rates(run(D, lambda D, tr, s=s: np.linalg.qr(np.random.default_rng(s).standard_normal((D.shape[1], k)))[0])[0]) for s in range(5)]
            out[f"random_k{k}"] = {kk: float(np.mean([r[kk] for r in rr])) for kk in rr[0]}
    res[name] = out; print(name, json.dumps(out), flush=True)
json.dump(res, open(outp, "w"), indent=1)
