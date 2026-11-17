"""Share of feature-change energy lying inside the lighting subspace U_k.
U_k from lighting-only pairs of two shapes; energy measured on the held-out shape (leave-one-geometry-out).
energy(d) = ||U^T d||^2 / ||d||^2, averaged over pairs. Random-subspace expectation = k / D."""
import sys, json, glob, os, numpy as np
fd, pj, outp = sys.argv[1:4]; P = json.load(open(pj)); KS = [1, 4, 16, 64]
geom = np.array([p["geom"] for p in P]); ch = np.array([p["changed"] for p in P]); res = {}
for f in sorted(glob.glob(os.path.join(fd, "*.npz"))):
    z = np.load(f); F = z["feat"] / np.linalg.norm(z["feat"], axis=1, keepdims=True); idx = {n: i for i, n in enumerate(z["files"])}
    D = np.stack([F[idx[p["file"]]] - F[idx[p["base_file"]]] for p in P]); name = os.path.basename(f)[:-4]; r = {}
    for k in KS:
        acc = {c: [] for c in ["material", "lighting"]}
        for g in np.unique(geom):
            tr = (geom != g) & (ch != "material"); X = D[tr] - D[tr].mean(0)
            U = np.linalg.svd(X, full_matrices=False)[2][:k].T
            te = geom == g; e = ((D[te] @ U) ** 2).sum(1) / (D[te] ** 2).sum(1)
            acc["material"] += e[ch[te] == "material"].tolist(); acc["lighting"] += e[ch[te] != "material"].tolist()
        r[k] = {c: float(np.mean(v)) for c, v in acc.items()}; r[k]["random"] = k / D.shape[1]
    res[name] = r; print(name, {k: {c: round(v, 3) for c, v in d.items()} for k, d in r.items()})
json.dump(res, open(outp, "w"), indent=1)
