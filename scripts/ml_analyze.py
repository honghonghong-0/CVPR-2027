"""Real photos: does removing a lighting subspace make material recognition more robust to lighting change?
usage: python ml_analyze.py <feat_dir> <out_json>
Lights: 25 flash directions, split (seeded) into fit 12 / val 5 / held-out 8. Scenes: train scenes for U, probe, k-selection
(80/20 fit/val by scene); the 30 official test scenes only for the final numbers.
U_k: top-k PCs of within-patch deviations across fit lights (train-fit scenes). Control: random k-dim subspace (5 seeds).
Tasks on test scenes, held-out lights:
  probe : logistic regression on train scenes x fit lights -> balanced accuracy of material label
  xnn   : cross-light 1-NN; query (patch, light a), gallery = patches of OTHER test scenes under a different held-out light b
k chosen on train-val scenes x val lights (probe), never on test scenes or held-out lights.
"""
import sys, glob, os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
fd, outp = sys.argv[1:3]; rng = np.random.default_rng(0)
perm = rng.permutation(25); FIT, VAL, HO = perm[:12], perm[12:17], perm[17:]
KS = [0, 4, 16, 64]; nz = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
res = {"lights": {"fit": FIT.tolist(), "val": VAL.tolist(), "heldout": HO.tolist()}}
for f in sorted(glob.glob(fd + "/*.npz")):
    n = os.path.basename(f)[:-4]; z = np.load(f); X = nz(z["feat"]); lab = z["label"]; sc = z["scene"]; sp = z["split"]
    trs = sorted(set(sc[sp == "train"])); r2 = np.random.default_rng(1).permutation(len(trs)); vs = set(trs[i] for i in r2[:len(trs) // 5])
    fit = (sp == "train") & ~np.isin(sc, list(vs)); val = np.isin(sc, list(vs)); te = sp == "test"
    keep = [c for c in np.unique(lab) if (lab[fit] == c).sum() >= 30 and (lab[te] == c).sum() >= 5]
    ok = np.isin(lab, keep); fit &= ok; val &= ok; te &= ok
    Dv = X[fit][:, FIT]; Dv = (Dv - Dv.mean(1, keepdims=True)).reshape(-1, X.shape[-1]); Dv -= Dv.mean(0)
    V = np.linalg.svd(Dv[rng.permutation(len(Dv))[:20000]], full_matrices=False)[2]
    def P(A, U): return A if U is None else nz(A - (A @ U) @ U.T)
    def probe(U, trm, tem, trL, teL):
        A = P(X[trm][:, trL].reshape(-1, X.shape[-1]), U); ya = np.repeat(lab[trm], len(trL))
        B = P(X[tem][:, teL].reshape(-1, X.shape[-1]), U); yb = np.repeat(lab[tem], len(teL))
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(A, ya)
        return float(balanced_accuracy_score(yb, clf.predict(B)))
    def xnn(U):
        idx = np.where(te)[0]; acc = []
        for a in HO:
            for b in HO:
                if a == b: continue
                Q = P(X[idx, a], U); G = P(X[idx, b], U); S = Q @ G.T
                S[sc[idx][:, None] == sc[idx][None]] = -9  # exclude same scene from the gallery
                acc.append((lab[idx][S.argmax(1)] == lab[idx]).mean())
        return float(np.mean(acc))
    sel = {k: probe(None if k == 0 else V[:k].T, fit, val, FIT, VAL) for k in KS}; kb = max(sel, key=sel.get)
    out = {"n_classes": len(keep), "n_test_patches": int(te.sum()), "val": sel, "k_selected": kb}
    for k in sorted({0, kb, 16, 64}):
        U = None if k == 0 else V[:k].T
        out[f"light_k{k}"] = {"probe": probe(U, fit, te, FIT, HO), "xnn": xnn(U)}
        if k > 0:
            rr = [np.linalg.qr(np.random.default_rng(s).standard_normal((X.shape[-1], k)))[0] for s in range(5)]
            out[f"random_k{k}"] = {"probe": float(np.mean([probe(Ur, fit, te, FIT, HO) for Ur in rr])), "xnn": float(np.mean([xnn(Ur) for Ur in rr]))}
    res[n] = out; print(n, json.dumps(out), flush=True)
json.dump(res, open(outp, "w"), indent=1)
