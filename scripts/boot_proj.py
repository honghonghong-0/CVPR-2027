"""Paired bootstrap CIs for analyze_xfer_proj.py: MAE(projected at k*) - MAE(k=0), same queries.
Query order = mats x geoms x heldout lights(4) x levels(5) -> cluster = (mat, geom), 15 clusters of 20.
Reports query-level and cluster-level (resample mat x geom cells) 95% CIs, B=10000."""
import json, sys, numpy as np
r = json.load(open(sys.argv[1])); rng = np.random.default_rng(0); B = 10000; out = {}
for m, v in r.items():
    for cond, d in v.items():
        k = str(d["k_selected"]); a = np.array(d["heldout_curve"][k]["errs"]); b = np.array(d["heldout_curve"]["0"]["errs"])
        diff = a - b; n = len(diff); cl = diff.reshape(15, -1)
        qi = rng.integers(0, n, (B, n)); ci_q = np.percentile(diff[qi].mean(1), [2.5, 97.5])
        ci_c = np.percentile(cl[rng.integers(0, 15, (B, 15))].mean((1, 2)), [2.5, 97.5])
        out[f"{m}/{cond}"] = dict(k=int(k), base=b.mean(), proj=a.mean(), delta=diff.mean(), ci_query=ci_q.tolist(), ci_cluster=ci_c.tolist())
        print(f"{m:14s} {cond:14s} k*={k:>2s} {b.mean():.3f}->{a.mean():.3f} d={diff.mean():+.3f} q95[{ci_q[0]:+.3f},{ci_q[1]:+.3f}] cl95[{ci_c[0]:+.3f},{ci_c[1]:+.3f}]")
json.dump(out, open(sys.argv[2], "w"), indent=1)
