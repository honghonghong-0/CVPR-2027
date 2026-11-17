"""Evaluate ZeST outputs (raw | light | random) against Blender ground truth.
usage: python zest_eval.py <zest_dir> <out_json>
Per trial (m, j, g, h, Le, Lt):  GT = render(m,g,Lt,j)   LEAK = render(m,g,Le,j)  (exemplar's lighting)
Metrics (inside the object mask of shape g):
  lpips_gt      LPIPS(O, GT)                       lower = better transfer
  leak          LPIPS(O, LEAK) - LPIPS(O, GT)      higher = output follows target lighting, not exemplar's
  color_gt      |mean RGB(O) - mean RGB(GT)|       global color (Kelvin) leakage proxy
  level_err     |argmax_j' cos(DINOv2(O), DINOv2(render(m,g,Lt,j'))) - j|   (DINOv2 is NOT the encoder that was projected)
Paired bootstrap (B=10000) of (cond - raw), resampling trials and, separately, material x shape clusters.
"""
import sys, os, json, glob, numpy as np, torch, lpips
from PIL import Image
P = "."; ZD, OUTP = sys.argv[1:3]; dev = "cuda"
M = [json.loads(l) for f in sorted(glob.glob(P + "/renders_xfer/meta_*.jsonl")) for l in open(f)]
key = {(m["mat"], m["geom"], m["light_id"], m["level"]): P + "/renders_xfer/" + m["file"] for m in M}
T = json.load(open(ZD + "/trials.json")); conds = ["raw", "light", "random"]
mask = {g: (np.array(Image.open(glob.glob(P + f"/renders_aux/*_mask_{g}_aux0.png")[0]).convert("L").resize((512, 512))) > 127) for g in ["sphere", "blob", "cloth"]}
lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
from transformers import AutoModel, AutoImageProcessor
dm = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval(); dp = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
load = lambda p: np.asarray(Image.open(p).convert("RGB").resize((512, 512)), dtype=np.float32) / 255
def masked(x, m): return x * m[..., None]
@torch.no_grad()
def lpd(a, b, m):
    t = lambda x: torch.tensor(masked(x, m)).permute(2, 0, 1)[None].to(dev) * 2 - 1
    return float(lp(t(a), t(b)))
@torch.no_grad()
def dino(xs):
    x = dp(images=[Image.fromarray((v * 255).astype("uint8")) for v in xs], return_tensors="pt").to(dev)
    f = dm(**x).last_hidden_state[:, 0]; return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()
rows = []
for t in T:
    tag = f"{t['mat']}_{t['level']}_{t['g']}_{t['h']}_E{t['Le']}_T{t['Lt']}"
    if not all(os.path.exists(f"{ZD}/{tag}_{c}.png") for c in conds): continue
    m = mask[t["g"]]; GT = load(key[(t["mat"], t["g"], t["Lt"], t["level"])]); LK = load(key[(t["mat"], t["g"], t["Le"], t["level"])])
    cands = dino([masked(load(key[(t["mat"], t["g"], t["Lt"], j)]), m) for j in range(5)])
    O = {c: load(f"{ZD}/{tag}_{c}.png") for c in conds}; fo = dino([masked(O[c], m) for c in conds])
    r = {"tag": tag, "cluster": f"{t['mat']}_{t['g']}"}
    for i, c in enumerate(conds):
        g_, l_ = lpd(O[c], GT, m), lpd(O[c], LK, m)
        r[c] = {"lpips_gt": g_, "leak": l_ - g_, "color_gt": float(np.abs(O[c][m].mean(0) - GT[m].mean(0)).sum()),
                "level_err": abs(int(np.argmax(cands @ fo[i])) - t["level"])}
    rows.append(r)
mets = ["lpips_gt", "leak", "color_gt", "level_err"]; rng = np.random.default_rng(0); res = {"n": len(rows), "summary": {}}
cl = sorted({r["cluster"] for r in rows}); ci = {c: [i for i, r in enumerate(rows) if r["cluster"] == c] for c in cl}
for c in conds:
    res["summary"][c] = {k: float(np.mean([r[c][k] for r in rows])) for k in mets}
for c in ["light", "random"]:
    for k in mets:
        d = np.array([r[c][k] - r["raw"][k] for r in rows])
        bq = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        bc = [np.concatenate([d[ci[x]] for x in rng.choice(cl, len(cl))]).mean() for _ in range(10000)]
        res["summary"][f"{c}-raw_{k}"] = {"mean": float(d.mean()), "q95": np.percentile(bq, [2.5, 97.5]).tolist(), "cl95": np.percentile(bc, [2.5, 97.5]).tolist()}
res["rows"] = rows; json.dump(res, open(OUTP, "w"), indent=1)
for k, v in res["summary"].items(): print(k, v)
