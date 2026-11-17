"""Qualitative figure. (a) material vs key/fill-ratio change with amplified difference maps (same base, view 0).
(b) ZeST trials at the 25/50/75th percentile of LPIPS change (light - raw), chosen by rank, not by look."""
import json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
P = json.load(open("results/pairs_diag.json"))
def img(p, s=256): return np.asarray(Image.open(p).convert("RGB").resize((s, s)), dtype=np.float32) / 255
def crop(x, f=0.18):
    h = x.shape[0]; a = int(h * f); return x[a:h - a, a:h - a]
rowsA = []
for mat in ["velvet", "satin", "pearl"]:
    q = [p for p in P if p["mat"] == mat and p["geom"] == "sphere" and p["view"] == 0.0 and p["color"] == "pink"]
    m = [p for p in q if p["changed"] == "material" and p["level"] == 2][0]; r = [p for p in q if p["changed"] == "ratio" and p["level"] == 2][0]
    b = img("renders_diag/" + m["base_file"]); xm = img("renders_diag/" + m["file"]); xr = img("renders_diag/" + r["file"])
    dm, dr = np.abs(xm - b).mean(2), np.abs(xr - b).mean(2); s = max(dm.max(), dr.max())
    rowsA.append((mat, [crop(b), crop(xm), crop(xr), crop(dm / s), crop(dr / s)]))
M = [json.loads(l) for f in sorted(glob.glob("renders_xfer/meta_*.jsonl")) for l in open(f)]
key = {(x["mat"], x["geom"], x["light_id"], x["level"]): "renders_xfer/" + x["file"] for x in M}
ze = json.load(open("results/zest_eval.json")); rows = sorted(ze["rows"], key=lambda r: r["light"]["lpips_gt"] - r["raw"]["lpips_gt"])
pick = [rows[int(q * (len(rows) - 1))] for q in (0.25, 0.5, 0.75)]
T = {f"{t['mat']}_{t['level']}_{t['g']}_{t['h']}_E{t['Le']}_T{t['Lt']}": t for t in json.load(open("zest_main/trials.json"))}
rowsB = []
for r in pick:
    t = T[r["tag"]]
    ims = [img(key[(t["mat"], t["h"], t["Le"], t["level"])]), img(key[("satin", t["g"], t["Lt"], 2)]), img(key[(t["mat"], t["g"], t["Lt"], t["level"])]),
           img(f"zest_main/{r['tag']}_raw.png"), img(f"zest_main/{r['tag']}_light.png")]
    rowsB.append((t["mat"], [crop(x, 0.12) for x in ims], r))
fig = plt.figure(figsize=(7.0, 3.1)); W = 0.188; H = 0.135
ta = ["Base", "Material +1", "Key/fill ratio", r"$|\Delta|$ material", r"$|\Delta|$ ratio"]
tb = ["Exemplar", "Target input", "Ground truth", "ZeST", "ZeST + removal"]
for i, (mat, ims) in enumerate(rowsA):
    for j, x in enumerate(ims):
        ax = fig.add_axes([0.06 + j * 0.19, 0.83 - i * H, W * 0.72, H * 0.95]) if False else fig.add_axes([0.045 + j * 0.091, 0.70 - i * 0.30, 0.088, 0.27])
        ax.imshow(x, cmap="magma" if j >= 3 else None, vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_visible(False)
        if i == 0: ax.set_title(ta[j], fontsize=6.5, pad=2)
        if j == 0: ax.set_ylabel(mat, fontsize=7)
for i, (mat, ims, r) in enumerate(rowsB):
    for j, x in enumerate(ims):
        ax = fig.add_axes([0.53 + j * 0.091, 0.70 - i * 0.30, 0.088, 0.27]); ax.imshow(x); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_visible(False)
        if i == 0: ax.set_title(tb[j], fontsize=6.5, pad=2)
        if j == 0: ax.set_ylabel(f"{mat}, level {r['tag'].split('_')[1]}", fontsize=6.5)
        if j == 3: ax.set_xlabel("LPIPS %.3f" % r["raw"]["lpips_gt"], fontsize=6)
        if j == 4: ax.set_xlabel("LPIPS %.3f" % r["light"]["lpips_gt"], fontsize=6)
fig.text(0.27, -0.02, "(a) Material change and key/fill ratio change", ha="center", fontsize=7.5)
fig.text(0.76, -0.02, "(b) ZeST transfer with and without lighting removal", ha="center", fontsize=7.5)
fig.savefig("figs_out/qual.pdf", bbox_inches="tight", dpi=300); fig.savefig("figs_out/qual.png", bbox_inches="tight", dpi=150)
print([r["tag"] for _, _, r in rowsB])
