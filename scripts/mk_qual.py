"""Qualitative figure. (a) one-step material edit vs one-step key/fill ratio change (same base, view 0), with mean absolute
RGB difference to the base on ONE color scale shared by all six maps. (b) ZeST trials at the 25/50/75th percentile of
LPIPS change (removal - original), picked by rank."""
import json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams["font.family"] = "DejaVu Sans"
P = json.load(open("results/pairs_diag.json"))
def img(p, s=256): return np.asarray(Image.open(p).convert("RGB").resize((s, s)), dtype=np.float32) / 255
def crop(x, f=0.18): h = x.shape[0]; a = int(h * f); return x[a:h - a, a:h - a]
A = []
for mat in ["velvet", "satin", "pearl"]:
    q = [p for p in P if p["mat"] == mat and p["geom"] == "sphere" and p["view"] == 0.0 and p["color"] == "pink"]
    m = [p for p in q if p["changed"] == "material" and p["level"] == 2][0]; r = [p for p in q if p["changed"] == "ratio" and p["level"] == 2][0]
    b = img("renders_diag/" + m["base_file"]); xm = img("renders_diag/" + m["file"]); xr = img("renders_diag/" + r["file"])
    A.append((mat, [crop(b), crop(xm), crop(xr)], [crop(np.abs(xm - b).mean(2)), crop(np.abs(xr - b).mean(2))]))
vmax = max(d.max() for _, _, ds in A for d in ds)
M = [json.loads(l) for f in sorted(glob.glob("renders_xfer/meta_*.jsonl")) for l in open(f)]
key = {(x["mat"], x["geom"], x["light_id"], x["level"]): "renders_xfer/" + x["file"] for x in M}
ze = json.load(open("results/zest_eval.json")); rows = sorted(ze["rows"], key=lambda r: r["light"]["lpips_gt"] - r["raw"]["lpips_gt"])
pick = [rows[int(q * (len(rows) - 1))] for q in (0.25, 0.5, 0.75)]
T = {f"{t['mat']}_{t['level']}_{t['g']}_{t['h']}_E{t['Le']}_T{t['Lt']}": t for t in json.load(open("zest_main/trials.json"))}
B = []
for r in pick:
    t = T[r["tag"]]
    B.append((f"{t['mat'].capitalize()}\nlevel {t['level']}", [crop(img(p), 0.12) for p in [key[(t["mat"], t["h"], t["Le"], t["level"])], key[("satin", t["g"], t["Lt"], 2)],
              key[(t["mat"], t["g"], t["Lt"], t["level"])], f"zest_main/{r['tag']}_raw.png", f"zest_main/{r['tag']}_light.png"]], r))
fig = plt.figure(figsize=(7.0, 3.2))
cw, ch, gx = 0.0795, 0.235, 0.007            # cell width/height (fig fraction), column gap
lw_ = 0.060                                # row-label column width
x0a, x0b = 0.0, 0.525                      # panel origins
ia, ib = x0a + lw_, x0b + lw_              # first image column of each panel
ytop = 0.80; rowstep = 0.285
pw = 5 * cw + 4 * gx
def cell(x0, i, j): return fig.add_axes([x0 + j * (cw + gx), ytop - i * rowstep - ch, cw, ch])
def clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
def rowlab(x, i, t): fig.text(x, ytop - i * rowstep - ch / 2, t, fontsize=6.5, fontweight="bold", va="center", ha="left", linespacing=1.3)
ta = ["Base", "Material +1", "Ratio +1", r"$|\Delta|$ material", r"$|\Delta|$ ratio"]
tb = ["Exemplar", "Target input", "Ground truth", "ZeST", "+ removal"]
for i, (mat, ims, ds) in enumerate(A):
    for j, x in enumerate(ims + ds):
        ax = cell(ia, i, j); clean(ax)
        im = ax.imshow(x, cmap="magma", vmin=0, vmax=vmax) if j >= 3 else ax.imshow(x)
        if i == 0: ax.set_title(ta[j], fontsize=6.3, pad=3)
    rowlab(x0a, i, mat.capitalize())
for i, (lab, ims, r) in enumerate(B):
    for j, x in enumerate(ims):
        ax = cell(ib, i, j); clean(ax); ax.imshow(x)
        if i == 0: ax.set_title(tb[j], fontsize=6.3, pad=3)
    rowlab(x0b, i, lab)
    yb = ytop - i * rowstep - ch - 0.006
    d = r["light"]["lpips_gt"] - r["raw"]["lpips_gt"]
    fig.text(ib + 3 * (cw + gx) + cw / 2, yb, "LPIPS %.3f" % r["raw"]["lpips_gt"], fontsize=5.4, ha="center", va="top")
    fig.text(ib + 4 * (cw + gx) + cw / 2, yb, "%.3f (%s %.3f)" % (r["light"]["lpips_gt"], "↑" if d > 0 else "↓", abs(d)), fontsize=5.4, ha="center", va="top",
             color="#b2182b" if d > 0 else "#2166ac")
cax = fig.add_axes([ia + 3 * (cw + gx) + 0.01, ytop - 2 * rowstep - ch - 0.075, 2 * cw + gx - 0.02, 0.016])
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=5, length=2, pad=1); cb.outline.set_linewidth(0.4)
cb.set_label("mean |RGB difference|, shared by all six maps", fontsize=5.2, labelpad=1)
ty = ytop + 0.085
fig.text(x0a, ty, "(a) Material edit vs. key/fill ratio change", fontsize=7.3, fontweight="bold", va="bottom")
fig.text(x0b, ty, "(b) Lighting removal in ZeST transfer", fontsize=7.3, fontweight="bold", va="bottom")
for x0 in (x0a, x0b): fig.add_artist(plt.Line2D([x0, x0 + lw_ + pw], [ty - 0.018, ty - 0.018], color="#A0A0A0", lw=0.5))
fig.savefig("figs_out/qual.pdf", bbox_inches="tight", dpi=300); fig.savefig("figs_out/qual.png", bbox_inches="tight", dpi=150)
print("ok")
