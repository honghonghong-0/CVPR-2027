"""Per-pair change magnitude (pixel mean |d| and LPIPS-alex) for a diag render folder.

usage: PYTHONPATH=$P/pylib python pair_magnitude.py <render_dir> <out_json>
Each row: file, pair_id, changed, level, pix, lpips (variant vs its base).
"""
import sys, glob, json, os, numpy as np, torch, lpips
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
M = [json.loads(l) for f in sorted(glob.glob(os.path.join(src, "meta_*.jsonl"))) for l in open(f)]
dev = "cuda:" + os.environ.get("CVPR_GPU", "1") if torch.cuda.is_available() and os.environ.get("CPU") != "1" else "cpu"
net = lpips.LPIPS(net="alex", verbose=False).to(dev)
def load(f):
    return np.asarray(Image.open(os.path.join(src, f)).convert("RGB"), np.float32) / 255
def t(a):
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(dev) * 2 - 1
base = {m["pair_id"]: m["file"] for m in M if m["changed"] == "base"}
rows = []
for m in M:
    if m["changed"] == "base" or m["pair_id"] not in base: continue
    a, b = load(base[m["pair_id"]]), load(m["file"])
    with torch.no_grad():
        lp = float(net(t(a), t(b)))
    rows.append({k: m.get(k) for k in ["file", "pair_id", "changed", "level", "mat", "geom", "view", "color"]}
                | {"base_file": base[m["pair_id"]], "pix": float(np.abs(a - b).mean()), "lpips": lp})
json.dump(rows, open(out, "w"))
print("pairs", len(rows))
