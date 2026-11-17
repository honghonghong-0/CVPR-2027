"""Frozen Qwen2.5-VL-7B vision-tower features (the encoder the VLM itself sees with), mean-pooled.

usage: PYTHONPATH=$P/pylib_vlm:$P/pylib python extract_qwen_vis.py <render_dir> <out_dir>
Writes <out_dir>/qwen25vl_vis.npz (files, feat) in the same format as extract_feats.py, so
analyze_e2.py / analyze_xfer*.py pick it up unchanged. Same pixel budget as the E3 prompts.
"""
import os, sys, glob, numpy as np, torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
dst = os.path.join(out, "qwen25vl_vis.npz")
if os.path.exists(dst): print("skip"); sys.exit()
dev = "cuda:" + os.environ.get("CVPR_GPU", "1"); mid = "Qwen/Qwen2.5-VL-7B-Instruct"
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16)
vis = m.visual.to(dev).eval(); del m; torch.cuda.empty_cache()
ip = AutoProcessor.from_pretrained(mid, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28).image_processor
files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(src, "*.png"))); feats = []
for i, f in enumerate(files):
    x = ip(images=[Image.open(os.path.join(src, f)).convert("RGB")], return_tensors="pt")
    with torch.no_grad():
        h = vis(x["pixel_values"].to(dev, torch.bfloat16), grid_thw=x["image_grid_thw"].to(dev))
    feats.append(h.float().mean(0).cpu().numpy())
    if i % 500 == 0: print(i, h.shape, flush=True)
np.savez(dst, files=np.array(files), feat=np.stack(feats)); print("done", np.stack(feats).shape)
