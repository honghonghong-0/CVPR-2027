"""Real-photo check on the Multi-Illumination dataset (Murmann et al., ICCV 2019): 25 flash directions per scene,
per-pixel material labels. Extract patch features for every patch under every lighting direction.
usage: python ml_feats.py <out_dir> [n_train_scenes]
Patches: 224x224 crops (stride 160) of the mip2 images (1000x1500) whose pixels are >= 80% one material label (label != 0),
excluding the bottom 22% of the frame (light probes). At most 12 patches per scene (seeded). Scenes: all 30 test + N train.
Output <out_dir>/<model>.npz: feat [n_patch, 25, D], label, scene, split.
"""
import os, sys, glob, json, numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor, CLIPVisionModel, SiglipVisionModel
R = "./data/multilum"; out = sys.argv[1]; NTR = int(sys.argv[2]) if len(sys.argv) > 2 else 300
os.makedirs(out, exist_ok=True); rng = np.random.default_rng(0)
te = sorted(glob.glob(R + "/test/*/")); tr = sorted(glob.glob(R + "/train/*/")); tr = [tr[i] for i in sorted(rng.choice(len(tr), min(NTR, len(tr)), replace=False))]
patches = []
for split, S in [("test", te), ("train", tr)]:
    for s in S:
        if not os.path.exists(s + "materials_mip2.png"): continue
        L = np.array(Image.open(s + "materials_mip2.png")); H, W = L.shape; cand = []
        for y in range(0, int(H * 0.78) - 224 + 1, 160):
            for x in range(0, W - 224 + 1, 160):
                c = L[y:y + 224, x:x + 224]; v, n = np.unique(c, return_counts=True); i = n.argmax()
                if v[i] != 0 and n[i] >= 0.8 * c.size: cand.append((y, x, int(v[i])))
        for k in rng.permutation(len(cand))[:12]: patches.append((split, s, *cand[k]))
print("patches", len(patches), "scenes", len({p[1] for p in patches}), flush=True)
MODELS = {"clip_l14": ("openai/clip-vit-large-patch14", CLIPVisionModel), "siglip_so400m": ("google/siglip-so400m-patch14-384", SiglipVisionModel),
          "dinov2_l": ("facebook/dinov2-large", AutoModel)}
dev = "cuda"
loaded = {n: (c.from_pretrained(h, torch_dtype=torch.float16).to(dev).eval(), AutoImageProcessor.from_pretrained(h)) for n, (h, c) in MODELS.items()}
F = {n: [] for n in MODELS}
by_scene = {}
for i, p in enumerate(patches): by_scene.setdefault(p[1], []).append(i)
order = [i for s in by_scene for i in by_scene[s]]
for s, idxs in by_scene.items():
    ims = [Image.open(f"{s}dir_{d}_mip2.jpg").convert("RGB") for d in range(25)]
    crops = [ims[d].crop((patches[i][3], patches[i][2], patches[i][3] + 224, patches[i][2] + 224)) for i in idxs for d in range(25)]
    for n, (m, pr) in loaded.items():
        with torch.no_grad():
            fs = []
            for b in range(0, len(crops), 100):
                o = m(pixel_values=pr(images=crops[b:b + 100], return_tensors="pt")["pixel_values"].to(dev, torch.float16))
                fs.append((torch.cat([o.last_hidden_state[:, 0], o.last_hidden_state[:, 1:].mean(1)], 1) if n.startswith("dinov2") else o.pooler_output).float().cpu().numpy())
        F[n].append(np.concatenate(fs).reshape(len(idxs), 25, -1))
for n in MODELS:
    np.savez(f"{out}/{n}.npz", feat=np.concatenate(F[n]), label=np.array([patches[i][4] for i in order]),
             scene=np.array([os.path.basename(patches[i][1][:-1]) for i in order]), split=np.array([patches[i][0] for i in order]))
print("DONE", flush=True)
