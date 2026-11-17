"""Extract frozen image features for every rendered PNG in a folder.

usage: PYTHONPATH=$P/pylib python extract_feats.py <render_dir> <out_dir> [min_free_gb]
Writes <out_dir>/<model>.npz with keys: files, feat (N x D, float32).
Models: CLIP ViT-L/14 (pooled), SigLIP so400m (pooled), DINOv2-L (CLS + mean patch, concat).
Runs on the GPU with most free memory if >= min_free_gb free, else CPU.
"""
import os, sys, glob, subprocess, numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoImageProcessor, CLIPVisionModel, SiglipVisionModel

src, out = sys.argv[1], sys.argv[2]
min_free = float(sys.argv[3]) if len(sys.argv) > 3 else 12
os.makedirs(out, exist_ok=True)
files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(src, "*.png")))
print("images", len(files))

def pick_device():
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]).decode()
        gpu = int(os.environ.get("CVPR_GPU", "1"))  # GPU1 reserved for CVPR by teammate agreement
        free = {int(l.split(",")[0]): int(l.split(",")[1]) for l in q.strip().splitlines()}
        if free.get(gpu, 0) / 1024 >= min_free:
            return f"cuda:{gpu}"
    except Exception as e:
        print("nvidia-smi failed", e)
    return "cpu"

dev = pick_device(); print("device", dev)
MODELS = {
    "clip_l14": ("openai/clip-vit-large-patch14", CLIPVisionModel, AutoImageProcessor),
    "siglip_so400m": ("google/siglip-so400m-patch14-384", SiglipVisionModel, AutoImageProcessor),
    "dinov2_l": ("facebook/dinov2-large", AutoModel, AutoImageProcessor),
}
only = os.environ.get("ONLY")
for name, (hf, cls, pcls) in MODELS.items():
    if only and name not in only.split(","): continue
    dst = os.path.join(out, name + ".npz")
    if os.path.exists(dst): print("skip", name); continue
    model = cls.from_pretrained(hf, torch_dtype=torch.float16 if dev != "cpu" else torch.float32).to(dev).eval()
    proc = pcls.from_pretrained(hf)
    feats = []
    for i in range(0, len(files), 32):
        ims = [Image.open(os.path.join(src, f)).convert("RGB") for f in files[i:i + 32]]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev, model.dtype)
        with torch.no_grad():
            o = model(pixel_values=x)
        if name.startswith("dinov2"):
            h = o.last_hidden_state
            f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
        else:
            f = o.pooler_output
        feats.append(f.float().cpu().numpy())
    np.savez(dst, files=np.array(files), feat=np.concatenate(feats))
    print(name, np.concatenate(feats).shape)
    del model; torch.cuda.empty_cache()
