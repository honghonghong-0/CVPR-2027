"""Real material transfer with ZeST (Cheng et al., ECCV 2024), with/without lighting-subspace removal on the
IP-Adapter image embedding (OpenCLIP ViT-H/14 image_embeds, 1024-d).

usage: python zest_run.py <out_dir> [--n N]
Trial: material m, level j, target shape g, exemplar shape h != g, exemplar light Le, target light Lt (Le != Lt, both held-out 8..11).
  exemplar = render(m, h, Le, j);  target input = render(satin, g, Lt, level 2) (grayscale inside mask, ZeST default)
  GT       = render(m, g, Lt, j);  LEAK = render(m, g, Le, j)  (right material, exemplar's lighting)
Conditions: raw | light (remove lighting subspace, k fixed below) | random (random k-dim subspace, same k).
Subspace U: ViT-H embeds of xfer renders, lights 0..7 only, shapes != g (same estimator as the method).
Embedding after projection is rescaled to the original norm (IP-Adapter expects unnormalized embeds).
Same seed across conditions -> paired comparison.
"""
import sys, os, json, glob, numpy as np, torch
from PIL import Image, ImageChops, ImageEnhance
P = "."; Z = P + "/ext/zest_code"; W = P + "/ext/weights/IP-Adapter"
sys.path.insert(0, Z)
OUT = sys.argv[1]; N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10**9
K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 64
os.makedirs(OUT, exist_ok=True); dev = "cuda"
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from transformers import DPTForDepthEstimation, DPTImageProcessor
from ip_adapter import IPAdapterXL
from ip_adapter.ip_adapter import IPAdapter

M = [json.loads(l) for f in sorted(glob.glob(P + "/renders_xfer/meta_*.jsonl")) for l in open(f)]
key = {(m["mat"], m["geom"], m["light_id"], m["level"]): P + "/renders_xfer/" + m["file"] for m in M}
mats = sorted({m["mat"] for m in M}); geoms = ["sphere", "blob", "cloth"]
mask_of = {g: Image.open(f).convert("L") for g in geoms for f in glob.glob(P + f"/renders_aux/*_mask_{g}_aux0.png")}

cn = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(dev)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn,
        use_safetensors=True, torch_dtype=torch.float16, variant="fp16", add_watermarker=False).to(dev)
ip = IPAdapterXL(pipe, W + "/models/image_encoder", W + "/sdxl_models/ip-adapter_sdxl_vit-h.bin", dev)
dpt = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(dev).eval(); dproc = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

@torch.no_grad()
def embed(paths):
    out = []
    for i in range(0, len(paths), 32):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + 32]]
        x = ip.clip_image_processor(images=ims, return_tensors="pt").pixel_values.to(dev, torch.float16)
        out.append(ip.image_encoder(x).image_embeds.float().cpu().numpy())
    return np.concatenate(out)

# lighting subspace per target shape g (shapes != g, lights 0..7)
allk = sorted(key); _F = embed([key[k] for k in allk]); E = dict(zip(allk, _F))
os.makedirs(P + "/feats_vith", exist_ok=True)
np.savez(P + "/feats_vith/openclip_vith.npz", feat=_F, files=np.array([os.path.basename(key[k]) for k in allk]))
if "--embed_only" in sys.argv: sys.exit(0)
nz = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
def light_basis(g):
    R = []
    for m in mats:
        for h in geoms:
            if h == g: continue
            for j in range(5):
                X = nz(np.stack([E[(m, h, l, j)] for l in range(8)])); R.append(X - X.mean(0))
    D = np.concatenate(R); D -= D.mean(0); return np.linalg.svd(D, full_matrices=False)[2][:K].T
UL = {g: light_basis(g) for g in geoms}
UR = np.linalg.qr(np.random.default_rng(0).standard_normal((E[allk[0]].shape[0], K)))[0]

def project(e, U):
    n = np.linalg.norm(e); u = e / n; u = u - U @ (U.T @ u); return u / np.linalg.norm(u) * n

@torch.no_grad()
def depth_of(img):
    x = dproc(images=img, return_tensors="pt").to(dev); d = dpt(**x).predicted_depth[0].cpu().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255
    return Image.fromarray(d.astype("uint8")).resize((1024, 1024)).convert("RGB")

def zest_inputs(target, mask):
    m = mask.convert("RGB"); g = ImageEnhance.Brightness(target.convert("L").convert("RGB")).enhance(1.0)
    init = ImageChops.lighter(ImageChops.darker(target, ImageChops.invert(m)), ImageChops.darker(g, m))
    return init.resize((1024, 1024)), m.resize((1024, 1024))

trials = []
import random as _r
_g = _r.Random(27)
XL = [{"size": _g.choice([0.1, 1.0, 3.0]), "ratio": _g.choice([2.0, 4.0, 8.0]), "dir": _g.choice([0.0, 25.0, 50.0]), "kelvin": _g.choice([3200.0, 5600.0, 7500.0])} for _ in range(12)]
ndiff = lambda a, b: sum(XL[a][f] != XL[b][f] for f in XL[a])
for m in mats:
    for j in [0, 2, 4]:
        for gi, g in enumerate(geoms):
            h = geoms[(gi + 1) % 3]
            for Le in range(8, 12):
                Lt = max([l for l in range(8, 12) if l != Le], key=lambda l: (ndiff(Le, l), -l))  # most different held-out light
                trials.append(dict(mat=m, level=j, g=g, h=h, Le=Le, Lt=Lt))
trials = trials[:N]; json.dump(trials, open(OUT + "/trials.json", "w"))
for t in trials:
    tag = f"{t['mat']}_{t['level']}_{t['g']}_{t['h']}_E{t['Le']}_T{t['Lt']}"
    if os.path.exists(f"{OUT}/{tag}_random.png"): continue
    target = Image.open(key[("satin", t["g"], t["Lt"], 2)]).convert("RGB")
    init, mask = zest_inputs(target, mask_of[t["g"]]); depth = depth_of(target)
    e = E[(t["mat"], t["h"], t["Le"], t["level"])]
    for cond, emb in [("raw", e), ("light", project(e, UL[t["g"]])), ("random", project(e, UR))]:
        ce = torch.tensor(emb[None], dtype=torch.float16, device=dev)
        ip.get_image_embeds = lambda pil_image=None, ce=ce: IPAdapter.get_image_embeds(ip, clip_image_embeds=ce)
        im = ip.generate(pil_image=target, image=init, control_image=depth, mask_image=mask, controlnet_conditioning_scale=0.9,
                         num_samples=1, num_inference_steps=30, seed=42)[0]
        im.resize((512, 512)).save(f"{OUT}/{tag}_{cond}.png")
    print(tag, flush=True)
print("DONE")
