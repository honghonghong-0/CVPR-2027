"""E3: VLM change attribution. Base and variant shown side by side; model picks what changed.

usage: PYTHONPATH=$P/pylib_vlm:$P/pylib python e3_vlm.py <render_dir> <pairs.json> <out_json> [max_pairs]
Model: Qwen2.5-VL-7B-Instruct (bf16) on GPU CVPR_GPU (default 1).
Five options, order shuffled per pair (seeded); answer = single letter. Records raw output.
"""
import os, sys, json, random, re, torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

rd, pj, outp = sys.argv[1:4]
maxn = int(sys.argv[4]) if len(sys.argv) > 4 else 0
dev = "cuda:" + os.environ.get("CVPR_GPU", "1")
P = json.load(open(pj))
if maxn: P = random.Random(0).sample(P, min(maxn, len(P)))
OPTS = {"material": "the surface material of the object (e.g., glossiness, sheen, coating, translucency, iridescence)",
        "size": "the size of the light source (harder vs. softer shadows and highlights)",
        "ratio": "the balance between the main light and the fill light (overall contrast)",
        "dir": "the direction the main light comes from",
        "kelvin": "the color temperature of the light (warmer vs. cooler light)"}
mid = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained(mid, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)

def side(a, b):
    A, B = Image.open(os.path.join(rd, a)).convert("RGB"), Image.open(os.path.join(rd, b)).convert("RGB")
    W = Image.new("RGB", (A.width + B.width + 16, A.height), "white")
    W.paste(A, (0, 0)); W.paste(B, (A.width + 16, 0)); return W

rows = []
for i, p in enumerate(P):
    rng = random.Random(i)
    keys = list(OPTS); rng.shuffle(keys)
    letters = "ABCDE"
    q = ("The left and right images show the same object from the same viewpoint. Exactly one thing differs between them. "
         "Which one?\n" + "\n".join(f"({letters[j]}) {OPTS[k]}" for j, k in enumerate(keys)) +
         "\nAnswer with a single letter.")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=[side(p["base_file"], p["file"])], return_tensors="pt").to(dev)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=4, do_sample=False)
    ans = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    m = re.search(r"[A-E]", ans)
    pred = keys[letters.index(m.group(0))] if m else None
    rows.append({**{k: p[k] for k in ["file", "changed", "mat", "geom", "lpips"]}, "raw": ans, "pred": pred})
    if i % 100 == 0: print(i, p["changed"], ans, pred, flush=True)
json.dump(rows, open(outp, "w"))
acc = sum(r["pred"] == r["changed"] for r in rows) / len(rows)
print("n", len(rows), "acc", round(acc, 3), "parse_fail", sum(r["pred"] is None for r in rows))
