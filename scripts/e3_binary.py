"""E3b: binary VLM attribution (material vs lighting), order-counterbalanced, with null (identical) pairs.

usage: PYTHONPATH=$P/pylib_vlm:$P/pylib python e3_binary.py <render_dir> <pairs.json> <out_json> [n_null]
Each pair asked twice (A=material/B=lighting and swapped). Score = P(material) from next-token logits
over {A,B}, averaged across both orders (removes letter-position bias). Null pairs show the base image
twice; their mean P(material) is the model's prior bias toward 'material' with nothing changed.
"""
import os, sys, json, random, torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
rd, pj, outp = sys.argv[1:4]; n_null = int(sys.argv[4]) if len(sys.argv) > 4 else 300
dev = "cuda:" + os.environ.get("CVPR_GPU", "1")
P = json.load(open(pj))
bases = sorted({p["base_file"] for p in P}); bases = random.Random(0).sample(bases, min(n_null, len(bases)))
items = [dict(file=p["file"], base_file=p["base_file"], changed=p["changed"], mat=p["mat"], geom=p["geom"], lpips=p["lpips"]) for p in P]
items += [dict(file=b, base_file=b, changed="null", mat=None, geom=None, lpips=0.0) for b in bases]
mid = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained(mid, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
tA, tB = [proc.tokenizer.encode(x, add_special_tokens=False)[0] for x in ["A", "B"]]
D = {"material": "the surface material of the object", "lighting": "the lighting of the scene"}
def side(a, b):
    A, B = Image.open(os.path.join(rd, a)).convert("RGB"), Image.open(os.path.join(rd, b)).convert("RGB")
    W = Image.new("RGB", (A.width + B.width + 16, A.height), "white"); W.paste(A, (0, 0)); W.paste(B, (A.width + 16, 0)); return W
def p_first(img, first, second):
    q = ("The left and right images show the same object from the same viewpoint. "
         f"What changed between them?\n(A) {D[first]}\n(B) {D[second]}\nAnswer with a single letter.")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    inp = proc(text=[proc.apply_chat_template(msgs, add_generation_prompt=True)], images=[img], return_tensors="pt").to(dev)
    with torch.no_grad(): lg = model(**inp).logits[0, -1, [tA, tB]].float()
    return torch.softmax(lg, 0)[0].item()  # P(choose A)
for i, it in enumerate(items):
    img = side(it["base_file"], it["file"])
    p1 = p_first(img, "material", "lighting")          # P(material) when material is A
    p2 = 1 - p_first(img, "lighting", "material")      # P(material) when material is B
    it.update(p_mat_order1=p1, p_mat_order2=p2, p_mat=(p1 + p2) / 2)
    if i % 200 == 0: print(i, it["changed"], round(p1, 3), round(p2, 3), flush=True)
json.dump(items, open(outp, "w"))
