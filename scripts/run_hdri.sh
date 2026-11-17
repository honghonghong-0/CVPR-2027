#!/bin/bash
P=.; cd $P
export PYTHONPATH=pylib HF_HOME=./cache/hf TORCH_HOME=./cache/torch
PY=python
until [ "$(ls renders_hdri/*.png 2>/dev/null | wc -l)" -ge 900 ]; do sleep 60; done
echo "[$(date)] hdri renders done"
# remap so held-out HDRIs {4,5,8,11} -> ids 8..11 (analysis scripts assume 0..7 match, 8..11 held-out)
mkdir -p renders_hdri_m && cd renders_hdri_m && ln -sf ../renders_hdri/*.png . && cd ..
$PY - <<"PY"
import json,glob
mp={o:i for i,o in enumerate([0,1,2,3,6,7,9,10,4,5,8,11])}
L=[json.loads(l) for f in glob.glob("renders_hdri/meta_*.jsonl") for l in open(f)]
with open("renders_hdri_m/meta_0.jsonl","w") as f:
    for m in L: m["hdri_id"]=m["light_id"]; m["light_id"]=mp[m["light_id"]]; f.write(json.dumps(m)+"\n")
print(len(L))
PY
CVPR_GPU=1 $PY scripts/extract_feats.py renders_hdri_m feats_hdri 12
$PY scripts/analyze_xfer.py feats_hdri renders_hdri_m results/hdri_xfer.json
$PY scripts/analyze_xfer_proj.py feats_hdri renders_hdri_m feats_diag results/pairs_diag.json results/hdri_proj.json
$PY scripts/boot_proj.py results/hdri_proj.json results/hdri_proj_ci.json
$PY scripts/controls_proj.py feats_hdri renders_hdri_m results/hdri_controls.json
echo "[$(date)] HDRI DONE"
