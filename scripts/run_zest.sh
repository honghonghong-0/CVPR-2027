#!/bin/bash
P=.; cd $P
export PYTHONPATH=pylib_zest:pylib:pylib_vlm HF_HOME=./cache/hf CUDA_VISIBLE_DEVICES=1 TMPDIR=./tmp
PY=python
while pgrep -f "zest_run.py zest_smoke" >/dev/null; do sleep 10; done
$PY scripts/zest_run.py zest_tmp --embed_only
mkdir -p feats_vith_only && cp feats_vith/openclip_vith.npz feats_vith_only/
$PY scripts/controls_proj.py feats_vith_only renders_xfer results/controls_vith.json
K=$($PY -c "import json;d=json.load(open(\"results/controls_vith.json\"))[\"openclip_vith\"][\"light_single\"][0];print(d[\"k\"])")
echo "[$(date)] selected k=$K"
[ "$K" -gt 0 ] || K=64
$PY scripts/zest_run.py zest_main --k $K
echo "[$(date)] ZEST DONE k=$K"
