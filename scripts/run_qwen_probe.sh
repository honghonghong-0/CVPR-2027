#!/bin/bash
# After E3b finishes: Qwen vision features -> E2 probe -> transfer projection + CIs (GPU1 only).
P=.; cd $P
PY=python
export HF_HOME=./cache/hf CVPR_GPU=1
while pgrep -u ubuntu -f "scripts/e3_binary.py" >/dev/null; do sleep 60; done
echo "[$(date)] e3b done, start probe"
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)" -ge 20000 ]; do sleep 60; done
PYTHONPATH=$P/pylib_vlm:$P/pylib $PY scripts/extract_qwen_vis.py renders_diag feats_diag
PYTHONPATH=$P/pylib_vlm:$P/pylib $PY scripts/extract_qwen_vis.py renders_xfer feats_xfer
export PYTHONPATH=$P/pylib
$PY scripts/analyze_e2.py feats_diag results/pairs_diag.json results/e2_diag_withqwen.json
$PY scripts/analyze_xfer.py feats_xfer renders_xfer results/xfer_withqwen.json
$PY scripts/analyze_xfer_proj.py feats_xfer renders_xfer feats_diag results/pairs_diag.json results/xfer_proj_withqwen.json
$PY scripts/boot_proj.py results/xfer_proj_withqwen.json results/xfer_proj_ci_withqwen.json
echo "[$(date)] ALL DONE"
