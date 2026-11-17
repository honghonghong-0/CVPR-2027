#!/bin/bash
P=.; cd $P
export PYTHONPATH=pylib:pylib_vlm HF_HOME=./cache/hf CUDA_VISIBLE_DEVICES=1
PY=python
while pgrep -f "unzip -qo train_jpg" >/dev/null; do sleep 30; done
echo "[$(date)] unzip done"
$PY scripts/ml_feats.py feats_ml 300
$PY scripts/ml_analyze.py feats_ml results/ml_real.json
echo "[$(date)] ML DONE"
