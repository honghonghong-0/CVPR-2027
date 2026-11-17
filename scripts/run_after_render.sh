#!/bin/bash
# Wait for diag (1980) + xfer (900) renders, then run magnitude, features, E2, transfer analysis.
P=.; cd $P
export PYTHONPATH=$P/pylib HF_HOME=./cache/hf TORCH_HOME=./cache/torch
PY=python
until [ "$(ls renders_diag/*.png 2>/dev/null | wc -l)" -ge 1980 ] && [ "$(ls renders_xfer/*.png 2>/dev/null | wc -l)" -ge 900 ]; do sleep 120; done
echo "[$(date)] renders done"
$PY scripts/pair_magnitude.py renders_diag results/pairs_diag.json
$PY scripts/extract_feats.py renders_diag feats_diag 12
$PY scripts/extract_feats.py renders_xfer feats_xfer 12
$PY scripts/analyze_e2.py feats_diag results/pairs_diag.json results/e2_diag.json
$PY scripts/analyze_xfer.py feats_xfer renders_xfer results/xfer.json
echo "[$(date)] ALL DONE"
