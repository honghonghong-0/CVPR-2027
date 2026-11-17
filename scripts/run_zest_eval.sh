#!/bin/bash
P=.; cd $P
export PYTHONPATH=pylib_zest:pylib:pylib_vlm HF_HOME=./cache/hf CUDA_VISIBLE_DEVICES=1
until grep -q "ZEST DONE" logs/zest_main.log; do sleep 60; done
python scripts/zest_eval.py zest_main results/zest_eval.json
echo "[$(date)] EVAL DONE"
