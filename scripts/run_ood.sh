#!/bin/bash
# Remove old debug.txt files to avoid interfering with the current run
rm -rf log/eval/*/debug.txt
args=(
    # --mode tda
    --mode mmstta
    --config configs
    --datasets A
    # --datasets R
    # --datasets V
    # --datasets S
    --backbone ViT-B/16
    --features-root './features/vit_b16'
    # --backbone RN50
    # --features-root './features/rn50'
    --views 64
    --cupl-root './descriptions/CuPL_prompts_{}.json'
    --tau_gate 20
    --exp_name $2
)

# CUDA_VISIBLE_DEVICES=$1: GPU ID from first CLI arg
# python runner.py: main entry
# "${args[@]}": expand args into runner.py
CUDA_VISIBLE_DEVICES=$1 python runner.py "${args[@]}"
