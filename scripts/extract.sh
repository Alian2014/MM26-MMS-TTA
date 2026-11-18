#!/bin/bash
# Remove old debug.txt files to avoid interfering with the current run
rm -rf log/eval/*/debug.txt
args=(
    --save_path ./features/vit_b16
    --data-root /path/to/datasets
    # --datasets A/V/R/S
    --datasets A
    # --datasets R
    # --datasets V
    # --datasets S
    # -- datasets food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101
    # --datasets eurosat
    # --datasets food101
    # --datasets ucf101
    # --datasets dtd
    # --datasets fgvc
    # --datasets sun397
    # --datasets caltech101
    # --datasets oxford_flowers
    # --datasets oxford_pets
    # --datasets stanford_cars
    --backbone ViT-B/16
    # --backbone RN50
    --views 64
)

# CUDA_VISIBLE_DEVICES=$1: GPU ID from first CLI arg
# python extract.py: feature extraction entry
# "${args[@]}": expand args into extract.py
CUDA_VISIBLE_DEVICES=$1 python extract.py "${args[@]}"
