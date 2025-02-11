#!/bin/bash

input_dir="/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.3-no-repet"
output_dir="/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.3-merged"

mkdir -p $output_dir

python src/merge.py \
  --input-dir $input_dir \
  --output-path $output_dir/swallow-code-v0.3-llm-refactor-no-repet.jsonl
