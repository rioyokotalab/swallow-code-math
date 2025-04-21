#!/bin/bash

input_dir="/gs/bs/hp190122/fujii/datasets/finemath/finemath-4plus-jsonl-rewriting"
output_dir="/gs/bs/hp190122/fujii/datasets/finemath/finemath-4plus-merged"

mkdir -p $output_dir

python src/tools/merge.py \
  --input-dir $input_dir \
  --output-path $output_dir/swallow-math-rewriting.jsonl
