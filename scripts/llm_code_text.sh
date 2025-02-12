#!/bin/sh
#$ -cwd
#$ -l cpu_16=1
#$ -l h_rt=0:24:00:00
#$ -o outputs/convert/$JOB_ID.log
#$ -e outputs/convert/$JOB_ID.log
#$ -p -5


source .env/bin/activate

python src/llm_code_text.py \
  --input_file /gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.3-merged/swallow-code-v0.3-no-repet.jsonl \
  --output_file /gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.3-merged/swallow-code-v0.3-no-repetition-code-text.jsonl \
