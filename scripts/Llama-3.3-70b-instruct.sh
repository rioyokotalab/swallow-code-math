#!/bin/sh
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=0:24:00:00
#$ -o outputs/llm-filter/llama-3.3-70b-instruct/$JOB_ID.log
#$ -e outputs/llm-filter/llama-3.3-70b-instruct/$JOB_ID.log
#$ -p -5

# priority: -5: normal, -4: high, -3: highest

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

INPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.1-3-split-jsonl"
OUTPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.3-jsonl"

mkdir -p "$OUTPUT_DIR"

INDEX=$1
FORMATTED_INDEX=$(printf "%04d" $INDEX)

python src/code_score_refactor.py \
  --model-path "/gs/bs/tga-NII-LLM/hf-checkpoints/Llama-3.3-70B-Instruct" \
  --jsonl-path "$INPUT_DIR/split_$FORMATTED_INDEX.jsonl" \
  --output-path "$OUTPUT_DIR/python_scoring_Llama-3.3-70B-split_$FORMATTED_INDEX.jsonl" \
  --verbose \
  --tensor-parallel 2 \
  --resume
