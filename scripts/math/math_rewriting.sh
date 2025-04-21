#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:24:00:00
#$ -o outputs/math-rewriting/$JOB_ID.log
#$ -e outputs/math-rewriting/$JOB_ID.log
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

INPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/pretrain/finemath/finemath-4plus-jsonl"
OUTPUT_DIR="/gs/bs/hp190122/fujii/datasets/finemath/finemath-4plus-jsonl-rewriting"
mkdir -p "$OUTPUT_DIR"

INDEX=$1
FORMATTED_INDEX=$(printf "%05d" $INDEX)

BATCH_SIZE=512
echo "batch size: $BATCH_SIZE"

export TMPDIR="/gs/bs/hp190122/fujii/.cache"
export TMP="/gs/bs/hp190122/fujii/.cache"

# https://docs.vllm.ai/en/stable/serving/env_vars.html
export VLLM_CACHE_ROOT="/gs/bs/hp190122/fujii/.cache"

python src/math/math_rewriting.py \
  --model-path "/gs/bs/tga-NII-LLM/hf-checkpoints/gemma-3-27b-it" \
  --jsonl-path "$INPUT_DIR/train-$FORMATTED_INDEX-of-00064.jsonl" \
  --output-path "$OUTPUT_DIR/train-$FORMATTED_INDEX.jsonl" \
  --tensor-parallel 4 \
  --resume \
  --batch-size $BATCH_SIZE
