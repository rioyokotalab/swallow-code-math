#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:1:00:00
#$ -o outputs/rewrite/$JOB_ID.log
#$ -e outputs/rewrite/$JOB_ID.log
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

INPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.3-jsonl-self-contained"
OUTPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.4-jsonl-rewritten"

mkdir -p "$OUTPUT_DIR"

INDEX=$1
FORMATTED_INDEX=$(printf "%04d" $INDEX)

BATCH_SIZE=32
echo "batch size: $BATCH_SIZE"

export TMPDIR="/gs/bs/hp190122/fujii/.cache"
export TMP="/gs/bs/hp190122/fujii/.cache"

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# https://github.com/vllm-project/vllm/issues/6152#issuecomment-2211709345

python src/code/pretrain_dataset/code_index_not_implemented.py \
  --model-path "/gs/bs/tga-NII-LLM/hf-checkpoints/Llama-3.3-70B-Instruct" \
  --jsonl-path "$INPUT_DIR/split_$FORMATTED_INDEX.jsonl" \
  --output-path "$OUTPUT_DIR/train_$FORMATTED_INDEX.jsonl" \
  --tensor-parallel 4 \
  --resume \
  --batch-size $BATCH_SIZE
