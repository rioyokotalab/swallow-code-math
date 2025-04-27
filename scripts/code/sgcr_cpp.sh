#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:24:00:00
#$ -o outputs/llm-filter/llama-3.3-70b-instruct/$JOB_ID.log
#$ -e outputs/llm-filter/llama-3.3-70b-instruct/$JOB_ID.log
#$ -p -3

# priority: -5: normal, -4: high, -3: highest

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

INPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/pretrain/"
OUTPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.3-cpp"

mkdir -p "$OUTPUT_DIR"

INDEX=$1
FORMATTED_INDEX=$(printf "%04d" $INDEX)

BATCH_SIZE=2048
echo "batch size: $BATCH_SIZE"

export TMPDIR="/gs/bs/tge-gc24sp03/cache"
export TMP="/gs/bs/tge-gc24sp03/cache"

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# https://github.com/vllm-project/vllm/issues/6152#issuecomment-2211709345

python src/code/pretrain_dataset/sgcr_cpp.py \
  --model-path "/gs/bs/tga-NII-LLM/hf-checkpoints/Llama-3.3-70B-Instruct" \
  --jsonl-path "$INPUT_DIR/split_$FORMATTED_INDEX.jsonl" \
  --output-path "$OUTPUT_DIR/cpp_split_$FORMATTED_INDEX.jsonl" \
  --tensor-parallel 4 \
  --resume \
  --batch-size $BATCH_SIZE
