#!/bin/sh
#PBS -q rt_HF
#PBS -N stack-edu
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=18:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/stack-edu/

cd $PBS_O_WORKDIR
mkdir -p outputs/stack-edu

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles

module load cuda/12.8
module load cudnn/9.7.0
module load nccl/2.25.1-cuda12.8
module load hpcx/2.21.0

source .env/bin/activate

INPUT_DIR="/groups/gcg51558/datasets/raw/pretrain/stack-edu/Rust/jsonl"
OUTPUT_DIR="/groups/gcg51558/datasets/raw/pretrain/stack-edu/Rust/refactor-stage-1"

mkdir -p "$OUTPUT_DIR"

INDEX=11
FORMATTED_INDEX=$(printf "%05d" $INDEX)

BATCH_SIZE=1024
echo "batch size: $BATCH_SIZE"

export TMPDIR="/groups/gcg51558/fujii/tmp"
export TMP="/groups/gcg51558/fujii/tmp"

export HF_CACHE="/groups/gcg51558/fujii/tmp"
export HF_HOME="/groups/gcg51558/fujii/tmp"

# https://docs.vllm.ai/en/stable/serving/env_vars.html
export VLLM_CACHE_ROOT="/groups/gcg51558/fujii/tmp"

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# https://github.com/vllm-project/vllm/issues/6152#issuecomment-2211709345

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python src/code/pretrain_dataset/sgcr_rust.py \
  --model-path "/groups/gcg51558/hf_checkpoints/Llama-3.3-70B-Instruct" \
  --jsonl-path "$INPUT_DIR/data-$FORMATTED_INDEX-of-00012.arrow.jsonl" \
  --output-path "$OUTPUT_DIR/rust_split_$FORMATTED_INDEX.jsonl" \
  --tensor-parallel 8 \
  --resume \
  --batch-size $BATCH_SIZE
