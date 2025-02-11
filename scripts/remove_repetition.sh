#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=0:24:00:00
#$ -o outputs/remove_repetition/$JOB_ID.log
#$ -e outputs/remove_repetition/$JOB_ID.log
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
source /home/1/ug02141/.bashrc

INPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.3-jsonl"
OUTPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.3-no-repet"

# 出力ディレクトリがない場合は作成
mkdir -p "${OUTPUT_DIR}"

process_file() {
  local input_file=$1
  local rel_path=${input_file#"${INPUT_DIR}"/}
  local output_file="${OUTPUT_DIR}/${rel_path}"
  local output_dir=$(dirname "${output_file}")

  mkdir -p "${output_dir}"
  python src/remove_repetition.py \
    --input_path "${input_file}" \
    --output_path "${output_file}" \
    --mode exclude
}

export -f process_file
export INPUT_DIR
export OUTPUT_DIR

find "${INPUT_DIR}" -name "*.jsonl" |
  parallel --jobs 80 \
    --retries 3 \
    --progress \
    --joblog "${OUTPUT_DIR}/parallel_job.log" \
    process_file {}
