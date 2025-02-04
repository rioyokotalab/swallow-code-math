#!/bin/sh
#$ -cwd
#$ -l cpu_16=1
#$ -l h_rt=0:24:00:00
#$ -o outputs/convert/$JOB_ID.log
#$ -e outputs/convert/$JOB_ID.log
#$ -p -5

# priority: -5: normal, -4: high, -3: highest

set -e

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

INPUT_DIR="/gs/bs/tgh-24IDU/datasets/open-coder/opc-fineweb-code-corpus/data"
OUTPUT_DIR="/gs/bs/tgh-24IDU/datasets/open-coder/opc-fineweb-code-corpus/jsonl"

echo "Converting Arrow files to JSONL in $INPUT_DIR"

mkdir -p "$OUTPUT_DIR"

export TMPDIR="/gs/bs/tge-gc24sp03/cache"
export TMP="/gs/bs/tge-gc24sp03/cache"

export HF_CACHE="/gs/bs/tge-gc24sp03/cache"
export HF_HOME="/gs/bs/tge-gc24sp03/cache"

for INPUT_FILE in "$INPUT_DIR"/*.parquet; do
  FILENAME=$(basename "$INPUT_FILE")
  OUTPUT_FILE="$OUTPUT_DIR/${FILENAME%.parquet}.jsonl"

  if [ -f "$OUTPUT_FILE" ]; then
    echo "Skipping $INPUT_FILE"
    continue
  fi

  echo "Processing $INPUT_FILE"

  python src/convert_parquet_to_jsonl.py \
    --parquet-file "$INPUT_FILE" \
    --jsonl-file "$OUTPUT_FILE"
done

wc -l "$OUTPUT_DIR"/*.jsonl

# merge jsonl files
MERGE_FILE="$OUTPUT_DIR/merged.jsonl"
if [ ! -f "$MERGE_FILE" ]; then
  echo "Merging JSONL files in $OUTPUT_DIR"
  cat "$OUTPUT_DIR"/*.jsonl > "$MERGE_FILE"
fi

wc -l "$MERGE_FILE"

echo "All done!"
