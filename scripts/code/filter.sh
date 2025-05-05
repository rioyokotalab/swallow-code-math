#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=10:50:00
#$ -o outputs/filter/$JOB_ID.log
#$ -e outputs/filter/$JOB_ID.log
#$ -p -5
set -e

source .env/bin/activate

ID=$1
FORMAT_ID=$(printf "%05d" $ID)

INPUTDIR="/gs/bs/tga-bayes-crest/Swallow/raw/the-stack-v2-train-smol-ids/the-stack-v2-train-smol-ids-outputs/train-$FORMAT_ID-of-00064"
OUTPUTDIR="/gs/bs/tgh-24IDU/datasets/raw/pretrain/stack_v2_python"

mkdir -p "$OUTPUTDIR"

for INPUT_FILE in "$INPUTDIR"/*.jsonl; do
  FILENAME=$(basename "$INPUT_FILE")
  OUTPUT_FILE="$OUTPUTDIR/train-$FORMAT_ID-${FILENAME%.jsonl}-filtered.jsonl"

  echo "Processing $INPUT_FILE"

  python src/code/pretrain_dataset/filtering.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --static-code-analysis
done
