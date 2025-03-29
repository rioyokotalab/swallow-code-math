#!/bin/sh
#PBS -q rt_HF
#PBS -N convert-parquet-jsonl
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/convert/

cd $PBS_O_WORKDIR
mkdir -p outputs/convert

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles

module load cuda/12.8
module load cudnn/9.7.0
module load nccl/2.25.1-cuda12.8
module load hpcx/2.21.0

source .env/bin/activate

INPUT_DIR="/groups/gcg51558/datasets/raw/instruct/OpenMathInstruct-2/data"
OUTPUT_DIR="/groups/gcg51558/datasets/raw/instruct/OpenMathInstruct-2/jsonl"

echo "Converting Arrow files to JSONL in $INPUT_DIR"

mkdir -p "$OUTPUT_DIR"

export TMPDIR="/groups/gcg51558/fujii/tmp"
export TMP="/groups/gcg51558/fujii/tmp"

export HF_CACHE="/groups/gcg51558/fujii/tmp"
export HF_HOME="/groups/gcg51558/fujii/tmp"

for INPUT_FILE in "$INPUT_DIR"/*.parquet; do
  FILENAME=$(basename "$INPUT_FILE")
  OUTPUT_FILE="$OUTPUT_DIR/${FILENAME%.parquet}.jsonl"

  if [ -f "$OUTPUT_FILE" ]; then
    echo "Skipping $INPUT_FILE"
    continue
  fi

  echo "Processing $INPUT_FILE"

  python src/tools/convert_parquet_to_jsonl.py \
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
