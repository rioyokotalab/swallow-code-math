#!/bin/sh

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

pip install --upgrade pip
pip install --upgrade wheel cmake ninja packaging

pip install -r requirements.txt

pip install vllm
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6
