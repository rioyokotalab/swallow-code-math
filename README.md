# Swallow Code Corpus

## References

### Llama-3.3-70B-Instruct

chat template: https://ollama.com/library/llama3.3/blobs/948af2743fc7

## swallow code v0.3

### Installation

Please see [scripts/install.sh](scripts/install.sh).

### Usage

```bash
qsub -g <group-name> scripts/Llama-3.3-70b-instruct.sh <index>
```

## swallow code v0.3 instruct

### Humaneval Style

ref: https://github.com/openai/human-eval/blob/master/data/example_problem.jsonl

```bash
qsub -g <group-name> scripts/synthetic_humaneval.sh <index>
```

### MT-Bench Style

ref: https://github.com/Stability-AI/FastChat/blob/jp-stable/fastchat/llm_judge/data/mt_bench/question.jsonl

```bash
qsub -g <group-name> scripts/synthetic_mtbench.sh <index>
```
