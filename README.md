![Swallow Code and Swallow Math](assets/swallow-code-math-logo.png)

# Swallow Code and Swallow Math

The effectiveness of large language models (LLMs) in mathematical reasoning and program synthesis is fundamentally limited by the quality of their pre-training corpora.
We present two openly licensed datasets derived from public data under the Llama-3.1 License.
**SwallowCode** (approximately 16.1 B tokens) is constructed by sequential syntax filtering, pylint-based style filtering, and a two-stage LLM rewriting procedure that first enforces style conformity and then rewrites each snippet into a self-contained, algorithmically efficient example.
**SwallowMath** (approximately 2.3 B tokens) is obtained from Finemath-4+ through an LLM rewriting step that removes boiler-plate, restores missing context, and reformats solutions into concise, step-by-step explanations.

SwallowCode: https://huggingface.co/datasets/tokyotech-llm/swallow-code

SwallowMath: https://huggingface.co/datasets/tokyotech-llm/swallow-math

## Data Pipeline 🧹

### Swallow Code

#### Programming Language Filter

#### Python Syntax Error Filter

#### Linter Filter

#### LLM Rewriting (SGCR)

#### LLM Rewriting (SGCR)

### Swallow Math

## Ablation Experiments

### Training

### Evaluation

## License 📜



## Citation 
```bibtex
@article{swallow-code-math,
  title={Rewriting Pre-Training Data: Boosting LLM Performance in Math and Code},
  author={Kazuki Fujii},
  journal={},
  year={2025}
}
```
