import argparse
import json
import os
import re
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """You are a smart software engineer. Please provide a question sentence corresponding to the following implementation code.
"""
EXAMPLE_USER_PROMPT = '''[Case 1]:
Implementation:
```python
def solution(lst: list[int]) -> int:
    """
    Input:
        - lst: list[int]

    Output:
        - int: the sum of all of the odd elements that are in even positions
    """
    return sum(lst[i] for i in range(0, len(lst), 2) if lst[i] % 2 != 0)
```
'''
EXAMPLE_ASSISTANT_PROMPT = """Question:
Write Python program to calculate the sum of all of the odd elements that are in even positions in the given list lst.
"""


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_results(data, output_path, mode="w"):
    with open(output_path, mode, encoding="utf-8") as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")


def main(args: argparse.Namespace) -> None:
    # Initialize the LLM
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=102688,
    )

    # Load and process the JSONL file
    data = load_jsonl(args.jsonl_path)

    # Determine the starting index
    start_index = 0
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as file:
            for line in file:
                last_processed = json.loads(line)
                start_index = last_processed.get("index", 0) + 1
        print(f"Resuming from index {start_index}")
    else:
        # Clear the output file if not resuming
        with open(args.output_path, "w", encoding="utf-8") as file:
            file.write("")

    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.7,
        max_tokens=8192,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    processed_data = []
    batch_size = args.batch_size
    batches = [data[i : i + batch_size] for i in range(start_index, len(data), batch_size)]

    for batch_idx, batch in enumerate(batches):
        start = time.perf_counter()
        texts = []
        improved_codes = []
        for item in batch:
            if "improved_code" not in item:
                continue

            code_text: str = "[Case 2]:\nImplementation:\n```python\n" + item["improved_code"] + "```"

            messages: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": EXAMPLE_USER_PROMPT},
                {"role": "assistant", "content": EXAMPLE_ASSISTANT_PROMPT},
                {"role": "user", "content": code_text},
            ]
            text: str = tokenizer.apply_chat_template(  # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)
            improved_codes.append(item["improved_code"])

        outputs = llm.generate(texts, sampling_params)

        for i, output in enumerate(outputs):
            output_text = output.outputs[0].text
            improved_code_text = improved_codes[i]

            if output_text is None or len(output_text) <= 50:
                continue
            if output_text.startswith("Question:\n"):
                output_text = output_text[10:]

            if args.add_code_block:
                improved_code_text = f"```python\n{improved_code_text}```"

            write_item = {
                "input": {"role": "user", "content": output_text},
                "output": {"role": "assistant", "content": improved_code_text},
                "conversation": [
                    {"role": "user", "content": output_text},
                    {"role": "assistant", "content": improved_code_text},
                ],
                "text": tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": output_text},
                        {"role": "assistant", "content": improved_code_text},
                    ],
                    tokenize=False,
                ),
            }

            if args.verbose:
                print(
                    f"Input: {output_text}\nOutput: {improved_code_text}\n\n\n",
                    flush=True,
                )

            processed_data.append(write_item)

        print(
            f"Processed batch {batch_idx + 1} in {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if len(processed_data) >= batch_size * 2:
            write_results(processed_data, args.output_path, mode="a")
            processed_data = []

    # Write any remaining processed data
    if processed_data:
        write_results(processed_data, args.output_path, mode="a")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scoring dataset by language model")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--jsonl-path", help="Path to the input JSONL file")
    parser.add_argument("--output-path", help="Path to save the output JSONL file with Japanese entries")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--resume", action="store_true", help="Resume from the last processed index")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--add-code-block", action="store_true", help="Add code block")

    args = parser.parse_args()
    main(args=args)
