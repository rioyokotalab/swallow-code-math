import argparse
import json
import os
import re
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


instruction = """You are a smart software engineer. Please evaluate the following code on a scale of 1 to 10 based on the following criteria:\n
1. Are variable names descriptive and consistent with naming conventions?
2. Are comments and doc-strings appropriately written to explain the purpose and functionality of the code?
3. Are type annotations used effectively where applicable?
4. Are functions appropriately modularized, with well-defined responsibilities and clear separation of concerns?
5. Are variables' lifetimes intentionally managed, avoiding frequent reassignment or overly long scopes?
6. Is error handling implemented appropriately where necessary?
7. Is the code properly indented and follows standard formatting guidelines?
8. Do comments provide context and rationale, rather than merely describing what the code does?
9. Are functions and classes designed with clear, single responsibilities?
10. Is the code formatted in a way that enhances readability?\n\n
And provide suggestions for improvement based on the evaluation criteria. You can also provide an improved version of the code like the following style:\n
### Evaluation: 7\n\n
### Suggestions:\n
    Provide specific, actionable suggestions to improve the code based on the evaluation criteria.\n\n
### Improved Code:\n
Provide a revised version of the code incorporating the suggested improvements.\n
```python\n
def improved_function(arg1: int, arg2: str) -> str:
    # Your improved code here
    pass
```\n\n
"""


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_results(data, output_path, mode="w"):
    with open(output_path, mode, encoding="utf-8") as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")


def parse_sections(text):
    """
    analyze scoring results, and parse them into sections

    Args:
        text (str): analyzed text

    Returns:
        dict: parsed sections
    """
    sections = {}
    current_section = None
    current_content = []

    for line in text.split("\n"):
        if line.startswith("###"):
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = line[4:].strip()  # eliminate ###
            current_content = []
        else:
            current_content.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_content).strip()

    results = {}

    for section, content in sections.items():
        if section.startswith("Evaluation:"):
            score = section.split(":")[1].strip()
            try:
                score = float(score)
            except ValueError:
                score = -1
            results["evaluation_score"] = score

        elif section == "Suggestions:":
            results["suggestions"] = [s.strip() for s in content.split("\n") if s.strip()]

        elif section == "Improved Code:":
            code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
            if code_match:
                results["improved_code"] = code_match.group(1).strip()
        else:
            # other sections
            results[section.lower().replace(" ", "_")] = content

    return results


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
        for item in batch:
            code_text: str = item["text"]
            messages: list[dict[str, str]] = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": code_text},
            ]
            text: str = tokenizer.apply_chat_template(  # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        outputs = llm.generate(texts, sampling_params)

        for i, output in enumerate(outputs):
            output_text = output.outputs[0].text
            if args.verbose:
                print(output_text, flush=True)

            results = parse_sections(output_text)
            item = batch[i]
            for key, value in results.items():
                if key == "evaluation_score":
                    item["score"] = value or -1
                elif key == "suggestions":
                    item["suggestions"] = value or []
                elif key == "improved_code":
                    item["improved_code"] = value or ""
                else:
                    item[key] = value

            item["generated_text"] = output_text
            item["index"] = start_index + batch_idx * batch_size + i  # Adjust index to match the original data
            processed_data.append(item)

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

    args = parser.parse_args()
    main(args=args)
