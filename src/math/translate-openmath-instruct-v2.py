import argparse
from email import message
import json
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """You are a smart engineer. Please translate the following mathematical sentence into Japanese. Please make sure simply translating the sentence, Not solving the problem.
"""


def parse_code(text):
    """
    Extract a single code block enclosed in triple backticks from text.

    Args:
        text (str): The input text containing a code block.

    Returns:
        str: The extracted code block without the backtick delimiters,
             or an empty string if no code block is found.
    """
    # Find the starting position of the first triple backticks
    start_pos = text.find("```")
    if start_pos == -1:
        # No code block found
        return ""

    # Find the end of the first line with backticks (to skip language identifier)
    first_line_end = text.find("\n", start_pos)
    if first_line_end == -1:
        # No newline after opening backticks
        return ""

    # Find the closing triple backticks
    end_pos = text.find("```", first_line_end)
    if end_pos == -1:
        # No closing backticks
        return ""

    # Extract the content between the backticks, excluding the markers themselves
    code_content = text[first_line_end + 1 : end_pos]

    # Remove leading/trailing whitespace
    return code_content.strip()


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
        for item in batch:
            if "problem" not in item or "generated_solution" not in item:
                continue

            messages: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["problem"]},
            ]
            problem_prompt: str = tokenizer.apply_chat_template(  # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if len(problem_prompt) > 102688:
                continue

            messages: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["generated_solution"]},
            ]
            solution_prompt: str = tokenizer.apply_chat_template(  # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if len(solution_prompt) > 102688:
                continue

            item["problem_prompt"] = problem_prompt
            item["solution_prompt"] = solution_prompt
            texts.append(item)

        problem_prompts = [item["problem_prompt"] for item in texts]
        solution_prompts = [item["solution_prompt"] for item in texts]

        problem_translations = llm.generate(problem_prompts, sampling_params)
        solution_translations = llm.generate(solution_prompts, sampling_params)

        for index, translated_problem, translated_solution in zip(
            range(len(texts)), problem_translations, solution_translations
        ):
            translated_problem = translated_problem.outputs[0].text
            translated_solution = translated_solution.outputs[0].text

            if translated_problem is None or translated_solution is None:
                continue
            if len(translated_problem) <= 50 or len(translated_solution) <= 50:
                continue

            write_item = texts[index]
            write_item["translated_problem"] = translated_problem
            write_item["translated_solution"] = translated_solution

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

    args = parser.parse_args()
    main(args=args)
