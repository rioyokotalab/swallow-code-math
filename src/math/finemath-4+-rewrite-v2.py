import argparse
import json
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """You are tasked with enhancing the quality of a dataset containing mathematical, logical, and programming problems.
Rewrite the given problem according to the following 10 rules. Execute the rules in the specified order and format the output as indicated.

**Rules**:
1. Remove any irrelevant timestamps, headers, or footers.
2. Convert the problem into a problem-answer pair format.
3. Explain the solution process step-by-step, explicitly including formulas and descriptions.
4. Unify all mathematical expressions in LaTeX notation (enclosed in $$ $$).
5. For programming problems, implement solutions in Python 3 or C++.
6. Do not include meta-descriptions such as "Here are the problems...".
7. Begin with a problem overview (Task Description) and the solution strategy (Approach).
8. Emphasize the final answer using $$ \\boxed{{}} $$ .
9. If verification is possible, perform a check and describe the process.
10. If the problem is too simple or inappropriate, revise it to an appropriate difficulty level.

**Output Format**:
**Task Description**: [Overview of the problem]
**Approach**: [Solution strategy]
**Solution Steps**:
- Step 1: [Description and LaTeX formula]
- Step 2: [Description and LaTeX formula]
- Step 3: [Description and LaTeX formula]
- ...
**Final Answer**: $$ \\boxed{{[Final Answer]}} $$

**Example**:
Input: [2023-10-01] Solve 2x + 3 = 7
Output:
**Task Description**: Solve the linear equation 2x + 3 = 7.
**Approach**: Subtract the constant from both sides and isolate x.
**Solution Steps**:
- Step 1: Subtract 3 from both sides: $$ 2x + 3 - 3 = 7 - 3 $$, yielding $$ 2x = 4 $$.
- Step 2: Divide both sides by 2: $$ \\frac{{2x}}{{2}} = \\frac{{4}}{{2}} $$, yielding $$ x = 2 $$.
- Step 3: Verification: $$ 2(2) + 3 = 4 + 3 = 7 $$, which is correct.
**Final Answer**: $$ \\boxed{{2}} $$
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
        max_model_len=131072,
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
            messages: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["text"]},
            ]
            text: str = tokenizer.apply_chat_template(  # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if len(text) > 102688:
                print(f"Skipping text with length {len(text)}")
                continue
            texts.append(text)

        outputs = llm.generate(texts, sampling_params)

        for i, output in enumerate(outputs):
            output_text = output.outputs[0].text

            if output_text is None or len(output_text) <= 50:
                continue

            write_item = {
                "text": output_text,
            }

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
