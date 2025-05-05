import argparse
import json
import tempfile
import subprocess
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def check_syntax_error(code: str) -> list[str]:
    """
    Args:
        code (str): python code

    Returns:
        list[str]: py compile errors
    """
    issues = []

    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        issues.append(f"Syntax error: {str(e)}")

    return issues


def check_comment_ratio(code: str):
    total_lines = 0
    comment_lines = 0

    try:
        tokens = tokenize.generate_tokens(StringIO(code).readline)
        for token_type, _, _, _, _ in tokens:
            total_lines += 1
            if token_type == tokenize.COMMENT:
                comment_lines += 1

    except tokenize.TokenError as e:
        # when token error happens, exit calculating ratio.
        print(f"Token error encountered: {str(e)}")
        return 0
    except IndentationError as e:
        print(f"indentation error encountered {str(e)}")
        return 0

    if total_lines == 0:  # if code is empty
        return 0

    return comment_lines / total_lines


def apply_comment_penalty(score: float, comment_ratio: float) -> float:
    """
    add penalty to score based on comment ratio
    """
    if comment_ratio == 1.0:
        return 0.0
    elif comment_ratio > 0:
        penalty_factor = 1 - comment_ratio
        score *= penalty_factor
    return score


def check_code_quality(code: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(code.encode())
        temp_file.flush()

        result = subprocess.run(
            [
                "pylint",
                "--persistent=n",
                "--disable=E0401,C0114,C0301,C0103,C0116,C0411,R0903,W0511,C0412",
                temp_file.name,
            ],
            capture_output=True,
            text=True,
        )

    pylint_output = result.stdout
    score = None

    for line in pylint_output.split("\n"):
        if "Your code has been rated at" in line:
            score = float(line.split("/")[0].split()[-1])

    comment_ratio = check_comment_ratio(code)

    if score is not None:
        score = apply_comment_penalty(score, comment_ratio)

    return score, pylint_output


def process_code_entry(data, is_static_code_analysis):
    if data.get("language") == "Python" and "text" in data:
        code = data["text"]

        issues = check_syntax_error(code) if is_static_code_analysis else []
        data["analysis_results"] = issues
        data["has_issues"] = len(issues) > 0
        data["language_type_issue"], data["language_type"] = check_language_issues(code)

        pylint_score, pylint_output = check_code_quality(code)
        data["pylint_score"] = pylint_score
        data["pylint_output"] = pylint_output

        return data

    return None


import tokenize
from io import StringIO
import re


def check_language_issues(code: str):
    issues = []
    language_type = "English or Japanese"

    allowed_characters = re.compile(r"^[\u0020-\u007E\u3000-\u30FF\u4E00-\u9FFF\uFF66-\uFF9F\s\n\t]*$")

    try:
        tokens = tokenize.generate_tokens(StringIO(code).readline)
        for token_type, token_string, _, _, _ in tokens:
            if token_type in {tokenize.STRING, tokenize.COMMENT}:
                if not allowed_characters.match(token_string):
                    truncated_token_string = token_string if len(token_string) <= 100 else token_string[:100] + "..."
                    issues.append(f"Non-Japanese/English characters found in: {truncated_token_string}")
                    language_type = "Contains non-English/Japanese characters"
                    break
    except tokenize.TokenError as e:
        issues.append(f"Token error: {str(e)}")
        language_type = "TokenError in code"
    except IndentationError as e:
        issues.append(f"Indentation error: {str(e)}")
        language_type = "IndentationError in code"

    return issues, language_type


def write_batch_to_file(output_file, result_data):
    with open(output_file, "a", encoding="utf-8") as out_f:
        for entry in result_data:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def process_jsonl_file(
    jsonl_file: str,
    output_file: str,
    is_static_code_analysis: bool = False,
    batch_size: int = 50000,
    max_workers: int = 1,
    resume_from: int = 0,
) -> None:
    result_data = []
    count = resume_from

    progress_file = f"{output_file}.progress"

    with open(jsonl_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with (
        open(jsonl_file, "r", encoding="utf-8") as f,
        tqdm(total=total_lines, initial=resume_from, desc="Processing JSONL") as pbar,
    ):
        for _ in range(resume_from):
            next(f)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for line in f:
                data = json.loads(line)
                futures.append(executor.submit(process_code_entry, data, is_static_code_analysis))

                count += 1
                if count % batch_size == 0:
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            result_data.append(result)
                    write_batch_to_file(output_file, result_data)
                    result_data = []
                    futures = []

                    save_progress(progress_file, count)
                pbar.update(1)

            for future in as_completed(futures):
                result = future.result()
                if result:
                    result_data.append(result)

    if result_data:
        write_batch_to_file(output_file, result_data)

    if os.path.exists(progress_file):
        os.remove(progress_file)


def save_progress(progress_file: str, count: int) -> None:
    with open(progress_file, "w") as f:
        json.dump({"processed_lines": count}, f)


def load_progress(progress_file: str) -> int:
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            data = json.load(f)
            return data.get("processed_lines", 0)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Analyze Python scripts in JSONL file")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--static-code-analysis", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4, help="Max number of threads for parallel processing")
    args = parser.parse_args()

    progress_file = f"{args.output_file}.progress"
    resume_from = load_progress(progress_file)

    process_jsonl_file(
        jsonl_file=args.input_file,
        output_file=args.output_file,
        is_static_code_analysis=args.static_code_analysis,
        max_workers=args.max_workers,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    main()
