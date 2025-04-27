import argparse
import json
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Check repetition in text field of jsonl and filter by condition.")
    parser.add_argument("--input_path", help="Path to the input JSONL file")
    parser.add_argument("--output_path", help="Path to the output JSONL file")
    parser.add_argument(
        "--mode",
        choices=["include", "exclude"],
        help="Filtering mode: 'include' -> output only records with repetition, 'exclude' -> remove records with repetition",
    )
    parser.add_argument("-n", "--limit", type=int, default=None, help="Process only the first N records (optional)")
    return parser.parse_args()


def contains_repetition(text: str) -> bool:
    """
    If text meets the following conditions, return True.
      1. length of text is equal to or greater than 1000
      2. consecutive phrases of 100 characters or more are repeated
    """
    if len(text) < 1000:
        return False

    # 2. Consecutive phrases of 100 characters or more are repeated
    pattern = re.compile(r"(.{100,})(\1)+", re.DOTALL)

    match = pattern.search(text)
    if match:
        return True

    return False


def main():
    args = parse_args()

    input_path = args.input_path
    output_path = args.output_path
    mode = args.mode
    limit = args.limit

    count_processed = 0
    count_output = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if limit is not None and count_output >= limit:
                break

            line = line.strip()
            if not line:
                continue

            # jsonl: 1 record = 1 sample
            record = json.loads(line)
            text = record.get("improved_code", "")

            has_repetition = contains_repetition(text)
            if has_repetition:
                print(f"Detected repetition in record {count_processed}", flush=True)

            if mode == "include" and has_repetition:
                json.dump(record, fout, ensure_ascii=False)
                fout.write("\n")
                count_output += 1
            elif mode == "exclude" and not has_repetition:
                json.dump(record, fout, ensure_ascii=False)
                fout.write("\n")
                count_output += 1

            count_processed += 1


if __name__ == "__main__":
    main()
