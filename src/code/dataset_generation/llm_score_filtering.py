import argparse
import json
from typing import Dict, Any


def process_jsonl(input_file: str, output_file: str, min_score: float = 6.0) -> None:
    """
    Args:
        input_file (str):
        output_file (str):
        min_score (float): (default: 6.0)
    """
    try:
        processed_count = 0
        extracted_count = 0

        with (
            open(input_file, "r", encoding="utf-8") as fin,
            open(output_file, "w", encoding="utf-8", buffering=8192) as fout,
        ):
            for line in fin:
                processed_count += 1
                try:
                    record = json.loads(line.strip())

                    if "text" in record and "score" in record:
                        score = float(record["score"])

                        if score >= min_score:
                            new_record = {"text": record["text"]}
                            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                            extracted_count += 1

                            if extracted_count % 10000 == 0:
                                print(f"Progress: {extracted_count} records extracted in {processed_count} records")

                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON line: {line.strip()[:100]}...")
                    continue
                except (ValueError, TypeError) as e:
                    print(f"Warning: score convert error: {str(e)}")
                    continue

        print(f"Processed Completed: All {processed_count} records | {extracted_count} records processed")
        print(f"Processed Rate: {(extracted_count / processed_count * 100):.2f}%")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Filter JSONL records by score")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--min_score", type=float, default=6.0)

    args = parser.parse_args()
    process_jsonl(args.input_file, args.output_file, args.min_score)


if __name__ == "__main__":
    main()
