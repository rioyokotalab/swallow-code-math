import argparse
import os
import json


def merge_jsonl_files(
    input_dir: str,
    output_file: str,
) -> None:
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in os.listdir(input_dir):
            if filename.endswith(".jsonl") or filename.endswith(".json"):
                filepath = os.path.join(input_dir, filename)
                with open(filepath, "r", encoding="utf-8") as infile:
                    for line in infile:
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON in {filepath}")
                            continue

                        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Merged JSONL files saved to {output_file}")


def main():
    # 引数をパースする
    parser = argparse.ArgumentParser(description="Merge JSONL files with optional filters.")
    parser.add_argument("--input-dir", required=True, help="Directory containing JSONL files to merge")
    parser.add_argument("--output-path", required=True, help="Output file path for the merged JSONL file")

    args = parser.parse_args()

    merge_jsonl_files(
        args.input_dir,
        args.output_path,
    )


if __name__ == "__main__":
    main()
