import argparse
from datasets import load_dataset
import json
from datasets import disable_caching


def convert_parquet_to_jsonl(parquet_file_path: str, jsonl_file_path: str) -> None:
    dataset = load_dataset(path="parquet", data_files=parquet_file_path, split="train")

    with open(jsonl_file_path, "w", encoding="utf-8") as jsonl_file:
        for record in dataset:
            json.dump(record, jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")

    print(f"Parquet file '{parquet_file_path}' has been converted to JSONL and saved as '{jsonl_file_path}'.")


def main() -> None:
    disable_caching()

    parser = argparse.ArgumentParser(description="Convert Parquet files to JSONL format.")
    parser.add_argument("--parquet-file", type=str, help="Path to the input Parquet file.")
    parser.add_argument("--jsonl-file", type=str, help="Path to the output JSONL file.")
    args = parser.parse_args()

    convert_parquet_to_jsonl(parquet_file_path=args.parquet_file, jsonl_file_path=args.jsonl_file)


if __name__ == "__main__":
    main()
