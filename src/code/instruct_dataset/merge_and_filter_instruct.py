import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List


def validate_data(data: Dict) -> bool:
    # check necessary keys
    if 'input' not in data or 'output' not in data:
        return False

    if not isinstance(data['input'], dict) or not isinstance(data['output'], dict):
        return False

    if data['input'].get('role') != 'user' or data['output'].get('role') != 'assistant':
        return False

    if 'content' not in data['input'] or len(data['input']['content']) > 3500:
        return False

    return True

def transform_data(data: Dict) -> Dict:
    return {
        "input": [{"content": data['input']['content'], "role": "user"}],
        "output": {"content": data['output']['content'], "role": "assistant"}
    }

def process_jsonl_files(input_dir: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_file = os.path.join(output_dir, "code_merged.jsonl")

    file_pattern = os.path.join(input_dir, "mtbench_instruct_0*.jsonl")
    buffer_data = []
    total_processed = 0

    for file_path in glob.glob(file_pattern):
        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    if not validate_data(data):
                        continue

                    transformed_data = transform_data(data)
                    buffer_data.append(transformed_data)

                    if len(buffer_data) >= 10000:
                        save_batch(buffer_data, output_file, total_processed == 0)
                        total_processed += len(buffer_data)
                        print(f"Total processed: {total_processed} items")
                        buffer_data = []

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line in {file_path}: {e}")
                    continue

    if buffer_data:
        save_batch(buffer_data, output_file, total_processed == 0)
        total_processed += len(buffer_data)
        print(f"Final total processed: {total_processed} items")

def save_batch(data: List[Dict], output_file: str, is_first_batch: bool):
    mode = 'w' if is_first_batch else 'a'
    with open(output_file, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main() -> None:
    parser = argparse.ArgumentParser(description='Process JSONL files')
    parser.add_argument('--input_dir', required=True, help='Input directory containing JSONL files')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed files')

    args = parser.parse_args()
    process_jsonl_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
