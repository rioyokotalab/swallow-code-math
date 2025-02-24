import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List


def validate_data(data: Dict) -> bool:
    """
    データのバリデーションを行う
    """
    # 必要なキーの存在確認
    if 'input' not in data or 'output' not in data:
        return False

    # dict型であることを確認
    if not isinstance(data['input'], dict) or not isinstance(data['output'], dict):
        return False

    # roleの確認
    if data['input'].get('role') != 'user' or data['output'].get('role') != 'assistant':
        return False

    # contentの長さ確認
    if 'content' not in data['input'] or len(data['input']['content']) > 3500:
        return False

    return True

def transform_data(data: Dict) -> Dict:
    """
    データを指定された形式に変換する
    """
    return {
        "input": [{"content": data['input']['content'], "role": "user"}],
        "output": {"content": data['output']['content'], "role": "assistant"}
    }

def process_jsonl_files(input_dir: str, output_dir: str):
    """
    JSONLファイルを処理し、変換したデータを保存する
    """
    # 出力ディレクトリの作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 出力ファイルのパス
    output_file = os.path.join(output_dir, "code_merged.jsonl")

    # すべての対象ファイルを取得
    file_pattern = os.path.join(input_dir, "mtbench_instruct_0*.jsonl")
    buffer_data = []
    total_processed = 0

    # ファイルの読み込みと処理
    for file_path in glob.glob(file_pattern):
        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    # バリデーション
                    if not validate_data(data):
                        continue

                    # データの変換
                    transformed_data = transform_data(data)
                    buffer_data.append(transformed_data)

                    # 10,000件ごとにファイルに保存
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

    # 残りのデータを保存
    if buffer_data:
        save_batch(buffer_data, output_file, total_processed == 0)
        total_processed += len(buffer_data)
        print(f"Final total processed: {total_processed} items")

def save_batch(data: List[Dict], output_file: str, is_first_batch: bool):
    """
    データバッチをJSONLファイルとして保存する
    mode: 最初のバッチは'w'、それ以降は'a'
    """
    mode = 'w' if is_first_batch else 'a'
    with open(output_file, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Process JSONL files')
    parser.add_argument('--input_dir', required=True, help='Input directory containing JSONL files')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed files')

    args = parser.parse_args()
    process_jsonl_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
