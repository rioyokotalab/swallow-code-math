import argparse
import json
import pyarrow as pa
import pyarrow.ipc as ipc

def convert_arrow_to_jsonl(arrow_file_path: str, jsonl_file_path: str):
    """
    Arrow ファイルを読み込み、JSONL 形式で出力します。
    """
    # Arrow ファイルをメモリマップでオープン
    with pa.memory_map(arrow_file_path, 'r') as source:
        try:
            # ファイルフォーマットの Arrow ファイルとして読み込み
            reader = ipc.open_file(source)
        except pa.ArrowInvalid:
            # ファイルがストリーム形式の場合はこちらを利用
            reader = ipc.open_stream(source)

        with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
            # 各 RecordBatch ごとに処理
            for batch in reader:
                # RecordBatch をリスト（各行を辞書として表現）に変換
                rows = batch.to_pylist()
                for row in rows:
                    json.dump(row, jsonl_file, ensure_ascii=False)
                    jsonl_file.write('\n')

    print(f"Arrow file '{arrow_file_path}' has been converted to JSONL and saved as '{jsonl_file_path}'.")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Apache Arrow files to JSONL format."
    )
    parser.add_argument(
        "--arrow-file", type=str, required=True,
        help="Path to the input Arrow file."
    )
    parser.add_argument(
        "--jsonl-file", type=str, required=True,
        help="Path to the output JSONL file."
    )
    args = parser.parse_args()

    convert_arrow_to_jsonl(
        arrow_file_path=args.arrow_file, jsonl_file_path=args.jsonl_file
    )

if __name__ == "__main__":
    main()
