import argparse
import json
from typing import Dict, Any


def process_jsonl(input_file: str, output_file: str) -> None:
    """
    JSONLファイルから 'improved_code' キーのデータのみを抽出し、
    'text' キーとして保存する高速な実装

    Args:
        input_file (str): 入力JSONLファイルのパス
        output_file (str): 出力JSONLファイルのパス
    """
    try:
        # バッファリングのために配列を使用せず、直接書き込み
        processed_count = 0

        with (
            open(input_file, "r", encoding="utf-8") as fin,
            open(output_file, "w", encoding="utf-8", buffering=8192) as fout,
        ):  # バッファサイズを指定
            for line in fin:
                try:
                    record = json.loads(line.strip())
                    if "improved_code" in record:
                        # 新しい形式のデータを作成
                        new_record = {"text": record["improved_code"]}
                        # 直接ファイルに書き込み
                        fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                        processed_count += 1

                        # 定期的に進捗を表示（オプション）
                        if processed_count % 10000 == 0:
                            print(f"進捗: {processed_count}件処理済み")

                except json.JSONDecodeError as e:
                    print(f"警告: 不正なJSONライン: {line.strip()[:100]}...")  # 長すぎる行は省略
                    continue

        print(f"処理完了: {processed_count}件のレコードを保存しました")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="JSONLファイルから improved_code キーのデータを抽出し、text キーとして保存"
    )
    parser.add_argument("--input_file", required=True, help="入力JSONLファイルのパス")
    parser.add_argument("--output_file", required=True, help="出力JSONLファイルのパス")

    args = parser.parse_args()
    process_jsonl(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
