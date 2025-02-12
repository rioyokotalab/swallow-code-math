import argparse
import json
from typing import Dict, Any


def process_jsonl(input_file: str, output_file: str, min_score: float = 6.0) -> None:
    """
    score が指定値以上のものを抽出し、'text' キーとして保存する

    Args:
        input_file (str): 入力JSONLファイルのパス
        output_file (str): 出力JSONLファイルのパス
        min_score (float): 抽出する最小スコア（デフォルト: 6.0）
    """
    try:
        processed_count = 0
        extracted_count = 0

        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8', buffering=8192) as fout:

            for line in fin:
                processed_count += 1
                try:
                    record = json.loads(line.strip())

                    if 'text' in record and 'score' in record:
                        score = float(record['score'])  # 数値型に変換

                        if score >= min_score:
                            new_record = {'text': record['text']}
                            fout.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                            extracted_count += 1

                            # 進捗表示（10000件ごと）
                            if extracted_count % 10000 == 0:
                                print(f"進捗: {processed_count}件中{extracted_count}件抽出済み")

                except json.JSONDecodeError as e:
                    print(f"警告: 不正なJSONライン: {line.strip()[:100]}...")
                    continue
                except (ValueError, TypeError) as e:
                    print(f"警告: スコアの変換エラー: {str(e)}")
                    continue

        print(f"処理完了: 全{processed_count}件中{extracted_count}件のレコードを保存しました")
        print(f"抽出率: {(extracted_count/processed_count*100):.2f}%")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='JSONLファイルからスコアが基準値以上のデータを抽出'
    )
    parser.add_argument('--input_file', required=True, help='入力JSONLファイルのパス')
    parser.add_argument('--output_file', required=True, help='出力JSONLファイルのパス')
    parser.add_argument('--min_score', type=float, default=6.0,
                      help='抽出する最小スコア（デフォルト: 6.0）')

    args = parser.parse_args()
    process_jsonl(args.input_file, args.output_file, args.min_score)


if __name__ == '__main__':
    main()
