#!/usr/bin/env python3
import argparse
import json
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Check repetition in text field of jsonl and filter by condition.")
    parser.add_argument("--input_path", help="Path to the input JSONL file")
    parser.add_argument("--output_path", help="Path to the output JSONL file")
    parser.add_argument("--mode", choices=["include", "exclude"],
                        help="Filtering mode: 'include' -> output only records with repetition, 'exclude' -> remove records with repetition")
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="Process only the first N records (optional)")
    return parser.parse_args()

def contains_repetition(text: str) -> bool:
    """
    以下2条件を満たせば True (反復あり) を返す:
      1. 全体の文字数が1000以上
      2. 連続した重複箇所(full_repeated)が100文字以上
    判定のために以下の正規表現を利用:
      - 連続するフレーズ(\1)の繰り返しを検出
    """
    # 1. 全体の文字数が1000以上かどうか
    if len(text) < 1000:
        return False

    # 2. 重複箇所(full_repeated)が100文字以上か
    pattern = re.compile(r'(.{100,})(\1)+', re.DOTALL)

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

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if limit is not None and count_output >= limit:
                break

            line = line.strip()
            if not line:
                continue

            # JSONLなので1行=1レコード
            record = json.loads(line)
            text = record.get("improved_code", "")

            # 重複検出
            has_repetition = contains_repetition(text)
            if has_repetition:
                print(f"Detected repetition in record {count_processed}", flush=True)

            # 出力モードに応じてフィルタ
            # include -> has_repetition == True のみ出力
            # exclude -> has_repetition == False のみ出力
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
