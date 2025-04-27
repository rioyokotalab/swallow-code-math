import argparse
import json


def process_jsonl(input_file: str, output_file: str) -> None:
    """

    Args:
        input_file (str):
        output_file (str):
    """
    try:
        processed_count = 0

        with (
            open(input_file, "r", encoding="utf-8") as fin,
            open(output_file, "w", encoding="utf-8", buffering=8192) as fout,
        ):
            for line in fin:
                try:
                    record = json.loads(line.strip())
                    if "improved_code" in record and "generated_text" in record and "text" in record:
                        new_record = {
                            "text": "You are a smart software engineer. Please evaluate the following code on a scale of 1 to 10 and provide feedback on how it can be improved. \n\n## Evaluation Code\n```python\n"
                            + record["text"]
                            + "\n```\n\n## Feedback\n"
                            + record["generated_text"],
                        }
                        fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                        processed_count += 1

                        if processed_count % 10000 == 0:
                            print(f"Progress: {processed_count} records processed...")

                except json.JSONDecodeError as e:
                    print(f"Warning: invalid line: {line.strip()[:100]}...")
                    continue

        print(f"Processed complicated: {processed_count} records.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="extract improved_code from jsonl file and save to another jsonl file"
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)

    args = parser.parse_args()
    process_jsonl(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
