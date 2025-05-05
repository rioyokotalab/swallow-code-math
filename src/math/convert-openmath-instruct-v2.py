import json
import os
import argparse
import glob
from typing import Any


def convert_json_to_output_format(json_data: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extracts problem and generated_solution fields from JSON object
    and creates a new JSON object in the desired output format.
    """
    problem = json_data.get("problem", "")
    solution = json_data.get("generated_solution", "")
    if not problem or not solution:
        return None

    output_json = json_data
    output_json["text"] = "[Question]\n" + problem + "\n[Answer]\n" + solution

    return output_json


def process_files(input_dir: str, output_path: str) -> None:
    """
    Process all JSONL files in the input directory and merge them into a single output JSONL file.
    """
    # Make sure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    # Find all JSONL files in the input directory
    jsonl_files = glob.glob(os.path.join(input_dir, "train_000*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return

    all_converted_entries = []

    # Process each JSONL file
    for jsonl_file in jsonl_files:
        filename = os.path.basename(jsonl_file)
        print(f"Processing {filename}...")

        # Read the JSONL file
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    # Parse JSON line
                    json_data = json.loads(line.strip())

                    # Convert to output format
                    output_json = convert_json_to_output_format(json_data)
                    if output_json is None:
                        continue

                    # Add to converted entries list
                    all_converted_entries.append(output_json)

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_number} in {filename}: {e}")
                except Exception as e:
                    print(f"Error processing line {line_number} in {filename}: {e}")

        print(f"Processed {filename}")

    # Create the directory for the merged output file if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write all converted entries to the merged output JSONL file
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in all_converted_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Merged {len(all_converted_entries)} entries into {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL files to a new JSONL format and merge them")
    parser.add_argument("--input_dir", required=True, help="Directory containing JSONL files")
    parser.add_argument("--output_path", required=True, help="Path for the merged output JSONL file")

    args = parser.parse_args()

    try:
        process_files(args.input_dir, args.output_path)
        print("Conversion completed successfully")
    except Exception as e:
        print(f"Error during conversion: {e}")


if __name__ == "__main__":
    main()
