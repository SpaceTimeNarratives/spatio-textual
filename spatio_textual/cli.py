import argparse
import json
from pathlib import Path
from .annotate import annotate_text


def annotate_file(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as infile:
        records = [json.loads(line) for line in infile]

    results = []
    for record in records:
        text = record.get("text", "")
        result = annotate_text(text)
        record.update(result)
        results.append(record)

    with output_path.open("w", encoding="utf-8") as outfile:
        for record in results:
            json.dump(record, outfile)
            outfile.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Annotate spatial entities and verbs in text records.")
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file (one text record per line)")
    parser.add_argument("-o", "--output", required=True, help="Output file path for annotated JSONL")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    annotate_file(input_path, output_path)


if __name__ == "__main__":
    main()