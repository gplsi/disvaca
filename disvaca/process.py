import argparse
import json
from disvaca import VACADiscriminator
from pathlib import Path

MODES = {
	"contains-va": "is_va",
	"contains-ca": "is_ca",
	"not-exclusively-va": "is_not_exclusively_from_va",
	"not-exclusively-ca": "is_not_exclusively_from_ca",
}


def get_text(record, line_number):
	if isinstance(record, str):
		return record
	if isinstance(record, dict) and isinstance(record.get("text"), str):
		return record["text"]
	raise ValueError(
		f"line {line_number}: expected a JSON string or an object with a string 'text' field"
	)


def output_path(input_path, mode):
	if input_path.suffix != ".jsonl":
		raise ValueError("input path must have a .jsonl suffix")
	return input_path.with_suffix(f".{mode}.jsonl")


def process_file(input_path, mode):
	output_file = output_path(input_path, mode)
	discriminator = VACADiscriminator.load()
	predicate = getattr(discriminator, MODES[mode])

	with input_path.open("r", encoding="utf-8") as source, output_file.open(
		"w", encoding="utf-8"
	) as destination:
		for line_number, line in enumerate(source, start=1):
			if not line.strip():
				continue
			record = json.loads(line)
			text = get_text(record, line_number)
			if predicate(text):
				destination.write(json.dumps(record, ensure_ascii=False))
				destination.write("\n")

	return output_file


def main():
	parser = argparse.ArgumentParser(description="Filter JSONL records by VA/CA words.")
	parser.add_argument("input", type=Path, help="Input .jsonl file")
	parser.add_argument("--mode", required=True, choices=MODES)
	args = parser.parse_args()

	output_file = process_file(args.input, args.mode)
	print(output_file)


if __name__ == "__main__":
	main()
