import json
import argparse
from pathlib import Path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge(base_data, other_data):
    assert len(base_data) == len(other_data), "Mismatch in dataset lengths"

    merged = []
    for base_item, other_item in zip(base_data, other_data):
        merged.append({
            "source_text": base_item["article_text"],
            "triples": base_item["triples"],
            "reference_summary": base_item["reference_summary"],
            "summary_without_kg": base_item["generated_summary"],
            "summary_with_kg": other_item["generated_summary"]
        })
    return merged


def main(args):
    base_path = Path(args.base)
    other_path = Path(args.other)
    output_path = Path(args.output)

    base_data = load_json(base_path)
    other_data = load_json(other_path)

    assert len(base_data) == len(other_data), "Mismatch before sampling"

    # 🔹 Apply sampling if specified
    if args.num_samples is not None:
        n = min(args.num_samples, len(base_data))
        base_data = base_data[:n]
        other_data = other_data[:n]

    merged_data = merge(base_data, other_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Saved {len(merged_data)} samples → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two summarization JSON files")

    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--other", type=str, required=True)

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of samples to merge"
    )

    parser.add_argument("--output", type=str, default="merged.json")

    args = parser.parse_args()
    main(args)