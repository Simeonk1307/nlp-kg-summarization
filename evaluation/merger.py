import json

DATA_DIR = "../src/results/"
with open(DATA_DIR+'base_model_test_results.json', 'r', encoding='utf-8') as f:
    base_data = json.load(f)

with open(DATA_DIR+'katsum_phase_1_model_test_results.json', 'r', encoding='utf-8') as f:
    custom_data = json.load(f)

merged_data = []

num_samples = len(base_data)

for base_item, custom_item in zip(base_data, custom_data):
    merged_item = {
        "source_text": base_item["article_text"],
        "triples": base_item["triples"],
        "reference_summary": base_item["reference_summary"],
        "summary_without_kg": base_item["generated_summary"],
        "summary_with_kg": custom_item["generated_summary"]
    }
    merged_data.append(merged_item)

with open(f'./summaries_{num_samples}.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=2)