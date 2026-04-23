import torch
from transformers import LongT5ForConditionalGeneration, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import json
import warnings
from utils import smart_chunk_text

warnings.filterwarnings("ignore")

MODEL_NAME = "google/long-t5-tglobal-base"
DEVICE = "cpu"

test_dataset = load_from_disk("./dataset/pubmed_with_triples_v/test")
print(f"Test samples: {len(test_dataset)}")

generation_config = {
    "max_new_tokens": 300,
    "num_beams": 4,
    "min_length": 50,
    "length_penalty": 2.0,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "repetition_penalty": 2.0,
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = LongT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
base_model.to(DEVICE)
base_model.eval()

def generate_summary_base(article_text: str, model, tokenizer, config: dict, max_length: int = 4096):
    inputs = smart_chunk_text(article_text, tokenizer, max_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    num_chunks = input_ids.shape[0]

    if num_chunks == 1:
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **config)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    chunk_summaries = []
    with torch.no_grad():
        for i in range(num_chunks):
            out = model.generate(
                input_ids=input_ids[i:i+1], attention_mask=attention_mask[i:i+1], **config
            )
            chunk_summaries.append(tokenizer.decode(out[0], skip_special_tokens=True))

    combined = " ".join(chunk_summaries)
    combined_tokens = tokenizer(combined, return_tensors="pt")

    if combined_tokens["input_ids"].shape[1] > max_length:
        return generate_summary_base(combined, model, tokenizer, config, max_length)

    polish_config = {**config, "max_new_tokens": 256, "repetition_penalty": 1.3, "no_repeat_ngram_size": 2}
    with torch.no_grad():
        final_out = model.generate(
            input_ids=combined_tokens["input_ids"].to(model.device), 
            attention_mask=combined_tokens["attention_mask"].to(model.device), 
            **polish_config
        )
    return tokenizer.decode(final_out[0], skip_special_tokens=True)

results = []

num_samples = 20
for i in tqdm(range(num_samples), desc="Generating summaries"):
    article_text = test_dataset[i]["article"]
    triples = test_dataset[i].get("rebel_triples", "")
    reference_summary = test_dataset[i].get("abstract", "")
    
    try:
        generated_summary = generate_summary_base(
            article_text="summarize:" + article_text,
            model=base_model,
            tokenizer=tokenizer,
            config=generation_config
        )
        
        results.append({
            "index": i,
            "article_text": article_text,
            "triples": triples,
            "reference_summary": reference_summary,
            "generated_summary": generated_summary,
            "status": "success"
        })
        
    except Exception as e:
        results.append({
            "index": i,
            "article_text": article_text,
            "triples": triples,
            "reference_summary": reference_summary,
            "generated_summary": "",
            "status": f"error: {str(e)}"
        })

RESULT_DIR = "./results"
with open(f"./results/base_model_test_results_{num_samples}.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nCompleted {len(results)} summaries")
print(f"Results saved to base_model_test_results_{num_samples}.json")

for i in range(min(3, len(results))):
    print(f"\n--- Sample {i+1} ---")
    print(f"Reference: {results[i]['reference_summary'][:200]}...")
    print(f"Generated: {results[i]['generated_summary'][:200]}...")
    print(f"Status: {results[i]['status']}")