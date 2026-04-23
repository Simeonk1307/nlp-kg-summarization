import torch
from transformers import LongT5ForConditionalGeneration, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import json
import warnings
from utils import smart_chunk_text
from base_model import KATSum
from kg_embedder import KGEncoder

warnings.filterwarnings("ignore")

MODEL_NAME = "google/long-t5-tglobal-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/phase1_best.pt"
NUM_SIDECAR_LAYERS = 12

generation_config = {
    "max_new_tokens": 300,
    "num_beams": 4,
    "min_length": 50,
    "length_penalty": 2.0,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "repetition_penalty": 2.0,
}

print("Loading test dataset...")
test_dataset = load_from_disk("./dataset/pubmed_with_triples_v/test")
print(f"Test samples: {len(test_dataset)}")

print(f"\nLoading model from {CHECKPOINT_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = LongT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

kg_embedder = KGEncoder(
    encoder=base_model.encoder,
    tokenizer=tokenizer,
    hidden_dim=base_model.config.d_model,
    device=DEVICE,
)

custom_model = KATSum(
    base_model=base_model,
    kg_embedder=kg_embedder,
    num_sidecar_layers=NUM_SIDECAR_LAYERS,
    freeze_base=True,
    device=DEVICE,
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
custom_model.kg_sidecar_layers.load_state_dict(checkpoint["kg_sidecar_state_dict"])
custom_model.kg_embedder.load_state_dict(checkpoint["kg_embedder_state_dict"])

custom_model.to(DEVICE)
custom_model.eval()

print(f"Model loaded successfully on {DEVICE}")
print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Val loss: {checkpoint['val_loss']:.4f}")
print(f"ROUGE-1: {checkpoint['val_rouge1']:.4f}")
print(f"ROUGE-2: {checkpoint['val_rouge2']:.4f}")
print(f"ROUGE-L: {checkpoint['val_rougeL']:.4f}\n")


def generate_summary_custom(article_text: str, triples: str, model, tokenizer, config: dict, max_length: int = 4096):
    inputs = smart_chunk_text(article_text, tokenizer, max_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    num_chunks = input_ids.shape[0]

    if num_chunks == 1:
        with torch.no_grad():
            out = model.generate_summary(
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples,
                **config
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    chunk_summaries = []
    with torch.no_grad():
        for i in range(num_chunks):
            out = model.generate_summary(
                input_ids=input_ids[i:i+1],
                attention_mask=attention_mask[i:i+1],
                triples=triples,
                **config
            )
            chunk_summaries.append(tokenizer.decode(out[0], skip_special_tokens=True))

    combined = " ".join(chunk_summaries)
    combined_tokens = tokenizer(combined, return_tensors="pt")

    if combined_tokens["input_ids"].shape[1] > max_length:
        return generate_summary_custom(combined, triples, model, tokenizer, config, max_length)

    polish_config = {
        **config,
        "max_new_tokens": 256,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 2
    }
    
    with torch.no_grad():
        final_out = model.generate_summary(
            input_ids=combined_tokens["input_ids"].to(model.device),
            attention_mask=combined_tokens["attention_mask"].to(model.device),
            triples=triples,
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
        generated_summary = generate_summary_custom(
            article_text= "summarize:" + article_text,
            triples=triples,
            model=custom_model,
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

RESULT_DIR="./results"
output_file = RESULT_DIR+f"katsum_phase_1_model_test_results_{num_samples}.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nCompleted {len(results)} summaries")
print(f"Results saved to {output_file}")

print("\n" + "="*80)
print("SAMPLE OUTPUTS")
print("="*80)

for i in range(min(3, len(results))):
    print(f"\n--- Sample {i+1} ---")
    print(f"Status: {results[i]['status']}")
    print(f"\nReference ({len(results[i]['reference_summary'].split())} words):")
    print(results[i]['reference_summary'][:200] + "...")
    print(f"\nGenerated ({len(results[i]['generated_summary'].split())} words):")
    print(results[i]['generated_summary'][:200] + "...")
    print(f"Triples: {results[i]['triples'][:150]}...")
    print("-"*80)