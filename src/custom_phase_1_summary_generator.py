import torch
from transformers import LongT5ForConditionalGeneration, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import json
import warnings
import argparse
from utils import smart_chunk_text
from base_model import KATSum
from kg_embedder import KGEncoder

warnings.filterwarnings("ignore")

def generate_summary_custom(article_text, triples, model, tokenizer, generation_config, max_length):
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
                **generation_config
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    chunk_summaries = []
    with torch.no_grad():
        for i in range(num_chunks):
            out = model.generate_summary(
                input_ids=input_ids[i:i+1],
                attention_mask=attention_mask[i:i+1],
                triples=triples,
                **generation_config
            )
            chunk_summaries.append(tokenizer.decode(out[0], skip_special_tokens=True))

    combined = " ".join(chunk_summaries)
    combined_tokens = tokenizer(combined, return_tensors="pt")

    if combined_tokens["input_ids"].shape[1] > max_length:
        return generate_summary_custom(combined, triples, model, tokenizer, generation_config, max_length)

    polish_config = {
        **generation_config,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/long-t5-tglobal-base")
    parser.add_argument("--dataset_path", type=str, default="./dataset/pubmed_with_triples_v/test")
    parser.add_argument("--result_dir", type=str, default="./results/")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/phase1_best.pt")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_sidecar_layers", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--min_length", type=int, default=50)
    parser.add_argument("--length_penalty", type=float, default=2.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "min_length": args.min_length,
        "length_penalty": args.length_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "early_stopping": args.early_stopping,
        "repetition_penalty": args.repetition_penalty,
    }

    print("Loading test dataset...")
    test_dataset = load_from_disk(args.dataset_path)
    print(f"Test samples: {len(test_dataset)}")

    print(f"\nLoading model from {args.checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = LongT5ForConditionalGeneration.from_pretrained(args.model_name)

    kg_embedder = KGEncoder(
        encoder=base_model.encoder,
        tokenizer=tokenizer,
        hidden_dim=base_model.config.d_model,
        device=args.device,
    )

    custom_model = KATSum(
        base_model=base_model,
        kg_embedder=kg_embedder,
        num_sidecar_layers=args.num_sidecar_layers,
        freeze_base=True,
        device=args.device,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    custom_model.kg_sidecar_layers.load_state_dict(checkpoint["kg_sidecar_state_dict"])
    custom_model.kg_embedder.load_state_dict(checkpoint["kg_embedder_state_dict"])

    custom_model.to(args.device)
    custom_model.eval()

    print(f"Model loaded successfully on {args.device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")
    print(f"ROUGE-1: {checkpoint['val_rouge1']:.4f}")
    print(f"ROUGE-2: {checkpoint['val_rouge2']:.4f}")
    print(f"ROUGE-L: {checkpoint['val_rougeL']:.4f}\n")

    results = []

    for i in tqdm(range(args.num_samples), desc="Generating summaries"):
        article_text = test_dataset[i]["article"]
        triples = test_dataset[i].get("rebel_triples", "")
        reference_summary = test_dataset[i].get("abstract", "")

        try:
            generated_summary = generate_summary_custom(
                "summarize:" + article_text,
                triples,
                custom_model,
                tokenizer,
                generation_config,
                args.max_length
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

    output_file = args.result_dir + f"katsum_phase_1_model_test_results_{args.num_samples}.json"

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

if __name__ == "__main__":
    main()