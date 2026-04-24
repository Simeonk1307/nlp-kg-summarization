import torch
from transformers import LongT5ForConditionalGeneration, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import json
import warnings
import argparse
from utils import smart_chunk_text

warnings.filterwarnings("ignore")

def generate_summary(article_text, model, tokenizer, generation_config, max_length):
    inputs = smart_chunk_text(article_text, tokenizer, max_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    num_chunks = input_ids.shape[0]

    if num_chunks == 1:
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_config)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    chunk_summaries = []
    with torch.no_grad():
        for i in range(num_chunks):
            out = model.generate(
                input_ids=input_ids[i:i+1],
                attention_mask=attention_mask[i:i+1],
                **generation_config
            )
            chunk_summaries.append(tokenizer.decode(out[0], skip_special_tokens=True))

    combined = " ".join(chunk_summaries)
    combined_tokens = tokenizer(combined, return_tensors="pt")

    if combined_tokens["input_ids"].shape[1] > max_length:
        return generate_summary(combined, model, tokenizer, generation_config, max_length)

    polish_config = {
        **generation_config,
        "max_new_tokens": 256,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 2
    }

    with torch.no_grad():
        final_out = model.generate(
            input_ids=combined_tokens["input_ids"].to(model.device),
            attention_mask=combined_tokens["attention_mask"].to(model.device),
            **polish_config
        )

    return tokenizer.decode(final_out[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/long-t5-tglobal-base")
    parser.add_argument("--dataset_path", type=str, default="./dataset/pubmed_with_triples_v/test")
    parser.add_argument("--result_dir", type=str, default="./results/")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
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

    test_dataset = load_from_disk(args.dataset_path)
    print(f"Test samples: {len(test_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = LongT5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    results = []

    for i in tqdm(range(len(test_dataset)), desc="Generating summaries"):
        article_text = test_dataset[i]["article"]
        triples = test_dataset[i].get("rebel_triples", "")
        reference_summary = test_dataset[i].get("abstract", "")

        try:
            generated_summary = generate_summary(
                "summarize:" + article_text,
                model,
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

    output_path = args.result_dir + f"base_model_test_results_{args.num_samples}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCompleted {len(results)} summaries")
    print(f"Results saved to {output_path}")

    for i in range(min(3, len(results))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Reference: {results[i]['reference_summary'][:200]}...")
        print(f"Generated: {results[i]['generated_summary'][:200]}...")
        print(f"Status: {results[i]['status']}")

if __name__ == "__main__":
    main()