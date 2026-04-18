import os

# os.environ['HF_HOME'] = './models/'
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "0"

import json
import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset, DatasetDict, load_from_disk

from kg_extractor import KGExtractor, Triple

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extract_triples.log"),
    ],
)
log = logging.getLogger(__name__)


# Batch extraction using KGExtractor
def triples_for_batch(
    texts: list[str],
    extractor: KGExtractor,
    max_input_tokens: int,
    long_strategy: str,
) -> list[list[Triple]]:
    """
    Return triples for every text in `texts` using KGExtractor.
    """
    results = []

    for text in texts:
        # KGExtractor.extract() chunks internally when text exceeds 512 tokens
        raw_triples = extractor.extract_chunk_batch(text)  # list of (head, rel, tail) tuples

        results.append(raw_triples)

    return results


# Dataset loading
def load_pubmed():
    dataset = load_dataset("ccdv/pubmed-summarization")
    
    # SAVE_DIR = "/scratch/b112301046-unnikrishnan/pubmed_cached"
    
    # dataset = load_from_disk(SAVE_DIR)

    # print(f"\nSaving dataset to {SAVE_DIR} ...")
    # dataset.save_to_disk(SAVE_DIR)

    return dataset

# Main pipeline
def main(args):
    extractor = KGExtractor(None)  # auto-detects cuda/cpu

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU VRAM: {vram_gb:.1f} GB")
    else:
        log.warning("CUDA not available — running on CPU (will be slow).")

    splits = load_pubmed()
    log.info(f"Splits found: {list(splits.keys())}")
    
    splits = DatasetDict({
        "train": splits["train"].select(range(20000)),
        "validation": splits["validation"].select(range(2500)),  # keep full — only 6,633
        "test": splits["test"].select(range(2500)),              # keep full — only 6,658
    })

    os.makedirs(args.output_path, exist_ok=True)

    if args.trial:
        splits = DatasetDict({k: v.select(range(1)) for k, v in splits.items()})
        log.info("TRIAL MODE: 1 example per split.")

    for split_name, split_ds in splits.items():
        log.info(f"\n{'═'*60}")
        log.info(f"Split: {split_name}  ({len(split_ds):,} examples)")
        log.info(f"{'═'*60}")

        ckpt_path = Path(args.output_path) / f".{split_name}_checkpoint.jsonl"

        # resume support
        done_triples: list[list[Triple]] = []
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                done_triples = [json.loads(line)["triples"] for line in f]
            log.info(f"Resuming from checkpoint: {len(done_triples):,} already done.")

        skip_count = 0
        ckpt_file = open(ckpt_path, "a")

        try:
            idx = len(done_triples)
            remaining = split_ds.select(range(idx, len(split_ds)))
            batch_texts = []

            def flush_batch(texts: list[str]):
                nonlocal skip_count
                batch_results = triples_for_batch(
                    texts,
                    extractor,
                    args.max_input_tokens,
                    args.long_strategy,
                )

                for tr in batch_results:
                    done_triples.append(tr)
                    ckpt_file.write(json.dumps({"triples": tr}) + "\n")
                    if not tr and args.long_strategy == "skip":
                        skip_count += 1

                ckpt_file.flush()

            for row in remaining:
                text = row.get(args.text_column, "") or ""
                batch_texts.append(text)

                if len(batch_texts) == args.batch_size:
                    flush_batch(batch_texts)
                    processed = len(done_triples)
                    log.info(
                        f"  {processed:>8,} / {len(split_ds):,}  "
                        f"({100*processed/len(split_ds):.1f}%)  "
                        f"skipped={skip_count}"
                    )
                    batch_texts = []

            if batch_texts:
                flush_batch(batch_texts)

        finally:
            ckpt_file.close()

        log.info(f"Done. Total skipped (too long): {skip_count:,}")

        # build enriched dataset
        log.info("Building enriched dataset")

        # Safety: align lengths
        done_triples = done_triples[: len(split_ds)]
        while len(done_triples) < len(split_ds):
            done_triples.append([])

        enriched = split_ds.add_column("rebel_triples", done_triples)

        def add_text_with_triples(example):
            triples_str = "; ".join(
                f"({t[0]}, {t[1]}, {t[2]})" for t in example["rebel_triples"]
            )
            article = example.get(args.text_column, "") or ""
            example["text_with_triples"] = (
                f"[Triples]: {triples_str}\n\n[Article]: {article}"
                if triples_str
                else f"[Article]: {article}"
            )
            return example

        enriched = enriched.map(
            add_text_with_triples,
            num_proc=4,
            desc="Formatting text_with_triples",
        )

        # Save HF Dataset 
        split_out = str(Path(args.output_path) / split_name)
        enriched.save_to_disk(split_out)
        log.info(f"  Saved HF dataset to {split_out}")

        # Save JSONL backup
        jsonl_out = str(Path(args.output_path) / f"{split_name}.jsonl")
        enriched.to_json(jsonl_out)
        log.info(f"  Saved JSONL to  {jsonl_out}")

        ckpt_path.unlink(missing_ok=True)  # clean up checkpoint after success

    log.info("\n All splits processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-extract REBEL triples for PubMed to save for LongT5 fine-tuning."
    )

    parser.add_argument(
        "--output_path",
        default="./datasets/pubmed_with_triples",
        help="Where to write the enriched dataset.",
    )

    parser.add_argument(
        "--text_column", default="article", help="Column containing article text."
    )

    parser.add_argument("--batch_size", type=int, default=16, help="Texts per batch.")

    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=512,
        help="Token ceiling : articles above this are handled by --long_strategy.",
    )

    parser.add_argument(
        "--long_strategy",
        choices=["skip", "chunk"],
        default="skip",
        help=(
            "What to do when an article exceeds --max_input_tokens.\n"
            "skip: store empty triple list (fast)\n"
            "chunk: KGExtractor handles sliding-window internally (slower, more complete)"
        ),
    )

    parser.add_argument(
        "--trial",
        action="store_true",
        help="Run on 1 example per split for quick testing.",
    )

    args = parser.parse_args()

    main(args)
