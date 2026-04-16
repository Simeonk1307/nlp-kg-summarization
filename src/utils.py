import torch
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from kg_extractor import KGExtractor, Triple
from base_model import KATSum
from rouge_score.rouge_scorer import RougeScorer
import numpy as np


class SummarizationDataset(Dataset):
    """
    Holds tokenized articles, summaries and their extracted KG triples.

    Args:
        articles: List of source article strings.
        summaries: List of target summary strings.
        tokenizer: LongT5 tokenizer.
        extractor: KGExtractor instance (from kg_extractor.py).
        src_max_len: Maximum token length for source articles. LongT5 supports up to 16,384.
        tgt_max_len: Maximum token length for summaries.
        cache_triples: If True, extract and cache triples once.
                       Extraction is slow when using REBEL model, so caching saves time.
    """

    def __init__(
        self,
        articles: List[str],
        summaries: List[str],
        triples: List[List[Triple]],
        tokenizer,
        src_max_len: int = 4096,  # maximum tokens for article
        tgt_max_len: int = 512,  # maximum tokens for summary
    ):
        assert len(articles) == len(
            summaries
        ), "articles and summaries must be the same length"

        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

        # Tokenize all articles
        # T5 and LongT5 expect a task prefix
        # For summarization it's "summarize: "
        print("Tokenizing articles...")
        prefixed = ["summarize: " + a for a in articles]
        src_encoding = tokenizer(
            prefixed,
            max_length=src_max_len,
            padding=False,  # padding done later
            truncation=True,
            return_tensors=None,  # Return plain Python lists, not tensors
        )
        self.input_ids = src_encoding["input_ids"]
        self.attention_masks = src_encoding["attention_mask"]

        # Tokenize all summaries
        print("Tokenizing summaries...")
        tgt_encoding = tokenizer(
            summaries,
            max_length=tgt_max_len,
            padding=False,
            truncation=True,
            return_tensors=None,
        )
        self.labels = tgt_encoding["input_ids"]
        self.triples = triples

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Return one example as a dictionary.
        All values are Python lists and not tensors yet.
        collate_fn will convert them to tensors.
        """
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
            "triples": self.triples[idx],
        }


def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict:
    """
    Takes a list of individual examples and stacks them into batch tensors.

    For labels, T5 uses -100 so the loss function ignores positions where the label is -100.
    So labels are padded with -100, not with the pad token.

    Args:
        batch: List of dicts from SummarizationDataset.__getitem__()
        pad_token_id: The tokenizer's padding token ID (for input padding)

    Returns:
        Dict with keys: input_ids, attention_mask, labels, triples_batch
    """

    # Find the maximum lengths in this batch
    max_src_len = max(len(item["input_ids"]) for item in batch)
    max_tgt_len = max(len(item["labels"]) for item in batch)

    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []
    triples_batch = []

    for item in batch:
        src_len = len(item["input_ids"])
        tgt_len = len(item["labels"])

        # Pad input_ids and attention_mask on the right
        src_pad = max_src_len - src_len
        input_ids_batch.append(item["input_ids"] + [pad_token_id] * src_pad)
        attention_mask_batch.append(item["attention_mask"] + [0] * src_pad)

        # Pad labels with -100
        tgt_pad = max_tgt_len - tgt_len
        labels_batch.append(item["labels"] + [-100] * tgt_pad)

        triples_batch.append(item["triples"])

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
        "triples_batch": triples_batch,  # stays as a list, KG Embedder will handle it
    }


def train_one_epoch(
    model: KATSum,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    logger,
    device: str = "cpu",
    grad_accumulation_steps: int = 4,
) -> float:
    """
    Train for one full pass over the dataset.

    Args:
        model: KATSum instance.
        dataloader: PyTorch's DataLoader wrapping the training dataset.
        optimizer: AdamW optimizer over trainable parameters only.
        scheduler: Learning rate scheduler.
        device: "cuda" or "cpu".
        grad_accumulation_steps: How many batches to accumulate gradients over before calling optimizer.step().

    Returns:
        Average loss over the epoch.
    """

    model.train()
    total_loss = 0.0
    num_batches = 0

    # Reset gradients from previous epoch
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Move tensors to the right device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        triples_batch = batch["triples_batch"]  # stays on CPU since it's a list

        # Forward pass — returns (loss, logits)
        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            triples_batch=triples_batch,
        )

        # Normalise loss by accumulation steps
        loss = loss / grad_accumulation_steps

        # Backward pass, computes gradients for all requires_grad parameters
        loss.backward()

        if num_batches % 100 == 0:  # Check every 100 batches to avoid cluttering logs

            def log_grad(name, param):
                if param is None:
                    logger.warning(f"{name}: parameter is None")
                    return
                if param.grad is None:
                    logger.warning(f"{name}: grad is None")
                    return

                grad = param.grad
                logger.info(
                    f"{name}: mean={grad.mean().item():.8f}, "
                    f"std={grad.std().item():.8f}, "
                    f"norm={grad.norm().item():.8f}"
                )

            embedder = model.kg_embedder
            log_grad("Embedder.projection.weight", embedder.projection.weight)
            log_grad("Embedder.layer_norm.weight", embedder.layer_norm.weight)

            for i in range(model.num_sidecar_layers):
                sidecar = model.kg_sidecar_layers[i]
                log_grad(f"Sidecar[{i}].fusion_gate.weight", sidecar.fusion_gate.weight)
                log_grad(f"Sidecar[{i}].layer_norm.weight", sidecar.layer_norm.weight)

        # Only update weights every grad_accumulation_steps batches
        if (step + 1) % grad_accumulation_steps == 0:
            # Clip gradients to prevent exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += (
            loss.item() * grad_accumulation_steps
        )  # undo normalisation for logging

        num_batches += 1

        if step % 10 == 0:
            logger.info(
                f"Step {step}/{len(dataloader)}  loss={loss.item()*grad_accumulation_steps:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: KATSum,
    dataloader: DataLoader,
    tokenizer,
    device: str,
    rouge_scorer: RougeScorer,
    num_examples: int = 1500,
    max_new_tokens: int = 512,
) -> Dict:
    """
    Evaluate on validation set - compute loss and ROUGE scores using batched generation.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    num_count = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        triples_batch = batch["triples_batch"]

        # Compute loss
        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            triples_batch=triples_batch,
        )

        total_loss += loss.item()
        num_batches += 1

        # Batched generation for ROUGE
        if num_count >= num_examples:
            continue

        # Slice off only as many examples as we still need
        remaining = num_examples - num_count
        slice_end = min(remaining, input_ids.shape[0])

        input_ids_slice = input_ids[:slice_end]
        attention_mask_slice = attention_mask[:slice_end]
        labels_slice = labels[:slice_end]
        triples_slice = triples_batch[:slice_end]

        # Single batched generate call instead of a per-example loop
        generated_ids = model.generate_summary_batch(
            input_ids=input_ids_slice,
            attention_mask=attention_mask_slice,
            triples_batch=triples_slice,
            max_new_tokens=max_new_tokens,
        )

        # Batch-decode generated summaries
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Batch-decode references (replace -100 before decoding)
        ref_ids = labels_slice.clone()
        ref_ids[ref_ids == -100] = tokenizer.pad_token_id
        reference_texts = tokenizer.batch_decode(
            ref_ids, skip_special_tokens=True
        )

        # Score all examples in the slice 
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            rg_scores = rouge_scorer.score(ref_text, gen_text)
            rouge1_scores.append(rg_scores["rouge1"].fmeasure)
            rouge2_scores.append(rg_scores["rouge2"].fmeasure)
            rougeL_scores.append(rg_scores["rougeL"].fmeasure)

        num_count += slice_end

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "rouge1": np.mean(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": np.mean(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": np.mean(rougeL_scores) if rougeL_scores else 0.0,
    }
