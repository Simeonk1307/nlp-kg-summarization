import torch
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from kg_extractor import KGExtractor, Triple
from base_model import KATSum
from rouge_score.rouge_scorer import RougeScorer
from summac.model_summac import SummaCZS
import bert_score
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
        tokenizer,
        extractor: KGExtractor,
        src_max_len: int = 4096,
        tgt_max_len: int = 256,
        cache_triples: bool = True,
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

        # Either Extract or cache triples
        if cache_triples:
            print(
                f"Extracting KG triples from {len(articles)} articles (this may take a while)..."
            )
            self.triples_cache = extractor.extract_batch(articles)
            print(
                f"Extraction done. "
                f"Average triples per article: {np.mean([len(t) for t in self.triples_cache]):.1f}"
            )
        else:
            self.extractor = extractor
            self.triples_cache = None

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Return one example as a dictionary.
        All values are Python lists and not tensors yet.
        collate_fn will convert them to tensors.
        """
        if self.triples_cache is not None:
            triples = self.triples_cache[idx]
        else:
            # Lazy extraction , decode token IDs back to text for extraction
            text = self.tokenizer.decode(self.input_ids[idx], skip_special_tokens=True)
            triples = self.extractor.extract(text)

        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
            "triples": triples,
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

        if step % 50 == 0:
            print(
                f"  Step {step}/{len(dataloader)}  loss={loss.item()*grad_accumulation_steps:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: KATSum,
    dataloader: DataLoader,
    tokenizer,
    device: str,
    num_rouge_examples: int = 100,
) -> Dict:
    """
    Evaluate on validation set i.e. compute loss and ROUGE scores.

    Args:
        num_rouge_examples: Generating summaries is slow. We compute ROUGE
                            on a subset of validation examples.
    Returns:
        Dict with keys: "val_loss", "rouge1", "rouge2", "rougeL"
    """

    rogue_scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    summac_scorer = SummaCZS(granularity="sentence", model_name="vitc", device="cpu")
    model.eval()
    total_loss = 0.0
    num_batches = 0

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bert_scores = []
    summac_scores = []
    rouge_count = 0

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

        # Generate and score a few examples for ROUGE
        if rouge_count < num_rouge_examples:
            for i in range(input_ids.shape[0]):
                if rouge_count >= num_rouge_examples:
                    break

                # Generate summary for example i
                generated_ids = model.generate_summary(
                    input_ids=input_ids[i : i + 1],
                    attention_mask=attention_mask[i : i + 1],
                    triples=triples_batch[i],
                    max_new_tokens=512,
                    num_beams=4,
                )

                source_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                source_text = source_text.removeprefix("summarize: ")

                # Decode to strings
                generated_text = tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )

                # Decode reference (replace -100 with pad_token_id before decoding)
                ref_ids = labels[i].clone()
                ref_ids[ref_ids == -100] = tokenizer.pad_token_id
                reference_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

                # Score
                rg_scores = rogue_scorer.score(reference_text, generated_text)
                _, _, f1_bert = bert_score.score(
                    [generated_text], [reference_text], lang="en"
                )
                summac_score = summac_scorer.score(source_text, generated_text)
                rouge1_scores.append(rg_scores["rouge1"].fmeasure)
                rouge2_scores.append(rg_scores["rouge2"].fmeasure)
                rougeL_scores.append(rg_scores["rougeL"].fmeasure)
                bert_scores.append(f1_bert.mean().item())
                summac_scores.append(summac_score['scores'][0])
                rouge_count += 1
                
                print(f"Source: {source_text[:200]}")
                print(f"Generated: {generated_text[:200]}")
                print(f"Reference: {reference_text[:200]}")

    results = {
        "val_loss": total_loss / max(num_batches, 1),
        "rouge1": np.mean(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": np.mean(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": np.mean(rougeL_scores) if rougeL_scores else 0.0,
        "rougeL": np.mean(rougeL_scores) if rougeL_scores else 0.0,
        "bert": np.mean(bert_scores) if bert_scores else 0.0,
        "summaC": np.mean(summac_scores) if summac_scores else 0.0,
    }

    return results
