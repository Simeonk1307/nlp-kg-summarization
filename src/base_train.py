import torch
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from kg_extractor import KGExtractor, Triple
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
