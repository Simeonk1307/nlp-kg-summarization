import torch
import torch.nn as nn
from typing import List, Tuple

Triple = Tuple[str, str, str]


class KGEncoder(nn.Module):
    """
    Encodes KG triples by converting them to text and running them through
    the T5 encoder. Returns one vector per triple.

    Args:
        encoder: T5 encoder
        tokenizer:  T5 tokenizer
        hidden_dim: Output dimension (should match LongT5 hidden dim = 768)
        device:     "cuda" or "cpu"
    """

    def __init__(
        self,
        encoder,
        tokenizer,
        hidden_dim: int = 768,
        max_triple_len: int = 64,
        device: str = "cpu",
    ):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.max_triple_len = max_triple_len
        self.device = device

    def triples_to_text(self, triples: List[Triple]) -> List[str]:
        """
        Converting triples to natural language strings.
        """
        texts = []
        for head, relation, tail in triples:
            text = f"{head} {relation} {tail}"
            texts.append(text)
        return texts

    def forward(self, triples: List[Triple]) -> torch.Tensor:
        """
        Encode a list of triples into a tensor.

        Args:
            triples: List of (head, relation, tail) tuples.

        Returns:
            Tensor of shape (1, num_triples, hidden_dim).
            This matches what the cross-attention layer expects.

        Steps for reference:
            1. Convert triples to text strings
            2. Tokenize all strings together
            3. Run through the encoder to get hidden states for each token
            4. Mean pool each triple's tokens to get one vector per triple
            5. Stack into (num_triples, hidden_dim)
            6. Project and normalise
            7. Add batch dimension i.e. unsqueeze to get (1, num_triples, hidden_dim)
        """

        # Return a zero tensor if no triples sp that decoder will attend to nothing
        if not triples:
            return torch.zeros(1, 1, self.hidden_dim, device=self.device)

        # Step 1: text strings
        texts = self.triples_to_text(triples)

        # Step 2: tokenize
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_triple_len,
        ).to(self.device)

        # Step 3: run through frozen encoder
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                return_dict=True,
            )

        # hidden_states shape: (num_triples, seq_len, 768)
        hidden_states = encoder_outputs.last_hidden_state

        # Step 4: mean-pool: average over the token dimension (dim=1)
        # We must ignore padding tokens when averaging, so multiply by mask first

        # mask shape: (num_triples, seq_len, 1)
        mask = encoding["attention_mask"].unsqueeze(-1).float()

        # summed shape: (num_triples, 768)
        summed = (hidden_states * mask).sum(dim=1)

        # counts shape: (num_triples, 1), clamp prevents div by zero
        counts = mask.sum(dim=1).clamp(min=1)

        # pooled shape: (num_triples, 768)
        pooled = summed / counts

        # Step 7: add batch dimension
        # final shape: (1, num_triples, 768)
        return pooled.unsqueeze(0)


if __name__ == "__main__":
    pass