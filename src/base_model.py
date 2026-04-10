import torch
import torch.nn as nn
from typing import List, Tuple, Optional

Triple = Tuple[str, str, str]


class KGSidecarLayer(nn.Module):
    """
    A single KG cross-attention sublayer which is inserted after the text cross-attention in each decoder block.

    It takes:
        - decoder_hidden: the decoder's current representation (after text cross-attention)
        - kg_embeddings:  the KG triple vectors

    And outputs: the decoder representation blended with KG context

    Args:
        hidden_dim: Model hidden size. Default = 768 for LongT5-base
        num_heads: Number of attention heads. Default = 8 for LongT5-base
        dropout: Dropout on attention weights during training.
    """

    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # cross attention betwen decoder output and KG embeddings to get KG context
        self.kg_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Dropout on the KG context before fusion
        self.dropout = nn.Dropout(dropout)

        # Fusion gate: maps from decoder hidden state to a scalar in [0, 1]
        self.fusion_gate = nn.Linear(hidden_dim, 1)

        # LayerNorm after fusion for training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        decoder_hidden: torch.Tensor,  # shape = (batch, tgt_len, hidden_dim)
        kg_embeddings: torch.Tensor,  # shape = (batch, num_triples, hidden_dim)
        kg_padding_mask: Optional[
            torch.Tensor
        ] = None,  # shape = (batch, num_triples) ,  True where padded present
    ) -> torch.Tensor:

        kg_context, attention_weights = self.kg_cross_attention(
            query=decoder_hidden,
            key=kg_embeddings,
            value=kg_embeddings,
            key_padding_mask=kg_padding_mask,
        )
        # kg_context shape: (batch, tgt_len, hidden_dim)
        
        kg_context = self.dropout(kg_context)

        gate = torch.sigmoid(self.fusion_gate(decoder_hidden))
        # gate output shape: (batch, tgt_len, 1)

        blended = (1 - gate) * decoder_hidden + gate * kg_context
        # blended shape: (batch, tgt_len, hidden_dim)

        # residual + layernorm
        output = self.layer_norm(blended + decoder_hidden)

        # output shape: (batch, tgt_len, hidden_dim)
        return output
