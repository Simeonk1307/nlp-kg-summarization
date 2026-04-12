import torch
import torch.nn as nn
from transformers import LongT5ForConditionalGeneration
from typing import List, Tuple, Optional
from kg_embedder import KGEncoder

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

        # Fusion gate: maps from decoder hidden state to a scalar in [0, 1]
        self.fusion_gate = nn.Linear(hidden_dim, 1)
        
        # Init bias to -3.0 so sigmoid output starts at ~0.05
        # To preserve pretrained T5 representations early in training
        nn.init.constant_(self.fusion_gate.bias, -3.0)

        # LayerNorm before fusion for training stability
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
        kg_context = self.layer_norm(kg_context)
        
        gate = torch.sigmoid(self.fusion_gate(decoder_hidden))
        # gate output shape: (batch, tgt_len, 1)

        # output is a mix of decoder hidden states and kg_context
        output = decoder_hidden + gate * kg_context

        # output shape: (batch, tgt_len, hidden_dim)
        return output


class KATSum(nn.Module):
    """
    LongT5 with sidecar KG cross-attention in each decoder layer.

    Args:
        base_model_name: HuggingFace model ID. Default = "google/long-t5-tglobal-base".
        kg_embedder: An instance of KGEncoder (from kg_embedding.py).
        num_sidecar_layers: How many decoder layers get a sidecar.
                            0 = only last layer (lightest)
                            -1 = all layers (heaviest, most expressive)
                            Default: last 3 layers.
        freeze_base: If True, freeze all original LongT5 weights.
                     Only the new KG layers train. Default = True.
    """

    def __init__(
        self,
        kg_embedder: KGEncoder,
        base_model_name: str = "google/long-t5-tglobal-base",
        num_sidecar_layers: int = 3,
        freeze_base: bool = True,
        device: str | None = "cpu",
    ):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load the base LongT5 model
        self.base_model: (
            LongT5ForConditionalGeneration
        ) = LongT5ForConditionalGeneration.from_pretrained(
            base_model_name
        )  # type: ignore
        self.base_model.to(self.device)

        # Freeze base weights
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Store the KG embedder
        self.kg_embedder = kg_embedder

        for param in self.kg_embedder.encoder.parameters():
            param.requires_grad = False

        # Build sidecar layers
        # Get model config to read hidden_dim and num_heads
        config = self.base_model.config
        hidden_dim = config.d_model  # 768 for long-t5-tglobal-base
        num_heads = config.num_heads  # 12 for long-t5-tglobal-base

        num_decoder_layers: int = config.num_decoder_layers  # type: ignore

        print(f"Number of decoder layers for KATSUM model: {num_decoder_layers}")

        # Decide which layers get a sidecar
        if num_sidecar_layers == -1:
            # All layers
            sidecar_indices = list(range(num_decoder_layers))
        else:
            # Last N layers
            sidecar_indices = list(
                range(num_decoder_layers - num_sidecar_layers, num_decoder_layers)
            )

        self.sidecar_indices = sidecar_indices

        # nn.ModuleList registers the layers so PyTorch tracks their parameters
        self.kg_sidecar_layers = nn.ModuleList(
            [
                KGSidecarLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                )
                for _ in sidecar_indices
            ]
        )

        self.kg_sidecar_layers.to(self.device)

        print(f"Sidecar indices: {self.sidecar_indices}")
        print(f"Number of sidecar layers: {len(self.kg_sidecar_layers)}")

    # Forward pass
    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, src_len) : source article tokens
        attention_mask: torch.Tensor,  # (batch, src_len)
        labels: torch.Tensor,  # (batch, tgt_len) : target summary tokens
        triples_batch: List[List[Triple]],  # one list of triples per batch item
    ):
        """
        Args:
            input_ids: Source article token IDs.
            attention_mask:  1 for real tokens, 0 for padding.
            labels: Target summary token IDs.
            triples_batch:   List of Triples per batch

        Returns:
            loss: Scalar cross-entropy over the summary tokens.
            logits: (batch, tgt_len, vocab_size) i.e. token distribution at each step.

        Hooks are used here to intercept decoder hidden states mid-way through
        the forward pass without rewriting the entire LongT5 decoder.
        """

        # Compute KG embeddings for every item in the batch
        kg_embeddings_batch, kg_mask_batch = self._embed_triples_batch(triples_batch)

        # Register forward hooks on decoder blocks
        hooks = []

        def make_hook(layer_idx_in_model, sidecar_layer):
            """
            Returns a hook function for a specific decoder block.
            """

            def hook(module, input, output):
                # hidden_state shape: (batch, tgt_len, hidden_dim)
                hidden_state = output[0]

                # Run the sidecar layer
                updated_hidden = sidecar_layer(
                    decoder_hidden=hidden_state,
                    kg_embeddings=kg_embeddings_batch,
                    kg_padding_mask=kg_mask_batch,
                )

                # Return tuple with updated hidden state
                return (updated_hidden,) + output[1:]

            return hook

        # Register hooks on the decoder blocks that should have sidecars
        decoder_blocks = self.base_model.decoder.block
        for sidecar_position, block_idx in enumerate(self.sidecar_indices):
            hook = (
                decoder_blocks[block_idx]
                .layer[1]
                .register_forward_hook(
                    make_hook(block_idx, self.kg_sidecar_layers[sidecar_position])
                )
            )
            hooks.append(hook)

        # Run the base model forward pase
        # The hooks fire automatically at the right decoder blocks
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs.loss, outputs.logits

    # Inference : generate a summary
    @torch.no_grad()
    def generate_summary(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        triples: List[Triple],
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ) -> torch.Tensor:
        """
        Generate a summary for a single example i.e. batch size = 1

        Args:
            input_ids: (1, src_len) i.e. source article tokens
            attention_mask: (1, src_len)
            triples: List of Triple
            max_new_tokens: Max summary length in tokens
            num_beams:      Beam search width, default = 4

        Returns:
            output_ids: (1, summary_len) tensor which is to be decoded with tokenizer.decode()
        """
        kg_embeddings, kg_mask = self._embed_triples_batch([triples])
        kg_embeddings = kg_embeddings.repeat_interleave(num_beams, dim=0)
        kg_mask = kg_mask.repeat_interleave(num_beams, dim=0)

        # Register hooks (same as in forward())
        def make_gen_hook(sidecar_layer):
            def hook(module, input, output):
                hidden_state = output[0]

                updated = sidecar_layer(
                    decoder_hidden=hidden_state,
                    kg_embeddings=kg_embeddings,
                    kg_padding_mask=kg_mask,
                )

                return (updated,) + output[1:]

            return hook

        hooks = []
        decoder_blocks = self.base_model.decoder.block
        for sidecar_position, block_idx in enumerate(self.sidecar_indices):
            hook = decoder_blocks[block_idx].register_forward_hook(
                make_gen_hook(self.kg_sidecar_layers[sidecar_position])
            )
            hooks.append(hook)

        output_ids = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        for hook in hooks:
            hook.remove()

        if isinstance(output_ids, torch.Tensor):
            return output_ids

        return output_ids.sequences

    # Helper Methods
    def _embed_triples_batch(
        self, triples_batch: List[List[Triple]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a batch of triple lists into a padded tensor.

        This is needed as different documents have different numbers of triples.
        To batch them, pad shorter documents with zero vectors to match the maximum.

        Returns:
            embeddings: (batch, max_triples, hidden_dim)

            mask: (batch, max_triples), True where padded so that it is ignored by attention
        """

        batch_embeddings = []
        for triples in triples_batch:
            # kg embber embed returns (1, num_triples, hidden_dim)
            # squeeze the batch dim to (num_triples, hidden_dim)
            emb = self.kg_embedder(triples).squeeze(0)
            batch_embeddings.append(emb)

        # Find maximum number of triples in this batch
        max_triples = max(e.shape[0] for e in batch_embeddings)
        hidden_dim = batch_embeddings[0].shape[-1]

        padded = []
        mask = []
        for emb in batch_embeddings:
            n = emb.shape[0]
            pad_size = max_triples - n

            if pad_size > 0:
                # Pad with zeros
                padding = torch.zeros(pad_size, hidden_dim, device=self.device)
                emb_padded = torch.cat([emb, padding], dim=0)

                # Mask: False for real triples, True for padding
                m = torch.cat(
                    [
                        torch.zeros(n, dtype=torch.bool, device=self.device),
                        torch.ones(pad_size, dtype=torch.bool, device=self.device),
                    ]
                )

            else:
                emb_padded = emb
                m = torch.zeros(n, dtype=torch.bool, device=self.device)

            padded.append(emb_padded)
            mask.append(m)

        # stack them back to get a tensor of appropriate shape
        embeddings = torch.stack(padded, dim=0)  # (batch, max_triples, hidden_dim)
        masks = torch.stack(mask, dim=0)  # (batch, max_triples)

        return embeddings, masks

    def trainable_parameters(self):
        """Return the parameters that should be trained."""
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_count(self):
        """Print count of parameters that are trainable or frozen."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        sidecar_params = sum(p.numel() for p in self.kg_sidecar_layers.parameters())

        kg_embedder_params = sum(p.numel() for p in self.kg_embedder.parameters())
        t5_encoder_params = sum(
            p.numel() for p in self.kg_embedder.encoder.parameters()
        )

        # since kg embedder has the t5 encoder params as well
        kg_embedder_params -= t5_encoder_params

        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.1f}%)")
        print(f"Frozen parameters: {frozen:,} ({100*frozen/total:.1f}%)")
        print(f"KG Embedder params: {kg_embedder_params:,}")
        print(f"Sidecar layer params: {sidecar_params:,}")
        print(f"Num sidecar layers: {len(self.kg_sidecar_layers)}")


if __name__ == "__main__":
    pass
