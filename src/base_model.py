import torch
import torch.nn as nn
from transformers import LongT5ForConditionalGeneration
from typing import List, Tuple, Optional
from kg_embedder import KGEncoder

Triple = Tuple[str, str, str]


class LongT5AttentionWrapper(nn.Module):
    """
    Wraps LongT5LayerCrossAttention to provide nn.MultiheadAttention-like interface.
    """

    def __init__(self, longt5_layer_cross_attention):
        super().__init__()
        # Store the entire LongT5LayerCrossAttention module which contains EncDecAttention, layer_norm and dropout
        self.layer = longt5_layer_cross_attention

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):

        attention_mask = None
        if key_padding_mask is not None:
            # Convert key_padding_mask to LongT5 format
            # key_padding_mask: True = padding, False = valid
            # LongT5 attention_mask: large negative = masked, 0 = valid
            # Invert: True (padding) -> large negative value
            attention_mask = key_padding_mask.float() * -1e9
            # Reshape to [batch, 1, 1, key_len] for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Call the layer (it handles attention + layernorm + dropout internally)
        outputs = self.layer(
            hidden_states=query,
            key_value_states=key,  # LongT5 uses same tensor for key/value
            position_bias=None,
            attention_mask=attention_mask,
            layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            query_length=None,
            output_attentions=True,  # We want attention weights
        )

        # outputs is a tuple: (hidden_states, present_key_value_state, attention_weights)
        context = outputs[0]
        attn_weights = outputs[2] if len(outputs) > 2 else None

        return context, attn_weights


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

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_gate_bias: float = -1.0,
        shared_cross_attn: Optional[nn.Module] = None,
        shared_ffn: Optional[nn.Module] = None,
    ):
        super().__init__()

        if shared_cross_attn is not None:
            # Share weights with source cross-attention and layer-norm
            # Will be auto frozen since weights are shared
            self.kg_cross_attention = LongT5AttentionWrapper(shared_cross_attn)
            self.layer_norm = nn.Identity()
            self._using_shared_weights = True
        else:
            # cross attention betwen decoder output and KG embeddings to get KG context
            self.kg_cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            # LayerNorm before fusion for training stability
            self.layer_norm = nn.LayerNorm(hidden_dim)
            self._using_shared_weights = False

        # Fusion gate: maps from decoder hidden state to a scalar in [0, 1]
        self.fusion_gate = nn.Linear(hidden_dim, 1)

        # To preserve pretrained T5 representations early in training
        nn.init.constant_(self.fusion_gate.bias, fusion_gate_bias)

        # Feed-forward network after fusion
        if shared_ffn is not None:
            # Share the FFN weights from the decoder block
            self.ffn = shared_ffn
            self._using_shared_ffn = True
        else:
            # Wont be used but here as fallback
            self.ffn = None
            self._using_shared_ffn = False

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

        self.last_attn_weights = attention_weights

        # kg_context shape: (batch, tgt_len, hidden_dim)
        kg_context = self.layer_norm(kg_context)

        gate = torch.sigmoid(self.fusion_gate(decoder_hidden))
        # gate output shape: (batch, tgt_len, 1)

        # output is a mix of decoder hidden states and kg_context
        fused_output = (1 - gate) * decoder_hidden + gate * kg_context

        if self.ffn is not None:
            # FFN expects: (hidden_states,) and returns (hidden_states,)
            ffn_output = self.ffn(fused_output)

            # LongT5LayerFF returns a tuple: (hidden_states,)
            if isinstance(ffn_output, tuple):
                final_output = ffn_output[0]
            else:
                final_output = ffn_output
        else:
            # No FFN
            final_output = fused_output

        return final_output
        # output shape: (batch, tgt_len, hidden_dim)


class KATSum(nn.Module):
    """
    LongT5 with sidecar KG cross-attention in each decoder layer.

    Args:
        base_model_name: HuggingFace model ID. Default = "google/long-t5-tglobal-base".
        kg_embedder: An instance of KGEncoder (from kg_embedding.py).
        num_sidecar_layers: How many decoder layers get a sidecar.
                            1 = only last layer (lightest)
                            12 = all layers (heaviest, most expressive)
                            Default: last 3 layers.
        freeze_base: If True, freeze all original LongT5 weights.
                     Only the new KG layers train. Default = True.
    """

    def __init__(
        self,
        kg_embedder: KGEncoder,
        base_model,
        num_sidecar_layers: int = 3,
        fusion_gate_biases: list[int] | None = None,
        freeze_base: bool = True,
        device: str | None = None,
    ):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if fusion_gate_biases is not None:
            if len(fusion_gate_biases) != num_sidecar_layers:
                raise ValueError(
                    f"Fusion gate biases length {len(fusion_gate_biases)} whereas number of sidecar layers: {num_sidecar_layers}"
                )

        # Load the base LongT5 model
        self.base_model = base_model
        self.base_model.to(self.device)

        # Freeze base weights
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Store the KG embedder
        self.kg_embedder = kg_embedder

        # Build sidecar layers
        # Get model config to read hidden_dim and num_heads
        config = self.base_model.config
        hidden_dim = config.d_model  # 768 for long-t5-tglobal-base
        num_heads = config.num_heads  # 12 for long-t5-tglobal-base

        num_decoder_layers: int = config.num_decoder_layers  # type: ignore

        print(f"Number of decoder layers for KATSUM model: {num_decoder_layers}")

        sidecar_indices = list(
            range(num_decoder_layers - num_sidecar_layers, num_decoder_layers)
        )

        self.sidecar_indices = sidecar_indices

        # nn.ModuleList registers the layers so PyTorch tracks their parameters
        self.kg_sidecar_layers = nn.ModuleList([])

        # Inside the loop building sidecars:
        for sidecar_position, block_idx in enumerate(sidecar_indices):
            # Extract the source cross-attention from this decoder block
            source_cross_attn = self.base_model.decoder.block[block_idx].layer[1]
            source_ffn_layer = self.base_model.decoder.block[block_idx].layer[2]

            # Adjust path based on LongT5's actual structure
            if fusion_gate_biases is not None:
                sidecar = KGSidecarLayer(
                    hidden_dim=hidden_dim,
                    shared_cross_attn=source_cross_attn,
                    shared_ffn=source_ffn_layer,
                    fusion_gate_bias=fusion_gate_biases[sidecar_position],
                )
            else:
                sidecar = KGSidecarLayer(
                    hidden_dim=hidden_dim,
                    shared_cross_attn=source_cross_attn,
                    shared_ffn=source_ffn_layer,
                )

            self.kg_sidecar_layers.append(sidecar)

        self.kg_sidecar_layers.to(self.device)
        self.num_sidecar_layers = len(self.kg_sidecar_layers)
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
            hook = decoder_blocks[block_idx].register_forward_hook(
                make_hook(block_idx, self.kg_sidecar_layers[sidecar_position])
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
        min_length=40,  # Prevent "one-sentence" lazy summaries
        length_penalty=2.0,  # Higher (>1.0) encourages longer, complete sentences
        no_repeat_ngram_size=3,  # Prevents the "participants... participants" loop
        early_stopping=True,  # Stop as soon as all beams hit the </s> token
        repetition_penalty=2.5,  # Heavily discourages copying the same phrases
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
            min_length=min_length,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            repetition_penalty=repetition_penalty,
        )

        for hook in hooks:
            hook.remove()

        if isinstance(output_ids, torch.Tensor):
            return output_ids

        return output_ids.sequences

    @torch.no_grad()
    def generate_summary_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        triples_batch: List[List[Triple]],
        max_new_tokens: int = 128,
        min_length=40,  # Prevent "one-sentence" lazy summaries
        length_penalty=2.0,  # Higher (>1.0) encourages longer, complete sentences
        no_repeat_ngram_size=3,  # Prevents the "participants... participants" loop
        early_stopping=True,  # Stop as soon as all beams hit the </s> token
        repetition_penalty=2.5,  # Heavily discourages copying the same phrases,
    ) -> torch.Tensor:
        """
        Generate summaries for a batch of examples.

        Args:
            input_ids: (B, src_len)
            attention_mask: (B, src_len)
            triples_batch: List of B triple lists
            max_new_tokens: Max summary length in tokens

        Returns:
            output_ids: (B, summary_len) to decoded with tokenizer.batch_decode()
        """
        kg_embeddings, kg_mask = self._embed_triples_batch(triples_batch)

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

        try:
            output = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                repetition_penalty=repetition_penalty,
            )

        finally:
            # Remove hooks even if generation raises
            for hook in hooks:
                hook.remove()

        return output if isinstance(output, torch.Tensor) else output.sequences

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
