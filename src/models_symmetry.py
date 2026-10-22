import math
import torch
import torch.nn as nn

from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class LayerwisePrefixSelfAttention(nn.Module):
    """BERT self-attention with trainable layer-wise prefix keys/values.

    The pretrained Q/K/V projections are reused and remain frozen when the
    backbone is frozen. Prefix vectors are inserted only into the key/value
    streams; they do not become query tokens.
    """

    def __init__(self, base_attention: nn.Module, prefix_length: int):
        super().__init__()

        self.query = base_attention.query
        self.key = base_attention.key
        self.value = base_attention.value
        self.dropout = base_attention.dropout

        self.num_attention_heads = int(base_attention.num_attention_heads)
        self.attention_head_size = int(base_attention.attention_head_size)
        self.all_head_size = int(base_attention.all_head_size)
        self.prefix_length = int(prefix_length)

        # Directly learn per-layer K/V prefix states in attention space.
        self.prefix_key = nn.Parameter(
            torch.empty(self.prefix_length, self.all_head_size)
        )
        self.prefix_value = nn.Parameter(
            torch.empty(self.prefix_length, self.all_head_size)
        )
        nn.init.normal_(self.prefix_key, mean=0.0, std=0.02)
        nn.init.normal_(self.prefix_value, mean=0.0, std=0.02)

    def _shape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return x.view(
            batch_size,
            -1,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)

    def _prefix_to_heads(self, p: torch.Tensor, batch_size: int) -> torch.Tensor:
        p = p.unsqueeze(0).expand(batch_size, -1, -1)
        return self._shape(p, batch_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        cache_position=None,
        **kwargs,
    ):
        if encoder_hidden_states is not None:
            raise ValueError("This experiment uses encoder-only BERT self-attention.")
        if past_key_values is not None:
            raise ValueError("Caching is not used for BERT sentiment classification.")

        batch_size = hidden_states.size(0)

        query_layer = self._shape(self.query(hidden_states), batch_size)
        key_layer = self._shape(self.key(hidden_states), batch_size)
        value_layer = self._shape(self.value(hidden_states), batch_size)

        prefix_key = self._prefix_to_heads(self.prefix_key, batch_size)
        prefix_value = self._prefix_to_heads(self.prefix_value, batch_size)

        key_layer = torch.cat([prefix_key, key_layer], dim=2)
        value_layer = torch.cat([prefix_value, value_layer], dim=2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # BertModel creates an additive mask for the real sequence. Extend
            # it with unmasked positions for the layer-wise prefix K/V states.
            if attention_mask.size(-1) + self.prefix_length == key_layer.size(-2):
                prefix_mask = torch.zeros(
                    *attention_mask.shape[:-1],
                    self.prefix_length,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
            elif attention_mask.size(-1) != key_layer.size(-2):
                raise ValueError(
                    f"Attention mask/key length mismatch: mask={attention_mask.size(-1)}, "
                    f"keys={key_layer.size(-2)}"
                )
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            context_layer.size(0), context_layer.size(1), self.all_head_size
        )

        return context_layer, attention_probs


class TrueHybridPrefixPromptBertClassifier(nn.Module):
    """True hybrid PEFT for BERT sentiment classification.

    - Soft prompt vectors are prepended at the input embedding level.
    - Prefix K/V vectors are injected into every BERT self-attention layer.
    - The pretrained BERT backbone stays frozen.
    - Only prompts, layer-wise prefix K/V parameters, and classifier train.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        prompt_length: int = 20,
        prefix_length: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            model_name,
            attn_implementation="eager",
        )
        self.num_labels = int(num_labels)
        self.prompt_length = int(prompt_length)
        self.prefix_length = int(prefix_length)
        hidden_size = int(self.bert.config.hidden_size)

        # Freeze the pretrained backbone first.
        for param in self.bert.parameters():
            param.requires_grad = False

        # Replace each BERT self-attention block with an equivalent block that
        # adds trainable layer-wise prefix K/V states while reusing frozen Q/K/V.
        for layer in self.bert.encoder.layer:
            layer.attention.self = LayerwisePrefixSelfAttention(
                layer.attention.self,
                prefix_length=self.prefix_length,
            )

        # Input-level soft prompt.
        self.prompt_embeddings = nn.Parameter(
            torch.empty(self.prompt_length, hidden_size)
        )
        nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_size, self.num_labels)

    @property
    def config(self):
        return self.bert.config

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.bert.set_input_embeddings(value)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required")

        batch_size, seq_len = input_ids.shape
        max_positions = int(self.bert.config.max_position_embeddings)
        max_real_tokens = max_positions - self.prompt_length

        if seq_len > max_real_tokens:
            input_ids = input_ids[:, :max_real_tokens]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_real_tokens]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :max_real_tokens]

        token_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prompt, token_embeddings], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prompt_mask = torch.ones(
            batch_size,
            self.prompt_length,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        if token_type_ids is not None:
            prompt_token_types = torch.zeros(
                batch_size,
                self.prompt_length,
                dtype=token_type_ids.dtype,
                device=token_type_ids.device,
            )
            token_type_ids = torch.cat([prompt_token_types, token_type_ids], dim=1)

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Original [CLS] token occurs immediately after the soft prompts.
        cls_output = outputs.last_hidden_state[:, self.prompt_length, :]
        logits = self.classifier(self.dropout(cls_output))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
