import math
import torch
import torch.nn as nn

from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ContextGatedPrefixSelfAttention(nn.Module):
    """Frozen BERT self-attention with trainable layer-wise prefix K/V and a
    content-dependent scalar gate.

    Gate scale = 2 * sigmoid(g(hidden_states)), so zero-initialized gating
    starts at exactly 1.0 (equivalent prefix strength to the fixed Hybrid)
    and can learn to attenuate or amplify prefix influence per example/layer.
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

        self.prefix_key = nn.Parameter(
            torch.empty(self.prefix_length, self.all_head_size)
        )
        self.prefix_value = nn.Parameter(
            torch.empty(self.prefix_length, self.all_head_size)
        )
        nn.init.normal_(self.prefix_key, mean=0.0, std=0.02)
        nn.init.normal_(self.prefix_value, mean=0.0, std=0.02)

        # One content-conditioned gate per attention layer.
        self.gate = nn.Linear(self.all_head_size, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self.last_gate_scale = None

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

        # Content summary from the current layer input.
        content_summary = hidden_states.mean(dim=1)
        gate_scale = 2.0 * torch.sigmoid(self.gate(content_summary))
        self.last_gate_scale = gate_scale.detach()

        prefix_key = self._prefix_to_heads(self.prefix_key, batch_size)
        prefix_value = self._prefix_to_heads(self.prefix_value, batch_size)

        # Per-example gate: [B,1,1,1]
        g = gate_scale.view(batch_size, 1, 1, 1)
        prefix_key = prefix_key * g
        prefix_value = prefix_value * g

        key_layer = torch.cat([prefix_key, key_layer], dim=2)
        value_layer = torch.cat([prefix_value, value_layer], dim=2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
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


class ContextGatedHybridBertClassifier(nn.Module):
    """Experimental adaptive Prefix+Prompt Hybrid.

    Differences from the fixed Hybrid:
    1) Soft-prompt strength is conditioned on the current input text.
    2) Every layer has a content-dependent prefix gate.
    3) All gates start at scale 1.0, so optimization starts from the same
       effective prompt/prefix strength as the fixed Hybrid.
    4) The pretrained BERT backbone stays frozen.
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

        for p in self.bert.parameters():
            p.requires_grad = False

        for layer in self.bert.encoder.layer:
            layer.attention.self = ContextGatedPrefixSelfAttention(
                layer.attention.self,
                prefix_length=self.prefix_length,
            )

        self.prompt_embeddings = nn.Parameter(
            torch.empty(self.prompt_length, hidden_size)
        )
        nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)

        # Prompt gate uses the mean frozen token embedding of each input,
        # making it content-dependent before the encoder is entered.
        self.prompt_gate = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.prompt_gate.weight)
        nn.init.zeros_(self.prompt_gate.bias)

        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        self.last_prompt_gate_scale = None

    @property
    def config(self):
        return self.bert.config

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.bert.set_input_embeddings(value)

    def prefix_attention_modules(self):
        return [layer.attention.self for layer in self.bert.encoder.layer]

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

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        mask_float = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        denom = mask_float.sum(dim=1).clamp(min=1.0)
        content_summary = (token_embeddings * mask_float).sum(dim=1) / denom

        prompt_gate_scale = 2.0 * torch.sigmoid(self.prompt_gate(content_summary))
        self.last_prompt_gate_scale = prompt_gate_scale.detach()

        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        prompt = prompt * prompt_gate_scale.view(batch_size, 1, 1)
        inputs_embeds = torch.cat([prompt, token_embeddings], dim=1)

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
