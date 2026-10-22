import torch
import torch.nn as nn

from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from models_symmetry import LayerwisePrefixSelfAttention


class PrefixOnlyBertClassifier(nn.Module):
    """Controlled Prefix-only ablation.

    Same layer-wise K/V prefix mechanism used by the true Hybrid model.
    The pretrained BERT backbone is frozen; only layer-wise prefix K/V
    parameters and the classifier are trainable.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        prefix_length: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            model_name,
            attn_implementation="eager",
        )
        self.num_labels = int(num_labels)
        self.prefix_length = int(prefix_length)
        hidden_size = int(self.bert.config.hidden_size)

        for p in self.bert.parameters():
            p.requires_grad = False

        for layer in self.bert.encoder.layer:
            layer.attention.self = LayerwisePrefixSelfAttention(
                layer.attention.self,
                prefix_length=self.prefix_length,
            )

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

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        cls_output = outputs.last_hidden_state[:, 0, :]
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


class PromptOnlyBertClassifier(nn.Module):
    """Controlled Prompt-only ablation.

    Same input-level soft prompt mechanism used by the true Hybrid model.
    The pretrained BERT backbone is frozen; only soft prompts and classifier
    are trainable.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        prompt_length: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            model_name,
            attn_implementation="eager",
        )
        self.num_labels = int(num_labels)
        self.prompt_length = int(prompt_length)
        hidden_size = int(self.bert.config.hidden_size)

        for p in self.bert.parameters():
            p.requires_grad = False

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
            token_type_ids = torch.cat(
                [prompt_token_types, token_type_ids],
                dim=1,
            )

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
