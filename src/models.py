import inspect
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


def load_base_model(model_name: str, num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=int(num_labels),
    )


# ============================================================
# 1) BERT-SPECIFIC HYBRID MODEL
# يستخدم في EXP1 إلى EXP7 مع bert-base-uncased
# ============================================================

class HybridPrefixPromptClassifierModel(nn.Module):
    """
    Hybrid Soft Prefix + Soft Prompt model for BERT sequence classification.

    الفكرة:
    - نجمّد BERT بالكامل.
    - نفتح آخر طبقة أو طبقتين اختياريًا.
    - ندرّب:
        1) prefix embeddings
        2) prompt embeddings
        3) classifier
        4) آخر طبقات من BERT إذا حددتها في config
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        prefix_length: int,
        prompt_length: int,
        unfreeze_last_n_layers: int = 0,
    ):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=int(num_labels),
        )

        if not hasattr(self.model, "bert"):
            raise ValueError(
                "Only BERT-based models are supported by HybridPrefixPromptClassifierModel. "
                "For RoBERTa/DistilBERT use GenericInputPrefixPromptClassifierModel."
            )

        self.bert = self.model.bert
        self.dropout = self.model.dropout
        self.classifier = self.model.classifier

        self.num_labels = int(num_labels)
        self.prefix_length = int(prefix_length)
        self.prompt_length = int(prompt_length)
        self.virtual_length = self.prefix_length + self.prompt_length

        hidden_size = self.model.config.hidden_size

        # =========================
        # Freeze BERT backbone
        # =========================
        for param in self.bert.parameters():
            param.requires_grad = False

        # =========================
        # Optional unfreeze last N layers
        # =========================
        unfreeze_last_n_layers = int(unfreeze_last_n_layers)

        if unfreeze_last_n_layers > 0:
            for layer in self.bert.encoder.layer[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

            if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True

        # =========================
        # Train classifier
        # =========================
        for param in self.classifier.parameters():
            param.requires_grad = True

        # =========================
        # Trainable soft prefix + prompt
        # =========================
        self.prefix_embeddings = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=0.02,
                size=(self.prefix_length, hidden_size),
            )
        )

        self.prompt_embeddings = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=0.02,
                size=(self.prompt_length, hidden_size),
            )
        )

    @property
    def config(self):
        return self.model.config

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.bert.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def _trim_inputs_if_needed(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        BERT max_position_embeddings غالبًا 512.
        إذا أضفنا prefix + prompt، لازم نقص من النص الأصلي حتى لا نتجاوز الحد.
        """
        max_positions = int(self.model.config.max_position_embeddings)
        max_real_tokens = max_positions - self.virtual_length

        if max_real_tokens <= 0:
            raise ValueError(
                "prefix_length + prompt_length is too large for BERT max_position_embeddings."
            )

        if input_ids.size(1) > max_real_tokens:
            input_ids = input_ids[:, :max_real_tokens]

            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_real_tokens]

            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :max_real_tokens]

        return input_ids, attention_mask, token_type_ids

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
            raise ValueError("input_ids is required.")

        input_ids, attention_mask, token_type_ids = self._trim_inputs_if_needed(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        batch_size = input_ids.size(0)
        device = input_ids.device

        # =========================
        # Token embeddings
        # =========================
        token_embeddings = self.bert.embeddings.word_embeddings(input_ids)

        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        inputs_embeds = torch.cat(
            [prefix, prompt, token_embeddings],
            dim=1,
        )

        total_length = inputs_embeds.size(1)

        # =========================
        # Position embeddings
        # =========================
        position_ids = torch.arange(
            total_length,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.bert.embeddings.position_embeddings(position_ids)

        if token_type_ids is not None:
            virtual_token_type_ids = torch.zeros(
                batch_size,
                self.virtual_length,
                dtype=token_type_ids.dtype,
                device=device,
            )

            token_type_ids = torch.cat(
                [virtual_token_type_ids, token_type_ids],
                dim=1,
            )

            token_type_embeddings = self.bert.embeddings.token_type_embeddings(
                token_type_ids
            )
        else:
            token_type_embeddings = torch.zeros_like(inputs_embeds)

        inputs_embeds = inputs_embeds + position_embeddings + token_type_embeddings
        inputs_embeds = self.bert.embeddings.LayerNorm(inputs_embeds)
        inputs_embeds = self.bert.embeddings.dropout(inputs_embeds)

        # =========================
        # Attention mask
        # =========================
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        virtual_attention = torch.ones(
            batch_size,
            self.virtual_length,
            dtype=attention_mask.dtype,
            device=device,
        )

        attention_mask = torch.cat(
            [virtual_attention, attention_mask],
            dim=1,
        )

        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask,
            attention_mask.shape,
            device,
        )

        # =========================
        # BERT encoder forward
        # =========================
        outputs = self.bert.encoder(
            inputs_embeds,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        # CLS الحقيقي بعد prefix + prompt
        cls_index = self.virtual_length
        cls_output = sequence_output[:, cls_index, :]

        pooled_output = self.dropout(cls_output)
        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ============================================================
# 2) GENERIC INPUT-LEVEL HYBRID MODEL
# يستخدم في EXP8 مع BERT / RoBERTa / DistilBERT
# ويمكن استخدامه في EXP1 لتجاوز مشكلة input_ids / inputs_embeds
# ============================================================

class GenericInputPrefixPromptClassifierModel(nn.Module):
    """
    Generic input-level Prefix + Prompt model.

    يدعم:
    - bert-base-uncased
    - distilbert-base-uncased
    - roberta-base

    التعديل المهم:
    - يستخدم AutoModel بدل AutoModelForSequenceClassification.
    - يضيف prefix + prompt قبل النص.
    - يأخذ تمثيل CLS الحقيقي بعد virtual tokens.
    - يدعم unfreeze_last_n_layers فعليًا.
    - يدرّب:
        1) prefix embeddings
        2) prompt embeddings
        3) classifier
        4) آخر N layers إذا طلبت ذلك
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        prefix_length: int,
        prompt_length: int,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        self.num_labels = int(num_labels)

        self.prefix_length = int(prefix_length)
        self.prompt_length = int(prompt_length)
        self.virtual_length = self.prefix_length + self.prompt_length

        self.unfreeze_last_n_layers = int(unfreeze_last_n_layers)

        hidden_size = int(self.backbone.config.hidden_size)

        # =========================
        # Trainable soft prefix
        # =========================
        self.prefix_embeddings = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=0.02,
                size=(self.prefix_length, hidden_size),
            )
        )

        # =========================
        # Trainable soft prompt
        # =========================
        self.prompt_embeddings = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=0.02,
                size=(self.prompt_length, hidden_size),
            )
        )

        # =========================
        # Freeze backbone
        # =========================
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # =========================
        # Optional unfreeze last N layers
        # =========================
        if self.unfreeze_last_n_layers > 0:
            self._unfreeze_last_transformer_layers(self.unfreeze_last_n_layers)

        # =========================
        # Classification head
        # =========================
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        for param in self.classifier.parameters():
            param.requires_grad = True

        self.forward_parameters = inspect.signature(self.backbone.forward).parameters

    @property
    def config(self):
        return self.backbone.config

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    def _unfreeze_last_transformer_layers(self, n_layers: int):
        """
        يفتح آخر N layers حسب نوع الموديل:
        - BERT/RoBERTa: encoder.layer
        - DistilBERT: transformer.layer
        """
        n_layers = int(n_layers)

        # BERT / RoBERTa style
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            layers = self.backbone.encoder.layer

            for layer in layers[-n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # DistilBERT style
        elif hasattr(self.backbone, "transformer") and hasattr(
            self.backbone.transformer,
            "layer",
        ):
            layers = self.backbone.transformer.layer

            for layer in layers[-n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        else:
            print(
                "Warning: Could not find transformer layers to unfreeze. "
                "Only prefix, prompt, and classifier will be trainable."
            )

        # Pooler في BERT إذا موجود
        if hasattr(self.backbone, "pooler") and self.backbone.pooler is not None:
            for param in self.backbone.pooler.parameters():
                param.requires_grad = True

    def _trim_inputs_if_needed(self, input_ids, attention_mask=None, token_type_ids=None):
        max_positions = getattr(self.backbone.config, "max_position_embeddings", None)

        if max_positions is None:
            return input_ids, attention_mask, token_type_ids

        max_real_tokens = int(max_positions) - self.virtual_length

        if max_real_tokens <= 0:
            raise ValueError(
                "prefix_length + prompt_length is too large for model max_position_embeddings."
            )

        if input_ids.size(1) > max_real_tokens:
            input_ids = input_ids[:, :max_real_tokens]

            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_real_tokens]

            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :max_real_tokens]

        return input_ids, attention_mask, token_type_ids

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
            raise ValueError("input_ids is required.")

        input_ids, attention_mask, token_type_ids = self._trim_inputs_if_needed(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        batch_size = input_ids.size(0)
        device = input_ids.device

        # =========================
        # Build embeddings
        # =========================
        token_embeddings = self.backbone.get_input_embeddings()(input_ids)

        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        inputs_embeds = torch.cat(
            [prefix, prompt, token_embeddings],
            dim=1,
        )

        # =========================
        # Attention mask
        # =========================
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        virtual_attention = torch.ones(
            batch_size,
            self.virtual_length,
            dtype=attention_mask.dtype,
            device=device,
        )

        attention_mask = torch.cat(
            [virtual_attention, attention_mask],
            dim=1,
        )

        backbone_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }

        # BERT يقبل token_type_ids، RoBERTa وDistilBERT غالبًا لا
        if token_type_ids is not None and "token_type_ids" in self.forward_parameters:
            virtual_token_type_ids = torch.zeros(
                batch_size,
                self.virtual_length,
                dtype=token_type_ids.dtype,
                device=device,
            )

            token_type_ids = torch.cat(
                [virtual_token_type_ids, token_type_ids],
                dim=1,
            )

            backbone_kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**backbone_kwargs)

        sequence_output = outputs.last_hidden_state

        # =========================
        # مهم جدًا:
        # نأخذ CLS الحقيقي بعد prefix + prompt
        # وليس أول virtual token
        # =========================
        cls_index = self.virtual_length
        cls_output = sequence_output[:, cls_index, :]

        pooled_output = self.dropout(cls_output)
        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ============================================================
# 3) ADAPTER-BERT MODEL
# يستخدم في EXP5
# ============================================================

class AdapterLayer(nn.Module):
    """
    Simple bottleneck adapter:
    hidden_size -> bottleneck -> hidden_size
    مع residual connection.
    """

    def __init__(self, hidden_size: int, bottleneck_size: int = 64, dropout: float = 0.1):
        super().__init__()

        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, hidden_states):
        residual = hidden_states

        x = self.down(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up(x)

        return residual + x


class AdapterBertClassifierModel(nn.Module):
    """
    Adapter-BERT بسيط للتجربة الخامسة.

    الفكرة:
    - نستخدم AutoModel بدل AutoModelForSequenceClassification.
    - نجمّد backbone.
    - نضيف adapter بعد آخر hidden states.
    - ندرّب adapter + classifier فقط.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        bottleneck_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        self.num_labels = int(num_labels)

        hidden_size = int(self.backbone.config.hidden_size)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.adapter = AdapterLayer(
            hidden_size=hidden_size,
            bottleneck_size=int(bottleneck_size),
            dropout=float(dropout),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

    @property
    def config(self):
        return self.backbone.config

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

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
        backbone_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }

        # RoBERTa لا يقبل token_type_ids عادة
        backbone_forward_params = inspect.signature(self.backbone.forward).parameters

        if token_type_ids is not None and "token_type_ids" in backbone_forward_params:
            backbone_kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**backbone_kwargs)

        sequence_output = outputs.last_hidden_state
        adapted_output = self.adapter(sequence_output)

        cls_output = adapted_output[:, 0, :]
        pooled_output = self.dropout(cls_output)
        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )