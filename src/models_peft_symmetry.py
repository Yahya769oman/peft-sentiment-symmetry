import torch
import torch.nn as nn

from transformers import BertModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType


class ResidualBottleneckAdapter(nn.Module):
    """Serial bottleneck adapter: residual + down projection + GELU + up projection."""

    def __init__(self, hidden_size: int, bottleneck_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Start near the frozen backbone behavior.
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states):
        residual = hidden_states
        x = self.norm(hidden_states)
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        x = self.dropout(x)
        return residual + x


class AdapterWrappedBertLayer(nn.Module):
    """Wrap one frozen BERT layer and apply a trainable serial adapter to its output."""

    def __init__(self, base_layer: nn.Module, hidden_size: int, bottleneck_size: int, dropout: float):
        super().__init__()
        self.base_layer = base_layer
        self.adapter = ResidualBottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            dropout=dropout,
        )

    def forward(self, *args, **kwargs):
        outputs = self.base_layer(*args, **kwargs)

        if isinstance(outputs, tuple):
            hidden = self.adapter(outputs[0])
            return (hidden,) + outputs[1:]

        # Defensive fallback; BertLayer normally returns a tuple.
        return self.adapter(outputs)


class LayerwiseAdapterBertClassifier(nn.Module):
    """Frozen BERT + one trainable residual bottleneck adapter after every encoder layer."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        bottleneck_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.num_labels = int(num_labels)
        hidden_size = int(self.bert.config.hidden_size)

        for p in self.bert.parameters():
            p.requires_grad = False

        for i, layer in enumerate(self.bert.encoder.layer):
            self.bert.encoder.layer[i] = AdapterWrappedBertLayer(
                base_layer=layer,
                hidden_size=hidden_size,
                bottleneck_size=int(bottleneck_size),
                dropout=float(dropout),
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
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def build_lora_bert(
    model_name: str,
    num_labels: int,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=int(num_labels),
    )

    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=["query", "value"],
        bias="none",
    )
    return get_peft_model(model, cfg)


def build_bitfit_bert(model_name: str, num_labels: int):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=int(num_labels),
    )

    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "bias" in name:
            p.requires_grad = True

    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    return model
