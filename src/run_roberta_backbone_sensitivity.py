import os
import sys
import csv
import json
import time
import math
import random
import platform
import subprocess

import numpy as np
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import RobertaModel, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters

EXPERIMENT = "roberta_backbone_sensitivity"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)

DATASETS = ["sst2", "imdb"]
METHODS = ["full_ft", "hybrid"]
SEEDS = [42, 123, 2026]
DATA_SEED = 2026

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
TRAIN_SIZE = 3000
EVAL_SIZE = 500
EPOCHS = 3

# Safer for a 4 GB Quadro P520 while preserving effective batch size 8.
TRAIN_BATCH = 4
EVAL_BATCH = 8
GRAD_ACCUM = 2

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

# Diagnostic backbone sensitivity:
# transfer the learning-rate policy used in the BERT study rather than retuning
# on the held-out evaluation split.
LEARNING_RATES = {
    "sst2": {"full_ft": 2e-5, "hybrid": 3e-4},
    "imdb": {"full_ft": 2e-5, "hybrid": 5e-4},
}

PREFIX_LENGTH = 16
PROMPT_LENGTH = 20
DROPOUT = 0.1

RESULTS_JSON = os.path.join(OUTPUT_DIR, "roberta_backbone_results.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "roberta_backbone_results.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "roberta_backbone_summary.json")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "roberta_backbone_summary.csv")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fixed_subset(split, n, seed):
    return split.shuffle(seed=int(seed)).select(range(min(int(n), len(split))))


class LayerwisePrefixRobertaSelfAttention(nn.Module):
    """RoBERTa eager self-attention with trainable layer-wise prefix K/V states."""

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

    def _shape(self, x, batch_size):
        return x.view(
            batch_size, -1, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)

    def _prefix_to_heads(self, p, batch_size):
        p = p.unsqueeze(0).expand(batch_size, -1, -1)
        return self._shape(p, batch_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        past_key_values=None,
        output_attentions=False,
        cache_position=None,
        **kwargs,
    ):
        if encoder_hidden_states is not None:
            raise ValueError("Cross-attention is not used in this encoder-only experiment.")

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
                    f"Attention mask/key mismatch: "
                    f"mask={attention_mask.size(-1)}, keys={key_layer.size(-2)}"
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


class RobertaClassificationHeadMatched(nn.Module):
    """Head matched across Full FT and Hybrid for an apples-to-apples backbone check."""

    def __init__(self, config):
        super().__init__()
        hidden = int(config.hidden_size)
        p = (
            config.classifier_dropout
            if getattr(config, "classifier_dropout", None) is not None
            else float(config.hidden_dropout_prob)
        )
        self.dropout = nn.Dropout(p)
        self.dense = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, 2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class FullRobertaSentimentClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            attn_implementation="eager",
        )
        self.num_labels = int(num_labels)
        self.classifier = RobertaClassificationHeadMatched(self.roberta.config)
        self.classifier.out_proj = nn.Linear(
            int(self.roberta.config.hidden_size), self.num_labels
        )

    @property
    def config(self):
        return self.roberta.config

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)

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


class HybridPrefixPromptRobertaClassifier(nn.Module):
    """Frozen RoBERTa + input soft prompts + per-layer prefix K/V + matched head."""

    def __init__(
        self,
        model_name,
        num_labels,
        prefix_length=16,
        prompt_length=20,
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            attn_implementation="eager",
        )
        self.num_labels = int(num_labels)
        self.prefix_length = int(prefix_length)
        self.prompt_length = int(prompt_length)
        hidden = int(self.roberta.config.hidden_size)

        for p in self.roberta.parameters():
            p.requires_grad = False

        for layer in self.roberta.encoder.layer:
            layer.attention.self = LayerwisePrefixRobertaSelfAttention(
                layer.attention.self,
                prefix_length=self.prefix_length,
            )

        self.prompt_embeddings = nn.Parameter(
            torch.empty(self.prompt_length, hidden)
        )
        nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)

        self.classifier = RobertaClassificationHeadMatched(self.roberta.config)
        self.classifier.out_proj = nn.Linear(hidden, self.num_labels)

    @property
    def config(self):
        return self.roberta.config

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required")

        batch_size, seq_len = input_ids.shape

        # RoBERTa has max_position_embeddings=514 with padding_idx=1,
        # corresponding to 512 usable sequential positions. Prompts consume
        # part of that position budget.
        padding_idx = int(self.roberta.embeddings.padding_idx)
        usable_positions = int(self.roberta.config.max_position_embeddings) - padding_idx - 1
        max_real_tokens = usable_positions - self.prompt_length

        if seq_len > max_real_tokens:
            input_ids = input_ids[:, :max_real_tokens]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_real_tokens]

        token_embeddings = self.roberta.embeddings.word_embeddings(input_ids)
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
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Original <s> token follows the learned soft prompts.
        cls = outputs.last_hidden_state[:, self.prompt_length, :]
        logits = self.classifier(cls)

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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    p, r, f1c, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_class0": p[0],
        "recall_class0": r[0],
        "f1_class0": f1c[0],
        "support_class0": support[0],
        "precision_class1": p[1],
        "recall_class1": r[1],
        "f1_class1": f1c[1],
        "support_class1": support[1],
    }


def build_model(method, num_labels):
    if method == "full_ft":
        model = FullRobertaSentimentClassifier(MODEL_NAME, num_labels)
        # Reduce memory pressure on the 4 GB GPU.
        model.roberta.gradient_checkpointing_enable()
        return model

    if method == "hybrid":
        return HybridPrefixPromptRobertaClassifier(
            MODEL_NAME,
            num_labels,
            prefix_length=PREFIX_LENGTH,
            prompt_length=PROMPT_LENGTH,
        )

    raise ValueError(method)


def worker_path(dataset_name, method, seed):
    return os.path.join(
        OUTPUT_DIR,
        f"worker_{dataset_name}_{method}_seed{seed}.json",
    )


def run_worker(dataset_name, method, seed):
    set_seed(seed)

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]
    train_ds = fixed_subset(dataset["train"], TRAIN_SIZE, DATA_SEED)
    eval_ds = fixed_subset(eval_source, EVAL_SIZE, DATA_SEED)

    set_seed(seed)
    model = build_model(method, num_labels)
    params = count_parameters(model)
    lr = float(LEARNING_RATES[dataset_name][method])

    run_dir = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_{method}_seed{seed}",
    )

    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        seed=int(seed),
        data_seed=int(seed),
        save_safetensors=False,
        dataloader_pin_memory=False,
        eval_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_total_mb = float(props.total_memory / (1024 ** 2))
    else:
        gpu_name = None
        gpu_total_mb = None

    started = time.time()
    train_output = trainer.train()
    train_seconds = time.time() - started
    metrics = trainer.evaluate()

    peak_alloc = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        if torch.cuda.is_available() else 0.0
    )
    peak_reserved = (
        float(torch.cuda.max_memory_reserved() / (1024 ** 2))
        if torch.cuda.is_available() else 0.0
    )

    row = {
        "experiment": EXPERIMENT,
        "purpose": "additional_backbone_sensitivity",
        "dataset": dataset_name,
        "method": method,
        "backbone": MODEL_NAME,
        "seed": int(seed),
        "data_seed": DATA_SEED,
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "effective_train_batch_size": TRAIN_BATCH * GRAD_ACCUM,
        "learning_rate": lr,
        "learning_rate_protocol": "transferred_from_primary_BERT_protocol",
        "prefix_length": PREFIX_LENGTH if method == "hybrid" else 0,
        "prompt_length": PROMPT_LENGTH if method == "hybrid" else 0,
        "matched_roberta_classification_head": True,
        **params,
        "train_time_seconds": float(train_seconds),
        "peak_train_gpu_allocated_mb": peak_alloc,
        "peak_train_gpu_reserved_mb": peak_reserved,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_mb,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "status": "success",
    }

    if hasattr(train_output, "metrics") and "train_loss" in train_output.metrics:
        row["train_loss"] = float(train_output.metrics["train_loss"])

    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            row[k] = float(v)
        else:
            row[k] = v

    return row


def worker_main(dataset_name, method, seed):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = worker_path(dataset_name, method, seed)

    try:
        row = run_worker(dataset_name, method, int(seed))
    except Exception as e:
        row = {
            "experiment": EXPERIMENT,
            "dataset": dataset_name,
            "method": method,
            "seed": int(seed),
            "status": "failed",
            "error": repr(e),
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print("\nWORKER RESULT")
    print(json.dumps(row, indent=2))

    if row.get("status") != "success":
        raise SystemExit(1)


def smoke_test():
    print("=" * 88)
    print("ROBERTA HYBRID SMOKE TEST")
    print("=" * 88)

    set_seed(42)
    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name="sst2",
        model_name=MODEL_NAME,
        max_length=32,
    )
    sample = dataset["train"].select(range(2))
    batch = {
        k: torch.stack([sample[i][k] for i in range(2)])
        for k in sample.column_names
        if k in {"input_ids", "attention_mask", "labels"}
    }

    model = HybridPrefixPromptRobertaClassifier(
        MODEL_NAME,
        num_labels,
        prefix_length=PREFIX_LENGTH,
        prompt_length=PROMPT_LENGTH,
    )
    model.eval()

    with torch.no_grad():
        out = model(**batch)

    print("logits shape:", tuple(out.logits.shape))
    print("loss:", float(out.loss))
    print("trainable params:", count_parameters(model)["trainable_params"])
    print("SMOKE TEST PASSED")


def collect_rows():
    rows = []
    for ds in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                p = worker_path(ds, method, seed)
                if os.path.exists(p):
                    try:
                        rows.append(json.load(open(p, "r", encoding="utf-8")))
                    except Exception:
                        pass
    return rows


def summarize(rows):
    ok = [r for r in rows if r.get("status") == "success"]
    summary = []

    for ds in DATASETS:
        for method in METHODS:
            g = [
                r for r in ok
                if r["dataset"] == ds and r["method"] == method
            ]
            if not g:
                continue

            def vals(key):
                return [float(r[key]) for r in g if r.get(key) is not None]

            a = vals("eval_accuracy")
            f = vals("eval_f1_macro")
            t = vals("train_time_seconds")
            m = vals("peak_train_gpu_allocated_mb")

            summary.append({
                "dataset": ds,
                "method": method,
                "backbone": MODEL_NAME,
                "n_runs": len(g),
                "seeds": ",".join(str(r["seed"]) for r in g),
                "learning_rate": float(g[0]["learning_rate"]),
                "learning_rate_protocol": g[0]["learning_rate_protocol"],
                "accuracy_mean": float(np.mean(a)) if a else None,
                "accuracy_std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                "macro_f1_mean": float(np.mean(f)) if f else None,
                "macro_f1_std": float(np.std(f, ddof=1)) if len(f) > 1 else 0.0,
                "train_time_seconds_mean": float(np.mean(t)) if t else None,
                "peak_train_gpu_allocated_mb_mean": float(np.mean(m)) if m else None,
                "trainable_params": int(g[0]["trainable_params"]),
                "trainable_ratio": float(g[0]["trainable_ratio"]),
            })

    return summary


def save_csv(rows, path):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def master_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 88)
    print("ROBERTA-BACKBONE SENSITIVITY | FULL FT vs HYBRID")
    print("roberta-base | SST-2 + IMDb | 3 seeds | fixed held-out subsets")
    print("=" * 88)

    for ds in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                p = worker_path(ds, method, seed)

                if os.path.exists(p):
                    try:
                        old = json.load(open(p, "r", encoding="utf-8"))
                        if old.get("status") == "success":
                            print(f"SKIP completed: {ds}/{method}/seed={seed}")
                            continue
                    except Exception:
                        pass

                print("\n" + "=" * 88)
                print(f"RUN | {ds} | {method} | seed={seed}")
                print("=" * 88)

                cmd = [
                    sys.executable,
                    "-u",
                    os.path.abspath(__file__),
                    "--worker",
                    ds,
                    method,
                    str(seed),
                ]
                completed = subprocess.run(cmd)

                if completed.returncode != 0:
                    print(
                        f"WARNING: failed {ds}/{method}/seed={seed}; "
                        "failure saved, continuing."
                    )

    rows = collect_rows()
    summary = summarize(rows)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_csv(rows, RESULTS_CSV)
    save_csv(summary, SUMMARY_CSV)

    print("\n" + "=" * 88)
    print("FINAL ROBERTA BACKBONE SUMMARY")
    print("=" * 88)

    for s in summary:
        print(
            f"{s['dataset']:5s} | {s['method']:7s} | n={s['n_runs']} | "
            f"acc={s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f} | "
            f"F1={s['macro_f1_mean']:.4f} +/- {s['macro_f1_std']:.4f} | "
            f"trainable={s['trainable_params']}"
        )

    print("\nResults:", RESULTS_JSON)
    print("Summary:", SUMMARY_JSON)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--smoke":
        smoke_test()
    elif len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        master_main()
