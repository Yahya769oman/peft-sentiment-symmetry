import os
import sys
import yaml
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import List, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters
from peft_methods import build_model_by_method
from trainer_utils import train_and_evaluate


# ========================
# Utils
# ========================

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_subset(dataset_split, size: int, seed: int):
    size = min(int(size), len(dataset_split))
    return dataset_split.shuffle(seed=seed).select(range(size))


def decode_tokens(tokenizer, input_ids: torch.Tensor) -> List[str]:
    ids = input_ids.detach().cpu().tolist()

    if len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]

    return tokenizer.convert_ids_to_tokens(ids)


def prepare_model_inputs(example: dict, device: str) -> dict:
    model_inputs = {}

    for key in ["input_ids", "attention_mask", "token_type_ids", "labels"]:
        if key in example:
            value = example[key]

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)

            if value.dim() == 1:
                value = value.unsqueeze(0)

            model_inputs[key] = value.to(device)

    return model_inputs


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x_norm = x / norms
    return np.matmul(x_norm, x_norm.T)


# ========================
# Plotting
# ========================

def plot_token_heatmap(values, tokens, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(max(10, len(tokens) * 0.4), 2.5))
    plt.imshow(values[np.newaxis, :], aspect="auto")
    plt.yticks([0], ["importance"])
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_layer_attention(layer_values, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x = list(range(len(layer_values)))

    plt.figure(figsize=(8, 4))
    plt.plot(x, layer_values, marker="o")
    plt.xticks(x, [f"L{i}" for i in x])
    plt.xlabel("Layer")
    plt.ylabel("Attention Score (CLS → Tokens)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ========================
# 🔥 التحليل الصحيح (المعدل)
# ========================

def analyze_single_example(model, tokenizer, example: dict, device: str) -> Dict:
    model_inputs = prepare_model_inputs(example, device)

    with torch.no_grad():
        outputs = model(
            **model_inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    logits = outputs.logits
    pred_label = int(torch.argmax(logits, dim=-1).cpu().item())

    true_label = None
    if "labels" in model_inputs:
        true_label = int(model_inputs["labels"].cpu().item())

    attentions = outputs.attentions
    hidden_states = outputs.hidden_states

    input_ids = model_inputs["input_ids"]
    tokens = decode_tokens(tokenizer, input_ids)

    attention_mask = model_inputs["attention_mask"]
    real_token_count = int(attention_mask[0].sum().item())

    seq_len_attn = attentions[-1].shape[-1]
    extra_tokens = seq_len_attn - len(tokens)

    start = extra_tokens
    end = start + real_token_count

    layer_attention_means = []

    for layer_attn in attentions:
        attn = layer_attn[0]  # [heads, seq, seq]

        # 🔥 CLS → tokens فقط
        cls_to_tokens = attn[:, 0, start:end]

        layer_score = float(cls_to_tokens.mean().cpu().item())
        layer_attention_means.append(layer_score)

    last_attention = attentions[-1][0]

    token_attention = last_attention[:, 0, start:end]
    token_importance = token_attention.mean(dim=0).cpu().numpy()

    clean_tokens = tokens[:real_token_count]

    last_hidden = hidden_states[-1][0]
    cls_representation = last_hidden[0].cpu().numpy()

    return {
        "pred_label": pred_label,
        "true_label": true_label,
        "layer_attention_means": layer_attention_means,
        "token_importance": token_importance,
        "pooled_repr": cls_representation,
        "tokens": clean_tokens,
    }


# ========================
# Main Analysis
# ========================

def analyze_model_outputs(model, tokenizer, dataset, output_dir, method_name):
    device = get_device()
    model.to(device)
    model.eval()

    all_layer_means = []
    all_representations = []
    all_preds = []
    all_labels = []

    for i, example in enumerate(dataset):
        result = analyze_single_example(model, tokenizer, example, device)

        all_layer_means.append(result["layer_attention_means"])
        all_representations.append(result["pooled_repr"])
        all_preds.append(result["pred_label"])
        all_labels.append(result["true_label"])

        # نرسم Token Heatmap لأول 3 عينات فقط
        if i < 3:
            plot_token_heatmap(
                result["token_importance"],
                result["tokens"],
                f"{method_name} Token Importance Example {i}",
                os.path.join(output_dir, f"{method_name}_tokens_{i}.png"),
            )

    avg_attention_per_layer = np.array(all_layer_means).mean(axis=0)

    plot_layer_attention(
        avg_attention_per_layer,
        f"{method_name} Attention",
        os.path.join(output_dir, f"{method_name}_layer_attention.png"),
    )

    representations = np.array(all_representations)
    cosine_sim = cosine_similarity_matrix(representations)

    return {
        "analysis_accuracy": float(np.mean(np.array(all_preds) == np.array(all_labels))),
        "avg_attention_per_layer": avg_attention_per_layer.tolist(),
        "avg_cosine_similarity": float(cosine_sim.mean()),
    }


# ========================
# Main
# ========================

def main():
    config = load_config(os.path.join(PROJECT_DIR, "configs", "exp3.yaml"))

    set_seed(config["seed"])

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=config["dataset"],
        model_name=config["model_name"],
        max_length=config["max_length"],
    )

    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]

    train_subset = make_subset(dataset["train"], 1000, config["seed"])
    eval_subset = make_subset(eval_source, 300, config["seed"])

    analysis_subset = make_subset(eval_subset, 50, config["seed"])

    all_results = []

    for method in config["methods"]:
        print(f"\nRunning EXP3: {method}")

        model = build_model_by_method(
            method_name=method,
            model_name=config["model_name"],
            num_labels=num_labels,
            config=config,
        )

        param_info = count_parameters(model)

        trainer, train_results = train_and_evaluate(
            model,
            tokenizer,
            train_subset,
            eval_subset,
            config,
            os.path.join(PROJECT_DIR, config["output_dir"], method),
            method,
        )

        analysis = analyze_model_outputs(
            trainer.model,
            tokenizer,
            analysis_subset,
            os.path.join(PROJECT_DIR, config["output_dir"], method),
            method,
        )

        result = {
            "method": method,
            "eval_accuracy": train_results.get("eval_accuracy"),
            "trainable_ratio": param_info["trainable_ratio"],
            **analysis
        }

        all_results.append(result)

        print(result)

    save_json(
        {"results": all_results},
        os.path.join(PROJECT_DIR, config["output_dir"], "exp3_results.json"),
    )


if __name__ == "__main__":
    main()