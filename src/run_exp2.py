import os
import sys
import yaml
import json
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from trainer_utils import make_small_subset, train_and_evaluate
from data import load_and_prepare_dataset
from metrics import count_parameters
from peft_methods import build_model_by_method


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_token_heatmap(values, tokens, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(max(10, len(tokens) * 0.35), 2.8))
    plt.imshow(values[np.newaxis, :], aspect="auto")
    plt.yticks([0], ["importance"])
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_bar_chart(labels, values, title, ylabel, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def prepare_inputs(example, device):
    inputs = {}

    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        if key in example:
            value = example[key]

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)

            if value.dim() == 1:
                value = value.unsqueeze(0)

            inputs[key] = value.to(device)

    return inputs


def extract_average_attention(
    model,
    tokenizer,
    eval_subset,
    max_examples,
    device,
):
    model.eval()
    model.to(device)

    collected_attentions = []
    selected_tokens = None

    for i in range(min(max_examples, len(eval_subset))):
        example = eval_subset[i]
        inputs = prepare_inputs(example, device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

        if outputs.attentions is None:
            continue

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        real_token_count = int(attention_mask.sum().item())
        tokens = tokenizer.convert_ids_to_tokens(input_ids[:real_token_count])

        last_attention = outputs.attentions[-1][0]

        seq_len_attention = last_attention.shape[-1]
        seq_len_input = input_ids.shape[0]

        extra_tokens = max(0, seq_len_attention - seq_len_input)

        start = extra_tokens
        end = start + real_token_count

        token_attention = last_attention[:, 0, start:end].mean(dim=0)

        token_attention = token_attention.detach().cpu().numpy()

        collected_attentions.append(token_attention)

        if selected_tokens is None:
            selected_tokens = tokens

    if not collected_attentions:
        return None, None, None

    min_len = min(len(x) for x in collected_attentions)
    collected_attentions = [x[:min_len] for x in collected_attentions]

    avg_attention = np.mean(collected_attentions, axis=0)
    selected_tokens = selected_tokens[:min_len]

    attention_concentration = float(np.max(avg_attention))
    attention_entropy = float(
        -np.sum(avg_attention * np.log(avg_attention + 1e-12))
    )

    return avg_attention, selected_tokens, {
        "attention_max": attention_concentration,
        "attention_entropy": attention_entropy,
    }


def build_ablation_summary(rows):
    summary = {}

    for row in rows:
        if row["status"] != "success":
            continue

        dataset = row["dataset"]
        method = row["method"]

        if dataset not in summary:
            summary[dataset] = {}

        summary[dataset][method] = row

    comparisons = []

    for dataset, methods in summary.items():
        if "hybrid" not in methods:
            continue

        hybrid = methods["hybrid"]

        for baseline in ["prefix", "prompt"]:
            if baseline not in methods:
                continue

            base = methods[baseline]

            comparisons.append({
                "dataset": dataset,
                "comparison": f"hybrid_vs_{baseline}",
                "accuracy_gain": float(hybrid["eval_accuracy"] - base["eval_accuracy"]),
                "f1_macro_gain": float(hybrid["eval_f1_macro"] - base["eval_f1_macro"]),
                "trainable_param_difference": int(
                    hybrid["trainable_params"] - base["trainable_params"]
                ),
                "train_time_difference_seconds": float(
                    hybrid["train_time_seconds"] - base["train_time_seconds"]
                ),
            })

    return comparisons


def main():
    config_path = os.path.join(PROJECT_DIR, "configs", "exp2.yaml")
    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    subset_cfg = config.get("subset", {})
    train_size = int(subset_cfg.get("train_size", 1500))
    eval_size = int(subset_cfg.get("eval_size", 300))

    analysis_cfg = config.get("analysis", {})
    attention_examples = int(analysis_cfg.get("attention_examples", 20))

    methods_to_run = [
        method for method in config["methods"]
        if method in ["prefix", "prompt", "hybrid"]
    ]

    print("=" * 70)
    print("Loaded experiment config successfully")
    print(f"Experiment name: {config['experiment_name']}")
    print(f"Datasets: {config['datasets']}")
    print(f"Methods : {methods_to_run}")
    print(f"Train size: {train_size}")
    print(f"Eval size : {eval_size}")
    print("=" * 70)

    all_results = []

    for dataset_name in config["datasets"]:
        print("\n" + "=" * 70)
        print(f"Running dataset: {dataset_name}")
        print("=" * 70)

        dataset, tokenizer, num_labels = load_and_prepare_dataset(
            dataset_name=dataset_name,
            model_name=config["model_name"],
            max_length=int(config["max_length"]),
        )

        train_subset, eval_subset = make_small_subset(
            dataset=dataset,
            train_size=train_size,
            eval_size=eval_size,
            seed=seed,
        )

        print(f"Training subset size: {len(train_subset)}")
        print(f"Eval subset size    : {len(eval_subset)}")

        dataset_rows = []

        for method_name in methods_to_run:
            print("\n" + "-" * 70)
            print(f"Experiment 2 | Dataset={dataset_name} | Method={method_name}")
            print("-" * 70)

            run_output_dir = os.path.join(
                PROJECT_DIR,
                config["output_dir"],
                dataset_name,
                method_name,
            )

            try:
                model = build_model_by_method(
                    method_name=method_name,
                    model_name=config["model_name"],
                    num_labels=num_labels,
                    config=config,
                )

                param_info = count_parameters(model)

                trainer, results = train_and_evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=train_subset,
                    eval_dataset=eval_subset,
                    config=config,
                    run_output_dir=run_output_dir,
                    method_name=method_name,
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"

                avg_attention, tokens, attention_stats = extract_average_attention(
                    model=trainer.model,
                    tokenizer=tokenizer,
                    eval_subset=eval_subset,
                    max_examples=attention_examples,
                    device=device,
                )

                if avg_attention is not None:
                    plot_token_heatmap(
                        values=avg_attention,
                        tokens=tokens,
                        title=f"{dataset_name} - {method_name} Average Attention",
                        save_path=os.path.join(
                            run_output_dir,
                            f"{dataset_name}_{method_name}_avg_attention_heatmap.png",
                        ),
                    )
                else:
                    attention_stats = {
                        "attention_max": None,
                        "attention_entropy": None,
                    }

                row = {
                    "experiment": "exp2",
                    "dataset": dataset_name,
                    "method": method_name,
                    "status": "success",
                    "total_params": param_info["total_params"],
                    "trainable_params": param_info["trainable_params"],
                    "trainable_ratio": param_info["trainable_ratio"],
                    "eval_accuracy": results.get("eval_accuracy"),
                    "eval_f1_macro": results.get("eval_f1_macro"),
                    "eval_f1_weighted": results.get("eval_f1_weighted"),
                    "eval_loss": results.get("eval_loss"),
                    "train_time_seconds": results.get("train_time_seconds"),
                    "peak_gpu_memory_mb": results.get("peak_gpu_memory_mb"),
                    "attention_max": attention_stats.get("attention_max"),
                    "attention_entropy": attention_stats.get("attention_entropy"),
                }

                all_results.append(row)
                dataset_rows.append(row)

                print("\nFinished run:")
                print(json.dumps(row, indent=2))

            except Exception as e:
                row = {
                    "experiment": "exp2",
                    "dataset": dataset_name,
                    "method": method_name,
                    "status": "failed",
                    "error": str(e),
                }

                all_results.append(row)
                dataset_rows.append(row)

                print("\nRun failed:")
                print(str(e))

        successful_rows = [
            row for row in dataset_rows
            if row["status"] == "success"
        ]

        if successful_rows:
            labels = [row["method"] for row in successful_rows]

            plot_bar_chart(
                labels=labels,
                values=[row["eval_accuracy"] for row in successful_rows],
                title=f"{dataset_name} Accuracy Comparison",
                ylabel="Accuracy",
                save_path=os.path.join(
                    PROJECT_DIR,
                    config["output_dir"],
                    dataset_name,
                    f"{dataset_name}_accuracy_comparison.png",
                ),
            )

            plot_bar_chart(
                labels=labels,
                values=[row["eval_f1_macro"] for row in successful_rows],
                title=f"{dataset_name} F1-Macro Comparison",
                ylabel="F1-Macro",
                save_path=os.path.join(
                    PROJECT_DIR,
                    config["output_dir"],
                    dataset_name,
                    f"{dataset_name}_f1_comparison.png",
                ),
            )

            plot_bar_chart(
                labels=labels,
                values=[row["trainable_ratio"] for row in successful_rows],
                title=f"{dataset_name} Trainable Parameter Ratio",
                ylabel="Trainable Ratio",
                save_path=os.path.join(
                    PROJECT_DIR,
                    config["output_dir"],
                    dataset_name,
                    f"{dataset_name}_trainable_ratio_comparison.png",
                ),
            )

    comparisons = build_ablation_summary(all_results)

    results_dir = os.path.join(PROJECT_DIR, config["output_dir"])
    os.makedirs(results_dir, exist_ok=True)

    results_json_path = os.path.join(results_dir, "exp2_results.json")
    results_csv_path = os.path.join(results_dir, "exp2_results.csv")
    comparison_json_path = os.path.join(results_dir, "exp2_ablation_comparison.json")
    comparison_csv_path = os.path.join(results_dir, "exp2_ablation_comparison.csv")

    save_json(all_results, results_json_path)
    save_csv(all_results, results_csv_path)

    save_json(comparisons, comparison_json_path)
    save_csv(comparisons, comparison_csv_path)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENT 2 RESULTS")
    print("=" * 70)
    print(json.dumps(all_results, indent=2))

    print("\n" + "=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)
    print(json.dumps(comparisons, indent=2))

    print(f"\nSaved results to: {results_json_path}")
    print(f"Saved CSV to: {results_csv_path}")
    print(f"Saved comparison to: {comparison_json_path}")
    print(f"Saved comparison CSV to: {comparison_csv_path}")


if __name__ == "__main__":
    main()