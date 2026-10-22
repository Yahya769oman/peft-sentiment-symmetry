import os
import time
import numpy as np
import torch

from transformers import Trainer, TrainingArguments
from metrics import compute_classification_metrics


def make_small_subset(dataset, train_size=300, eval_size=100, seed=42):
    train_ds = dataset["train"].shuffle(seed=seed).select(
        range(min(int(train_size), len(dataset["train"])))
    )

    if "validation" in dataset:
        eval_source = dataset["validation"]
    else:
        eval_source = dataset["test"]

    eval_ds = eval_source.shuffle(seed=seed).select(
        range(min(int(eval_size), len(eval_source)))
    )

    return train_ds, eval_ds


def normalize_strategy(value, default="no"):
    if value is None:
        return default

    if isinstance(value, bool):
        return "no" if value is False else default

    value = str(value).strip().lower()

    if value in ["false", "none", "null", "0"]:
        return "no"

    if value in ["true", "1"]:
        return default

    return value


def build_training_arguments(output_dir: str, config: dict, method_name: str):
    train_cfg = config["training"]

    learning_rate = float(train_cfg["learning_rate"])
    num_train_epochs = int(train_cfg["num_train_epochs"])

    method_overrides = config.get("method_overrides", {})

    if method_name in method_overrides:
        method_cfg = method_overrides[method_name]
        learning_rate = float(method_cfg.get("learning_rate", learning_rate))
        num_train_epochs = int(method_cfg.get("num_train_epochs", num_train_epochs))

    use_fp16 = bool(torch.cuda.is_available())

    save_strategy = normalize_strategy(
        train_cfg.get("save_strategy", "no"),
        default="no",
    )

    evaluation_strategy = normalize_strategy(
        train_cfg.get("evaluation_strategy", "epoch"),
        default="epoch",
    )

    load_best_model_at_end = bool(train_cfg.get("load_best_model_at_end", False))

    # إذا ما فيه حفظ، لا يمكن تحميل best model
    if save_strategy == "no":
        load_best_model_at_end = False

    # إذا طلبنا best model لازم save_strategy يطابق evaluation_strategy
    if load_best_model_at_end and save_strategy != evaluation_strategy:
        save_strategy = evaluation_strategy

    metric_for_best_model = str(
        train_cfg.get("metric_for_best_model", "eval_accuracy")
    )

    # Trainer يتوقع accuracy وليس eval_accuracy في بعض النسخ
    if metric_for_best_model.startswith("eval_"):
        metric_for_best_model = metric_for_best_model.replace("eval_", "", 1)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        learning_rate=learning_rate,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        logging_steps=int(train_cfg.get("logging_steps", 20)),

        # في نسختك السابقة كان eval_strategy هو الذي يشتغل
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,

        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=bool(train_cfg.get("greater_is_better", True)),

        report_to="none",
        save_total_limit=1,
        remove_unused_columns=False,
        fp16=use_fp16,

        # هذا هو التعديل المهم:
        # يحل مشكلة shared tensors في الموديل الهجين وقت حفظ checkpoint
        save_safetensors=False,
    )

    return args


def train_and_evaluate(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: dict,
    run_output_dir: str,
    method_name: str,
):
    os.makedirs(run_output_dir, exist_ok=True)

    args = build_training_arguments(
        output_dir=run_output_dir,
        config=config,
        method_name=method_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_classification_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    trainer.train()

    train_time = time.time() - start_time

    eval_results = trainer.evaluate()

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory = 0.0

    results = {
        "train_time_seconds": float(train_time),
        "peak_gpu_memory_mb": float(peak_memory),
    }

    for k, v in eval_results.items():
        if isinstance(v, (int, float, np.floating)):
            results[k] = float(v)
        else:
            results[k] = v

    return trainer, results