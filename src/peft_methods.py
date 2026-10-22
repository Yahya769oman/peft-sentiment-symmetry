from peft import (
    get_peft_model,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    LoraConfig,
    TaskType,
)

from models import (
    load_base_model,
    HybridPrefixPromptClassifierModel,
    GenericInputPrefixPromptClassifierModel,
    AdapterBertClassifierModel,
)


def build_full_model(model_name: str, num_labels: int):
    return load_base_model(model_name, num_labels)


def build_prefix_model(model_name: str, num_labels: int, num_virtual_tokens: int):
    model = load_base_model(model_name, num_labels)

    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=int(num_virtual_tokens),
    )

    return get_peft_model(model, peft_config)


def build_prompt_model(model_name: str, num_labels: int, num_virtual_tokens: int):
    model = load_base_model(model_name, num_labels)

    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=int(num_virtual_tokens),
    )

    return get_peft_model(model, peft_config)


def build_hybrid_model(
    model_name: str,
    num_labels: int,
    prefix_tokens: int,
    prompt_tokens: int,
    unfreeze_last_n_layers: int = 0,
    use_input_level_hybrid: bool = False,
):
    if use_input_level_hybrid:
        return GenericInputPrefixPromptClassifierModel(
            model_name=model_name,
            num_labels=num_labels,
            prefix_length=int(prefix_tokens),
            prompt_length=int(prompt_tokens),
            freeze_backbone=True,
        )

    return HybridPrefixPromptClassifierModel(
        model_name=model_name,
        num_labels=num_labels,
        prefix_length=int(prefix_tokens),
        prompt_length=int(prompt_tokens),
        unfreeze_last_n_layers=int(unfreeze_last_n_layers),
    )


def build_lora_model(
    model_name: str,
    num_labels: int,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
):
    model = load_base_model(model_name, num_labels)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=["query", "value"],
    )

    return get_peft_model(model, peft_config)


def build_bitfit_model(model_name: str, num_labels: int):
    model = load_base_model(model_name, num_labels)

    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def build_adapter_model(
    model_name: str,
    num_labels: int,
    bottleneck_size: int = 64,
    dropout: float = 0.1,
):
    return AdapterBertClassifierModel(
        model_name=model_name,
        num_labels=num_labels,
        bottleneck_size=int(bottleneck_size),
        dropout=float(dropout),
    )


def build_model_by_method(
    method_name: str,
    model_name: str,
    num_labels: int,
    config: dict,
):
    method_name = method_name.lower().strip()

    if method_name == "full":
        return build_full_model(model_name, num_labels)

    if method_name == "prefix":
        return build_prefix_model(
            model_name=model_name,
            num_labels=num_labels,
            num_virtual_tokens=config["prefix"]["num_virtual_tokens"],
        )

    if method_name == "prompt":
        return build_prompt_model(
            model_name=model_name,
            num_labels=num_labels,
            num_virtual_tokens=config["prompt"]["num_virtual_tokens"],
        )

    if method_name == "hybrid":
        hybrid_cfg = config["hybrid"]

        return build_hybrid_model(
            model_name=model_name,
            num_labels=num_labels,
            prefix_tokens=hybrid_cfg["prefix_virtual_tokens"],
            prompt_tokens=hybrid_cfg["prompt_virtual_tokens"],
            unfreeze_last_n_layers=hybrid_cfg.get("unfreeze_last_n_layers", 0),
            use_input_level_hybrid=hybrid_cfg.get("use_input_level_hybrid", False),
        )

    if method_name == "lora":
        lora_cfg = config.get("lora", {})

        return build_lora_model(
            model_name=model_name,
            num_labels=num_labels,
            r=lora_cfg.get("r", 8),
            alpha=lora_cfg.get("alpha", 16),
            dropout=lora_cfg.get("dropout", 0.1),
        )

    if method_name == "bitfit":
        return build_bitfit_model(
            model_name=model_name,
            num_labels=num_labels,
        )

    if method_name == "adapter":
        adapter_cfg = config.get("adapter", {})

        return build_adapter_model(
            model_name=model_name,
            num_labels=num_labels,
            bottleneck_size=adapter_cfg.get("bottleneck_size", 64),
            dropout=adapter_cfg.get("dropout", 0.1),
        )

    raise ValueError(f"Unsupported method: {method_name}")