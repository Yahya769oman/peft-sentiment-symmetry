from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_CONFIG = {
    "imdb": {
        "path": "imdb",
        "text_col": "text",
        "label_col": "label",
    },
    "sst2": {
        "path": "stanfordnlp/sst2",
        "text_col": "sentence",
        "label_col": "label",
    },
    # Used for EXP6 cross-domain transfer.
    # Binary sentiment labels: 0/1.
    "yelp_polarity": {
        "path": "yelp_polarity",
        "text_col": "text",
        "label_col": "label",
    },
    # Amazon Polarity has title + content; both are joined during preprocessing.
    "amazon_polarity": {
        "path": "amazon_polarity",
        "text_cols": ["title", "content"],
        "label_col": "label",
    },
}


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def load_raw_dataset(dataset_name: str):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {list(DATASET_CONFIG.keys())}"
        )

    cfg = DATASET_CONFIG[dataset_name]
    dataset = load_dataset(cfg["path"])
    label_col = cfg.get("label_col", "label")

    if "text_cols" in cfg:
        return dataset, cfg["text_cols"], label_col

    return dataset, cfg["text_col"], label_col


def _build_text_batch(examples, text_col):
    if isinstance(text_col, (list, tuple)):
        first_col = text_col[0]
        batch_size = len(examples[first_col])
        texts = []

        for i in range(batch_size):
            parts = []

            for col in text_col:
                value = examples[col][i]
                if value is not None:
                    parts.append(str(value))

            texts.append(" ".join(parts))

        return texts

    return examples[text_col]


def tokenize_dataset(dataset, tokenizer, text_col, max_length: int):
    def preprocess(examples):
        texts = _build_text_batch(examples, text_col)

        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=int(max_length),
        )

    return dataset.map(preprocess, batched=True)


def prepare_dataset_for_training(dataset, label_col: str):
    keep_columns = ["input_ids", "attention_mask", "token_type_ids", label_col]

    for split in dataset.keys():
        remove_cols = [
            col for col in dataset[split].column_names
            if col not in keep_columns
        ]

        dataset[split] = dataset[split].remove_columns(remove_cols)

    if label_col != "labels":
        dataset = dataset.rename_column(label_col, "labels")

    dataset.set_format(type="torch")

    return dataset


def load_and_prepare_dataset(dataset_name: str, model_name: str, max_length: int):
    raw_dataset, text_col, label_col = load_raw_dataset(dataset_name)

    tokenizer = get_tokenizer(model_name)

    tokenized_dataset = tokenize_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        text_col=text_col,
        max_length=max_length,
    )

    prepared_dataset = prepare_dataset_for_training(
        dataset=tokenized_dataset,
        label_col=label_col,
    )

    labels = prepared_dataset["train"]["labels"]
    num_labels = len(set(int(x) for x in labels))

    return prepared_dataset, tokenizer, num_labels