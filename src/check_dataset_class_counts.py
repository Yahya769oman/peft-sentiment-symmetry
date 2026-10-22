"""Verify fixed dataset-subset class counts without training any model.
Run from the project's src directory or any directory with `datasets` installed.
"""
import csv
import json
from pathlib import Path
from datasets import load_dataset

DATA_SEED = 2026
TRAIN_SIZE = 3000
EVAL_SIZE = 500
SOURCES = {'sst2': 'stanfordnlp/sst2', 'imdb': 'imdb'}

def counts(split):
    labels = split['label']
    return {'negative_0': sum(int(x) == 0 for x in labels),
            'positive_1': sum(int(x) == 1 for x in labels),
            'total': len(labels)}

def main():
    rows = []
    for name, path in SOURCES.items():
        dataset = load_dataset(path)
        evaluation_split = 'validation' if 'validation' in dataset else 'test'
        for label, source, n in [('train', 'train', TRAIN_SIZE),
                                 ('heldout', evaluation_split, EVAL_SIZE)]:
            subset = dataset[source].shuffle(seed=DATA_SEED).select(range(min(n, len(dataset[source]))))
            row = {'dataset': name, 'subset': label, 'original_split': source,
                   'data_seed': DATA_SEED, **counts(subset)}
            rows.append(row)
            print(f"{name:5} {label:7} negative={row['negative_0']} positive={row['positive_1']} total={row['total']}")
    dest = Path(__file__).resolve().parent
    (dest / 'dataset_class_counts.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    with (dest / 'dataset_class_counts.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved dataset_class_counts.json and dataset_class_counts.csv')

if __name__ == '__main__':
    main()
