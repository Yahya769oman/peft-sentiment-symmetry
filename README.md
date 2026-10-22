# Dataset and Backbone Effects on Accuracy–Efficiency Asymmetry in Hybrid Prefix–Prompt Tuning for Sentiment Analysis

This repository provides the experimental code and saved results accompanying the manuscript submitted to the journal Symmetry. Manuscript ID: symmetry-4631037.

The study examines sentiment classification on SST-2 and IMDb using BERT-base-uncased. It compares full fine-tuning, Hybrid Prefix–Prompt tuning, LoRA, layer-wise adapters, BitFit, and IA3. Additional experiments cover independently tuned Prefix-only and Prompt-only models, sequence-length sensitivity, inference efficiency, statistical analysis, and RoBERTa-base.

## Repository structure

| Path | Description |
| --- | --- |
| `src/` | Training, evaluation, model implementations, and analysis scripts |
| `outputs/` | Saved experiment results and summary files |
| `configs/` | Experiment configuration files, including historical configurations |
| `docs/EXPERIMENT_MAP.md` | Mapping of analyses to their scripts and saved results |
| `docs/REPRODUCIBILITY_NOTES.md` | Data protocols, supplementary experiments, and reproducibility limitations |
| `tools/verify_repository.py` | Offline checks for file integrity and selected numerical summaries |
| `requirements.txt` | Python dependencies |

## Software and hardware

The experiments used Python 3.10.20, PyTorch 2.6.0+cu124, Transformers 4.57.6, PEFT 0.17.1, and an NVIDIA Quadro P520 GPU with approximately 4 GB of memory. The original environment was Windows with Anaconda.

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Check the source files and stored numerical summaries without retraining:

```bash
python tools/verify_repository.py
```

The verification command parses the Python scripts and checks saved result consistency. It does not download datasets, load trained checkpoints, or reproduce training runs.

## Experimental protocol

The principal BERT comparison uses:

- Datasets: SST-2 and IMDb
- Training subset: 3,000 examples per dataset
- Held-out evaluation subset: 500 examples per dataset
- Training seeds: 42, 123, 2026, 3407, and 7777
- Sequence-length limit: 128 tokens
- Training budget: three epochs

SST-2 held-out examples were sampled from the official validation split; IMDb held-out examples were sampled from the official test split. Hyperparameter selection used a separate validation sample from the training split. Diagnostic experiments have their own protocols and repetition counts, documented in `docs/EXPERIMENT_MAP.md` and `docs/REPRODUCIBILITY_NOTES.md`.

## Running experiments

Run scripts from the repository root. Examples:

```bash
# Prefix-only and Prompt-only selection followed by held-out evaluation
python src/run_exp3_ablation_independent_tuning.py

# Retention estimates and confidence intervals from saved results
python src/run_retention_uncertainty_analysis.py

# Lightweight RoBERTa implementation check
python src/run_roberta_backbone_sensitivity.py --smoke
```

Training scripts can require internet access to download public models and datasets, as well as substantial GPU time. See the corresponding source files for configuration details.

## Results and reproducibility

Saved results for the principal five-seed comparisons are provided separately from supplementary and diagnostic runs. The additional wide-learning-rate screen under `outputs/reviewer3_wide_lr_matched_hybrid/` contains 31 completed SST-2 Prefix-only/Prompt-only validation trials; the full matched-budget grid and subsequent five-seed held-out evaluation were not completed. These validation results are not presented as final multi-seed test results.

Additional RoBERTa learning-rate and Full-FT learning-rate checks are single-seed analyses. They do not replace the corresponding principal multi-seed results.

Large model checkpoints and raw dataset copies are not distributed. A complete end-to-end rerun on a clean machine has not been verified. Details of available evidence and known limitations are in `docs/REPRODUCIBILITY_NOTES.md`.

## Data and reuse

The experiments use public SST-2 and IMDb datasets loaded through Hugging Face `datasets`. Results and scripts are included for inspection and reproducibility assessment. No software license is included; contact the authors to discuss permissions for reuse or redistribution. Third-party models and datasets are governed by their respective terms.
