# Experiment-to-file map

| Analysis | Core script(s) | Saved evidence | Status / interpretation |
|---|---|---|---|
| BERT Full FT primary | `src/run_full_ft_fresh_process.py`, `src/run_full_ft_5seed.py` | `outputs/full_ft_fresh_process/` | Five-seed held-out aggregate |
| BERT Hybrid / LoRA / Adapter / BitFit | `src/run_exp2_peft_official.py`, `src/models_symmetry.py`, `src/models_peft_symmetry.py` | `outputs/exp2_peft_official/` | Five-seed held-out aggregate |
| IA3 | `src/run_ia3_official_5seed.py` | `outputs/ia3_official_5seed/` | Five-seed held-out aggregate |
| Prefix-only / Prompt-only independently tuned ablation | `src/run_exp3_ablation_independent_tuning.py`, `src/models_ablation_symmetry.py` | `outputs/exp3_ablation_independent_tuning/` | Training-only validation selection followed by five-seed held-out evaluation |
| Hybrid length sensitivity | `src/run_hybrid_length_sensitivity.py`, `src/run_hybrid_tuned_official.py` | `outputs/hybrid_length_sensitivity/`, `outputs/hybrid_tuned_official/` | Different, three-seed diagnostic protocol |
| IMDb context length | `src/run_imdb_sequence_length_sensitivity.py` | `outputs/imdb_sequence_length_sensitivity/` | Three-seed diagnostic |
| Inference on trained checkpoints | `src/run_trained_checkpoint_inference_benchmark.py` | `outputs/trained_checkpoint_inference_benchmark/` | Timings/results stored; weight files not included |
| Subset robustness | `src/run_data_subset_robustness.py` | `outputs/data_subset_robustness/` | Three training-subset seeds, fixed training seed |
| Paired tests / Pareto | `src/run_paired_significance.py`, `src/run_pareto_figures.py` | `outputs/statistical_significance/`, `outputs/pareto_figures/` | Saved statistical and figure outputs |
| Seed-level retention uncertainty | `src/run_retention_uncertainty_analysis.py` | `outputs/retention_uncertainty_analysis/` | Reproduces reported retention intervals from saved records |
| RoBERTa backbone diagnostic | `src/run_roberta_backbone_sensitivity.py` | `outputs/roberta_backbone_sensitivity/` | Three-seed Full-FT vs Hybrid with transferred BERT LR policy |
| Additional RoBERTa LR check | `src/run_roberta_lr_review.py` | `outputs/roberta_lr_review/` | Training-only LR sweep and seed-42 held-out check; **not** three-seed re-evaluation |
| Additional Full FT LR tuning | `src/run_full_ft_lr_tuning_independent.py`, `src/run_full_ft_tuned_heldout_seed42.py` | `outputs/full_ft_lr_tuning_independent/`, `outputs/full_ft_tuned_heldout_seed42/` | Single-seed tuning and follow-up; **not** five-seed replacement |
| Extended learning-rate and matched-budget design | `src/run_review_lr_joint_fairness.py` | `outputs/reviewer3_wide_lr_matched_hybrid/` | **Incomplete**: SST-2 Prefix/Prompt validation records only; held-out / full Hybrid grid absent |

## Interpreting historical experiments

The `exp1` ... `exp8`, `context_gated_*`, and other exploratory directories retain archived pilot and sensitivity data. Do not pool their records with the principal fixed 3,000/500, five-seed protocol unless the datasets, seeds, hyperparameters and evaluation split are actually matched. Some `configs/configs/` files are historical duplicates from earlier project organization.
