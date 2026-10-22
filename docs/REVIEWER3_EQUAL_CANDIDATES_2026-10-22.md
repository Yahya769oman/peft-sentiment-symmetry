# 22 October 2026 supplementary equal-candidate experiment

This package adds the completed, *supplementary* matched-candidate-count experiment supporting the revised response to Reviewer 3 (comment 6) for Symmetry manuscript **symmetry-4631037**.

## Protocol

Nine predeclared selection candidates per dataset/method: for Hybrid, prefix/prompt pairs (8,10), (16,20), (32,30) crossed with learning rates 1e-4, 3e-4, 5e-4; for Prefix-only and Prompt-only, the nine candidate records were archived from the earlier independent-ablation grid. Selection uses one training seed (42), 3,000 training examples, 500 distinct training-derived validation examples, and final-epoch validation Macro-F1. After selection, new Hybrid held-out training used seeds 42, 123, 2026 and the same fixed official 500-example held-out subset used by the archived component-only runs.

## Three-seed held-out means ± sample SD

| Dataset | Prefix-only accuracy | Prompt-only accuracy | Hybrid accuracy | Prefix-only Macro-F1 | Prompt-only Macro-F1 | Hybrid Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| SST-2 | 83.80 ± 0.20% | 86.73 ± 0.81% | 87.07 ± 0.23% | 83.71 ± 0.20% | 86.70 ± 0.82% | 87.05 ± 0.22% |
| IMDb | 80.40 ± 0.20% | 81.27 ± 0.42% | 80.67 ± 1.17% | 80.40 ± 0.20% | 81.26 ± 0.41% | 80.66 ± 1.17% |

The Prefix-only/Prompt-only results are archived in `outputs/exp3_ablation_independent_tuning/ablation_tuned_official_results.json` (already in the main repository); this new folder includes Hybrid selection and held-out outputs only. The experiment does **not** equalize trainable parameter counts and is **not** an exhaustive prefix-length × prompt-length × LR factorial search. Small Hybrid-vs-Prompt-only differences have three-seed paired confidence intervals crossing zero; no statistically established universal superiority is claimed.

## Provenance / diagnostic caution

Two SST-2 selection-result JSONs, `sst2_hybrid_pfx32_prm30_lr1e-04_seed42.json` and `sst2_hybrid_pfx32_prm30_lr3e-04_seed42.json`, were reconstructed from the author's completed terminal output after a nonfinite gradient-norm logging value (`inf`) caused original JSON serialization to fail. They are explicitly marked as recovered, contain final performance values but *not* complete training histories or fingerprints, and neither was selected. The `src/run_8h_reviewer3_hybrid.py` version in this supplement includes the logging serialization hotfix; it does not change objective, validation, final evaluation, or model calculations.

The public project repository's earlier `reviewer3_wide_lr_matched_hybrid` run remains incomplete; that is **a separate experiment**. Do not overwrite or relabel its outcomes. The new candidate-count supplement is distinct from the principal 5-seed comparison. Saved results do not independently verify an end-to-end clean-machine reproduction, and intermediate model checkpoints are not published.

## Reproduction / checks

From the repository root, `python src/run_8h_reviewer3_hybrid.py --phase plan` summarizes the selection protocol; `--phase select` summarizes selected configurations; the script supports `--phase heldout` and `--phase all` for new training. These commands can download models and datasets and use GPU resources. `outputs/reviewer3_8h_fair_hybrid/heldout_summary.json` gives the completed held-out summary.

## Aggregate export note

`heldout_summary_hybrid_both_datasets.json` and `.csv` were calculated from the six saved Hybrid held-out seed files to show both datasets. The original `heldout_summary.json` is the final SST-2-specific phase export and is deliberately preserved unchanged. The previous prefix-only/prompt-only official results remain in the earlier `outputs/exp3_ablation_independent_tuning/` directory.
