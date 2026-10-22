# Reproducibility notes and archive review

## What was checked

- The original user-supplied archive opened cleanly with ZIP CRC verification.
- All 54 top-level Python modules in `src/` passed Python syntax compilation.
- Python import dependencies between local modules were checked statically; full dependency/runtime execution was not performed.
- Saved JSON result files were read as strict JSON, without `NaN` or `Infinity` tokens.
- Relevant principal summary means were checked using independent calculation against their recorded summary rows by `tools/verify_repository.py`.
- Personally identifying local Windows project-root strings in old result metadata were replaced with relative paths. Metric values, seeds, losses, times, and accuracy/F1 statistics were not intentionally changed.
- `src/collect_all_experiments.py` was made portable (relative root and neutral archive filename). SciPy was declared in `requirements.txt`, as it is explicitly imported by the statistical scripts.

## Data and evaluation separation

- Principal BERT training/evaluation subsets: 3,000 / 500; five model-training seeds.
- Ablation method selection uses another 500-example training-only validation slice before testing on the held-out 500 examples.
- Additional wide-LR trial results under `outputs/reviewer3_wide_lr_matched_hybrid/selection/` are *validation records only*, and no new five-seed outcomes are included for this trial series.
- `outputs/roberta_lr_review/` holds a four-rate validation sweep and single-seed held-out result for each dataset.
- `outputs/full_ft_lr_tuning_independent/` holds training-only validation LR selection and `outputs/full_ft_tuned_heldout_seed42/` contains single-seed held-out reevaluation.

## Known limits

- The main model checkpoint weight tensors were not part of the source archive and are not distributed. Historical output checkpoint folder metadata was excluded to avoid distributing redundant tokenizer snapshots and misleading references to missing weight files.
- End-to-end GPU reruns were not conducted during repository packaging.
- The saved trial-selection grid is incomplete; the `grid_status.json` may not reflect the latest individual per-run JSON files if `report` was not rerun. Count trials using `selection/*.json`, and do not report incomplete grids as matched searches.
- Per-example prediction arrays are not established as available from the shared archive. Some requested paired example-level tests cannot be regenerated without additional predictions.
- `README.md` describes source/result availability, not a guarantee of bitwise training reproducibility.

## Source changes during packaging

- No result array or numeric metric was fabricated.
- Minor portability edits to `src/collect_all_experiments.py`, dependency declaration in `requirements.txt`, and Windows-path normalization in a small number of output metadata strings were made. These edits do not reproduce any training and should not be represented as additional experimental evidence.
