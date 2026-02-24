# Results Template

Use this file to report benchmark and ablation outcomes in paper-style format.

## 1) Experimental Setup

- Code version / commit:
- Date:
- Python version:
- Command used:
  - `python evaluation\\run_benchmarks.py --seeds 5`
- Benchmark source:
  - `evaluation/benchmarks.json`

## 2) Datasets / Tasks

- liquid
- scifi_movies
- devices
- noisy_food_adversarial
- marine_animals_adversarial

## 3) Metrics

- Accuracy
- Brier score (calibration)
- Convergence step (stable top theory window=3)
- Entropy slope
- Correction count

## 4) Main Table (Mean ± Std across seeds)

Fill from `evaluation/results/summary_metrics.csv`.

| Benchmark | Accuracy | Brier | Convergence Step | Entropy Slope | Corrections |
|---|---:|---:|---:|---:|---:|
| liquid |  |  |  |  |  |
| scifi_movies |  |  |  |  |  |
| devices |  |  |  |  |  |
| noisy_food_adversarial |  |  |  |  |  |
| marine_animals_adversarial |  |  |  |  |  |

## 5) Plot Review

Attach/interpret:

- `accuracy_by_benchmark.png`
- `brier_by_benchmark.png`
- `convergence_by_benchmark.png`
- `corrections_by_benchmark.png`
- `entropy_slope_by_benchmark.png`
- `entropy_trend_by_benchmark.png`

## 6) Error Analysis Examples

Document at least 3 qualitative failures and what mechanism caused/recovered them.

1. Example:
   - Input sequence:
   - Failure mode:
   - Hypothesis drift observed:
   - Recovery mechanism:

2. Example:
   - Input sequence:
   - Failure mode:
   - Hypothesis drift observed:
   - Recovery mechanism:

3. Example:
   - Input sequence:
   - Failure mode:
   - Hypothesis drift observed:
   - Recovery mechanism:

## 7) Ablation Plan (Next Step)

Target toggles:

- recency blend
- stale-feature demotion
- active-learning cooldown
- confirmation-memory floor
- anchor override

Report delta from full model for each metric and benchmark.

## 8) Ablation Table (Full vs Variant)

Fill from `evaluation/results/ablation_deltas.csv`.

Interpretation:

- `delta_accuracy < 0` means removing that mechanism hurts accuracy (mechanism is beneficial).
- `delta_brier > 0` means removing mechanism worsens calibration.
- `delta_convergence_step > 0` means slower convergence when ablated.

| Benchmark | Ablation | Δ Accuracy | Δ Brier | Δ Convergence | Δ Entropy Slope | Δ Corrections |
|---|---|---:|---:|---:|---:|---:|
| liquid | ablate_recency_blend |  |  |  |  |  |
| liquid | ablate_stale_feature_demotion |  |  |  |  |  |
| scifi_movies | ablate_anchor_override |  |  |  |  |  |
| devices | ablate_confirmation_memory_floor |  |  |  |  |  |
