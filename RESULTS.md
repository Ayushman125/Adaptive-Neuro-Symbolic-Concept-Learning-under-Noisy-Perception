# Results

This report summarizes the current benchmark + ablation run outputs in `evaluation/results`.

## 1) Experimental Setup

- Code version / commit: `3e2e49a`
- Date: 2026-02-24
- Python version: `Python 3.11.9`
- Command family used:
  - `python evaluation\\run_benchmarks.py --seeds 2` (full + ablations)
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

## 4) Main Table (Mean ± Std across seeds, Full Model)

Source: `evaluation/results/summary_metrics.csv` where `ablation=full`.

| Benchmark | Accuracy | Brier | Convergence Step | Entropy Slope | Corrections |
|---|---:|---:|---:|---:|---:|
| liquid | 0.90 ± 0.00 | 0.4286 ± 0.0093 | 3.00 ± 0.00 | -0.1949 ± 0.0016 | 4.00 ± 0.00 |
| scifi_movies | 0.85 ± 0.05 | 0.3209 ± 0.0005 | 3.00 ± 0.00 | -0.1347 ± 0.0244 | 4.50 ± 0.50 |
| devices | 0.90 ± 0.00 | 0.4102 ± 0.0266 | 3.00 ± 0.00 | +0.0760 ± 0.0557 | 7.00 ± 0.00 |
| noisy_food_adversarial | 0.75 ± 0.05 | 0.2899 ± 0.0020 | 5.00 ± 1.00 | -0.2526 ± 0.0122 | 2.50 ± 0.50 |
| marine_animals_adversarial | 0.90 ± 0.00 | 0.3161 ± 0.0250 | 3.00 ± 0.00 | -0.0792 ± 0.0102 | 2.50 ± 2.50 |

### Quick interpretation

- Accuracy is strong on liquid/devices/marine tasks (0.90), moderate on sci-fi (0.85), and hardest on noisy_food (0.75).
- Calibration remains moderate (Brier ~0.29–0.43).
- Devices still shows positive entropy slope, indicating uncertainty growth in that benchmark.

## 5) Plot Review

Generated in `evaluation/results/`:

- `accuracy_by_benchmark.png`
- `brier_by_benchmark.png`
- `convergence_by_benchmark.png`
- `corrections_by_benchmark.png`
- `entropy_slope_by_benchmark.png`
- `entropy_trend_by_benchmark.png`
- `ablation_delta_accuracy.png`
- `ablation_delta_brier.png`
- `ablation_delta_convergence.png`

## 6) Error Analysis Examples

1. Liquid-style confounder drift
   - Input pattern: liquid positives mixed with metallic/toxic negatives
   - Failure mode: proxy features (e.g., non-core correlates) temporarily overtake concept anchor
   - Hypothesis drift observed: disjunctions include noisy side-features
   - Recovery mechanism: recency blend + active correction + confirmation floor

2. Noisy-food adversarial overlap
   - Input pattern: edible but non-fruit distractors share sugar/roundness traits
   - Failure mode: partial semantic overlap reduces discriminative clarity
   - Hypothesis drift observed: concept boundary broadens before tightening
   - Recovery mechanism: contrastive evidence and correction-driven relevance adjustment

3. Devices uncertainty growth
   - Input pattern: electronics mixed with object-level confounders
   - Failure mode: entropy increases despite high top-1 accuracy
   - Hypothesis drift observed: near-tie candidate programs persist
   - Recovery mechanism: targeted active-learning questions and recency-sensitive filtering

## 7) Ablation Summary

Source: `evaluation/results/ablation_deltas.csv` (delta vs full).

Interpretation:

- `delta_accuracy < 0` would indicate accuracy loss when a mechanism is removed.
- `delta_brier > 0` indicates worse calibration when ablated.
- `delta_convergence_step > 0` indicates slower convergence when ablated.

### Notable observed deltas

- `ablate_recency_blend` on devices: `Δ convergence = +1.5`, `Δ brier = +0.0069`, `Δ entropy_slope = +0.0221`.
- `ablate_confirmation_memory_floor` shows measurable calibration/entropy/correction changes on liquid, noisy-food, and marine benchmarks.
- `ablate_active_learning_cooldown` increases correction counts on several tasks.
- Several deltas remain near zero on this benchmark size (`seeds=2`), suggesting partial mechanism effects are subtle under current data volume.

## 8) Next Experimental Step

To reduce variance and strengthen claims:

1. Rerun with `--seeds 5` or `--seeds 10`.
2. Expand benchmark lengths for adversarial tasks.
3. Report confidence intervals and significance tests for key deltas.
