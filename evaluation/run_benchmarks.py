import argparse
import json
import os
import random
import sys
from collections import defaultdict

os.environ.setdefault("TM_PERCEPTION_DEBUG", "0")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Thinkingmachiene import ThinkingMachine
try:
    from evaluation.experiment_manager import prepare_managed_run
    from evaluation.reporting import (
        aggregate_summary,
        build_ablation_deltas,
        convergence_step,
        ensure_dir,
        linear_slope,
        plot_ablation_deltas,
        plot_metrics,
        write_csv,
    )
except Exception:
    from experiment_manager import prepare_managed_run
    from reporting import (  # type: ignore
        aggregate_summary,
        build_ablation_deltas,
        convergence_step,
        ensure_dir,
        linear_slope,
        plot_ablation_deltas,
        plot_metrics,
        write_csv,
    )


ABLATION_VARIANTS = {
    "full": {},
    "ablate_recency_blend": {"recency_blend": False},
    "ablate_stale_feature_demotion": {"stale_feature_demotion": False},
    "ablate_active_learning_cooldown": {"active_learning_cooldown": False},
    "ablate_confirmation_memory_floor": {"confirmation_memory_floor": False},
    "ablate_anchor_override": {"anchor_override": False},
}


def _load_config(path):
    if not path:
        return {}
    ext = os.path.splitext(path)[1].lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("YAML config requested but PyYAML is not installed. Install with: pip install pyyaml") from exc
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
        return data if isinstance(data, dict) else {}


def run(benchmarks_path, out_dir, seeds, run_ablations=True):
    ensure_dir(out_dir)

    with open(benchmarks_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    benchmarks = payload.get("benchmarks", [])
    if not benchmarks:
        raise ValueError("No benchmarks found in input JSON.")

    per_step_rows = []
    per_run_rows = []
    by_benchmark_entropy = defaultdict(list)
    variants = ABLATION_VARIANTS if run_ablations else {"full": {}}

    for benchmark in benchmarks:
        name = benchmark["name"]
        examples = list(benchmark.get("examples", []))

        for ablation_name, ablation_flags in variants.items():
            benchmark_label = f"{name} ({ablation_name})"

            for seed in range(seeds):
                rng = random.Random(seed)
                shuffled = list(examples)
                rng.shuffle(shuffled)

                tm = ThinkingMachine()
                tm.set_ablation_flags(ablation_flags)
                step_results = []

                for idx, ex in enumerate(shuffled, start=1):
                    step = tm.process_labeled_example(
                        item=ex["item"],
                        truth=bool(ex["label"]),
                        features_override=ex.get("features", {}),
                        enable_active_learning=True,
                        auto_feedback=True,
                    )

                    probability = float(step.get("probability", 0.5))
                    truth = bool(step.get("truth", False))
                    prediction = step.get("prediction")
                    if prediction is None:
                        prediction = bool(probability >= 0.5)

                    row = {
                        "benchmark": name,
                        "ablation": ablation_name,
                        "benchmark_label": benchmark_label,
                        "seed": seed,
                        "step": idx,
                        "item": ex["item"],
                        "truth": int(truth),
                        "prediction": int(bool(prediction)),
                        "probability": probability,
                        "brier": (probability - (1.0 if truth else 0.0)) ** 2,
                        "entropy": float(step.get("entropy", 0.0)),
                        "top_theory": step.get("top_theory") or "",
                        "top_weight": float(step.get("top_weight", 0.0)),
                        "correction_asked": int(bool(step.get("correction_asked", False))),
                        "correction_applied": int(bool(step.get("correction_applied", False))),
                        "correction_feature": step.get("correction_feature") or "",
                        "correction_action": step.get("correction_action") or "",
                    }
                    per_step_rows.append(row)
                    step_results.append(row)

                y_true = [r["truth"] for r in step_results]
                y_pred = [r["prediction"] for r in step_results]
                probs = [r["probability"] for r in step_results]
                entropies = [r["entropy"] for r in step_results]
                top_theories = [r["top_theory"] for r in step_results]

                accuracy = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
                brier = sum((p - y) ** 2 for p, y in zip(probs, y_true)) / max(1, len(y_true))
                convergence = convergence_step(top_theories, window=3)
                corrections = sum(r["correction_asked"] for r in step_results)
                entropy_slope = linear_slope(entropies)

                by_benchmark_entropy[benchmark_label].append(entropies)

                per_run_rows.append({
                    "benchmark": name,
                    "ablation": ablation_name,
                    "benchmark_label": benchmark_label,
                    "seed": seed,
                    "n_examples": len(step_results),
                    "accuracy": accuracy,
                    "brier": brier,
                    "convergence_step": convergence if convergence is not None else "",
                    "entropy_start": entropies[0] if entropies else 0.0,
                    "entropy_end": entropies[-1] if entropies else 0.0,
                    "entropy_slope": entropy_slope,
                    "correction_count": corrections,
                })

    step_fields = [
        "benchmark", "ablation", "benchmark_label", "seed", "step", "item", "truth", "prediction", "probability", "brier",
        "entropy", "top_theory", "top_weight", "correction_asked", "correction_applied",
        "correction_feature", "correction_action"
    ]
    run_fields = [
        "benchmark", "ablation", "benchmark_label", "seed", "n_examples", "accuracy", "brier", "convergence_step",
        "entropy_start", "entropy_end", "entropy_slope", "correction_count"
    ]

    write_csv(os.path.join(out_dir, "per_step_metrics.csv"), per_step_rows, step_fields)
    write_csv(os.path.join(out_dir, "per_run_metrics.csv"), per_run_rows, run_fields)

    summary = aggregate_summary(per_run_rows)

    summary_fields = [
        "benchmark", "ablation",
        "accuracy_mean", "accuracy_std", "accuracy_ci95_low", "accuracy_ci95_high",
        "brier_mean", "brier_std", "brier_ci95_low", "brier_ci95_high",
        "convergence_step_mean", "convergence_step_std", "convergence_step_ci95_low", "convergence_step_ci95_high",
        "entropy_slope_mean", "entropy_slope_std", "entropy_slope_ci95_low", "entropy_slope_ci95_high",
        "correction_count_mean", "correction_count_std", "correction_count_ci95_low", "correction_count_ci95_high",
        "runs"
    ]
    write_csv(os.path.join(out_dir, "summary_metrics.csv"), summary, summary_fields)

    ablation_delta_rows = build_ablation_deltas(summary, variants)

    ablation_path = os.path.join(out_dir, "ablation_deltas.csv")
    if run_ablations:
        write_csv(
            ablation_path,
            ablation_delta_rows,
            [
                "benchmark", "ablation", "delta_accuracy", "delta_brier", "delta_convergence_step",
                "delta_entropy_slope", "delta_correction_count"
            ]
        )
    elif os.path.exists(ablation_path):
        os.remove(ablation_path)

    plot_metrics(out_dir, per_run_rows, by_benchmark_entropy)
    if run_ablations:
        plot_ablation_deltas(out_dir, ablation_delta_rows)

    print(f"[DONE] Wrote metrics and plots to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run reproducible benchmark harness.")
    parser.add_argument("--benchmarks", default=os.path.join("evaluation", "benchmarks.json"))
    parser.add_argument("--out", default=os.path.join("evaluation", "results"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-ablations", action="store_true", help="Run only full model without ablation variants")
    parser.add_argument("--config", default="", help="Path to JSON/YAML experiment config")
    parser.add_argument("--managed-run", action="store_true", help="Create timestamped run folder and manifest under --out")
    parser.add_argument("--run-name", default="", help="Optional managed run name suffix")
    args = parser.parse_args()

    config = _load_config(args.config)
    benchmarks = config.get("benchmarks", args.benchmarks)
    out = config.get("out", args.out)
    seeds = int(config.get("seeds", args.seeds))
    run_ablations = bool(config.get("run_ablations", (not args.no_ablations)))
    if args.no_ablations:
        run_ablations = False

    if args.managed_run:
        managed_out, manifest_path = prepare_managed_run(
            base_out_dir=out,
            benchmarks_path=benchmarks,
            seeds=seeds,
            run_ablations=run_ablations,
            config_path=args.config,
            argv=sys.argv,
            run_name=args.run_name,
        )
        out = managed_out
        print(f"[INFO] Managed run manifest written to: {manifest_path}")

    run(benchmarks, out, seeds, run_ablations=run_ablations)


if __name__ == "__main__":
    main()
