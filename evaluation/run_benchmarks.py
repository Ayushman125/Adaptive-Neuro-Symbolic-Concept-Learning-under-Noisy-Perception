import argparse
import csv
import json
import math
import os
import random
import sys
import statistics
from collections import defaultdict

os.environ.setdefault("TM_PERCEPTION_DEBUG", "0")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Thinkingmachiene import ThinkingMachine


ABLATION_VARIANTS = {
    "full": {},
    "ablate_recency_blend": {"recency_blend": False},
    "ablate_stale_feature_demotion": {"stale_feature_demotion": False},
    "ablate_active_learning_cooldown": {"active_learning_cooldown": False},
    "ablate_confirmation_memory_floor": {"confirmation_memory_floor": False},
    "ablate_anchor_override": {"anchor_override": False},
}


def _linear_slope(values):
    if not values or len(values) < 2:
        return 0.0
    n = len(values)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den <= 1e-12:
        return 0.0
    return num / den


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


def _convergence_step(top_theories, window=3):
    if len(top_theories) < window:
        return None
    for i in range(0, len(top_theories) - window + 1):
        segment = top_theories[i:i + window]
        head = segment[0]
        if head and all(t == head for t in segment):
            return i + window
    return None


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metrics(out_dir, run_rows, by_benchmark_entropy):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not installed. Skipping plot generation.")
        return

    metrics = [
        ("accuracy", "Accuracy", "accuracy_by_benchmark.png"),
        ("brier", "Brier Score", "brier_by_benchmark.png"),
        ("convergence_step", "Convergence Step", "convergence_by_benchmark.png"),
        ("correction_count", "Correction Count", "corrections_by_benchmark.png"),
        ("entropy_slope", "Entropy Slope", "entropy_slope_by_benchmark.png"),
    ]

    grouped = defaultdict(list)
    for row in run_rows:
        if row.get("ablation") != "full":
            continue
        grouped[row["benchmark"]].append(row)

    for key, title, fname in metrics:
        names = []
        means = []
        stds = []
        for bench_label, rows in grouped.items():
            if key == "convergence_step":
                vals = [float(r[key]) if r[key] not in (None, "") else 0.0 for r in rows]
            else:
                vals = [float(r[key]) for r in rows if r[key] is not None and r[key] != ""]
            if not vals:
                continue
            names.append(bench_label)
            means.append(statistics.mean(vals))
            stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)

        if not names:
            continue

        plt.figure(figsize=(10, 5))
        plt.bar(names, means, yerr=stds, capsize=4)
        plt.title(title)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=140)
        plt.close()

    plt.figure(figsize=(10, 5))
    for bench_label, trajectories in by_benchmark_entropy.items():
        if not bench_label.endswith("(full)"):
            continue
        bench = bench_label.rsplit(" (", 1)[0]
        max_len = max(len(t) for t in trajectories)
        averaged = []
        for idx in range(max_len):
            vals = [t[idx] for t in trajectories if idx < len(t)]
            averaged.append(sum(vals) / len(vals))
        plt.plot(range(1, len(averaged) + 1), averaged, label=bench)
    plt.title("Entropy Trend (mean across seeds)")
    plt.xlabel("Example index")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_trend_by_benchmark.png"), dpi=140)
    plt.close()


def _plot_ablation_deltas(out_dir, ablation_delta_rows):
    if not ablation_delta_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    grouped = defaultdict(list)
    for row in ablation_delta_rows:
        grouped[row["ablation"]].append(row)

    metric_specs = [
        ("delta_accuracy", "Ablation Δ Accuracy", "ablation_delta_accuracy.png"),
        ("delta_brier", "Ablation Δ Brier", "ablation_delta_brier.png"),
        ("delta_convergence_step", "Ablation Δ Convergence", "ablation_delta_convergence.png"),
    ]

    for key, title, filename in metric_specs:
        labels = []
        means = []
        stds = []
        for ablation_name, rows in grouped.items():
            vals = []
            for row in rows:
                value = row.get(key, "")
                if value in ("", None):
                    continue
                vals.append(float(value))
            if not vals:
                continue
            labels.append(ablation_name)
            means.append(statistics.mean(vals))
            stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)

        if not labels:
            continue

        plt.figure(figsize=(10, 5))
        plt.bar(labels, means, yerr=stds, capsize=4)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.title(title)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=140)
        plt.close()


def run(benchmarks_path, out_dir, seeds, run_ablations=True):
    _ensure_dir(out_dir)

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
                convergence = _convergence_step(top_theories, window=3)
                corrections = sum(r["correction_asked"] for r in step_results)
                entropy_slope = _linear_slope(entropies)

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

    _write_csv(os.path.join(out_dir, "per_step_metrics.csv"), per_step_rows, step_fields)
    _write_csv(os.path.join(out_dir, "per_run_metrics.csv"), per_run_rows, run_fields)

    summary = []
    by_bench = defaultdict(list)
    for row in per_run_rows:
        by_bench[(row["benchmark"], row["ablation"])].append(row)

    for (bench, ablation_name), rows in by_bench.items():
        def agg(key, numeric=True):
            values = []
            for r in rows:
                value = r[key]
                if value == "":
                    continue
                values.append(float(value) if numeric else value)
            if not values:
                return "", ""
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            return mean, std

        acc_m, acc_s = agg("accuracy")
        br_m, br_s = agg("brier")
        conv_m, conv_s = agg("convergence_step")
        es_m, es_s = agg("entropy_slope")
        cor_m, cor_s = agg("correction_count")

        summary.append({
            "benchmark": bench,
            "ablation": ablation_name,
            "accuracy_mean": acc_m,
            "accuracy_std": acc_s,
            "brier_mean": br_m,
            "brier_std": br_s,
            "convergence_step_mean": conv_m,
            "convergence_step_std": conv_s,
            "entropy_slope_mean": es_m,
            "entropy_slope_std": es_s,
            "correction_count_mean": cor_m,
            "correction_count_std": cor_s,
            "runs": len(rows),
        })

    summary_fields = [
        "benchmark", "ablation", "accuracy_mean", "accuracy_std", "brier_mean", "brier_std",
        "convergence_step_mean", "convergence_step_std", "entropy_slope_mean",
        "entropy_slope_std", "correction_count_mean", "correction_count_std", "runs"
    ]
    _write_csv(os.path.join(out_dir, "summary_metrics.csv"), summary, summary_fields)

    ablation_delta_rows = []
    summary_index = {(r["benchmark"], r["ablation"]): r for r in summary}
    for benchmark in sorted({r["benchmark"] for r in summary}):
        full = summary_index.get((benchmark, "full"))
        if not full:
            continue
        for ablation_name in variants.keys():
            if ablation_name == "full":
                continue
            variant = summary_index.get((benchmark, ablation_name))
            if not variant:
                continue
            ablation_delta_rows.append({
                "benchmark": benchmark,
                "ablation": ablation_name,
                "delta_accuracy": float(variant["accuracy_mean"]) - float(full["accuracy_mean"]),
                "delta_brier": float(variant["brier_mean"]) - float(full["brier_mean"]),
                "delta_convergence_step": (
                    float(variant["convergence_step_mean"]) - float(full["convergence_step_mean"])
                    if variant["convergence_step_mean"] != "" and full["convergence_step_mean"] != "" else ""
                ),
                "delta_entropy_slope": float(variant["entropy_slope_mean"]) - float(full["entropy_slope_mean"]),
                "delta_correction_count": float(variant["correction_count_mean"]) - float(full["correction_count_mean"]),
            })

    ablation_path = os.path.join(out_dir, "ablation_deltas.csv")
    if run_ablations:
        _write_csv(
            ablation_path,
            ablation_delta_rows,
            [
                "benchmark", "ablation", "delta_accuracy", "delta_brier", "delta_convergence_step",
                "delta_entropy_slope", "delta_correction_count"
            ]
        )
    elif os.path.exists(ablation_path):
        os.remove(ablation_path)

    _plot_metrics(out_dir, per_run_rows, by_benchmark_entropy)
    if run_ablations:
        _plot_ablation_deltas(out_dir, ablation_delta_rows)

    print(f"[DONE] Wrote metrics and plots to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run reproducible benchmark harness.")
    parser.add_argument("--benchmarks", default=os.path.join("evaluation", "benchmarks.json"))
    parser.add_argument("--out", default=os.path.join("evaluation", "results"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-ablations", action="store_true", help="Run only full model without ablation variants")
    parser.add_argument("--config", default="", help="Path to JSON/YAML experiment config")
    args = parser.parse_args()

    config = _load_config(args.config)
    benchmarks = config.get("benchmarks", args.benchmarks)
    out = config.get("out", args.out)
    seeds = int(config.get("seeds", args.seeds))
    run_ablations = bool(config.get("run_ablations", (not args.no_ablations)))
    if args.no_ablations:
        run_ablations = False

    run(benchmarks, out, seeds, run_ablations=run_ablations)


if __name__ == "__main__":
    main()
