import csv
import math
import os
import statistics
from collections import defaultdict


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def linear_slope(values):
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


def convergence_step(top_theories, window=3):
    if len(top_theories) < window:
        return None
    for index in range(0, len(top_theories) - window + 1):
        segment = top_theories[index:index + window]
        head = segment[0]
        if head and all(theory == head for theory in segment):
            return index + window
    return None


def mean_std_ci(values):
    if not values:
        return "", "", "", ""
    if len(values) == 1:
        value = float(values[0])
        return value, 0.0, value, value

    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    std_error = statistics.stdev(values) / math.sqrt(len(values))
    margin = 1.96 * std_error
    return mean, std, mean - margin, mean + margin


def aggregate_summary(per_run_rows):
    summary = []
    grouped = defaultdict(list)
    for row in per_run_rows:
        grouped[(row["benchmark"], row["ablation"])].append(row)

    metric_map = [
        ("accuracy", "accuracy"),
        ("brier", "brier"),
        ("convergence_step", "convergence_step"),
        ("entropy_slope", "entropy_slope"),
        ("correction_count", "correction_count"),
    ]

    for (benchmark, ablation_name), rows in grouped.items():
        item = {
            "benchmark": benchmark,
            "ablation": ablation_name,
            "runs": len(rows),
        }

        for source_key, output_prefix in metric_map:
            values = []
            for row in rows:
                value = row[source_key]
                if value in ("", None):
                    continue
                values.append(float(value))

            mean, std, ci_low, ci_high = mean_std_ci(values)
            item[f"{output_prefix}_mean"] = mean
            item[f"{output_prefix}_std"] = std
            item[f"{output_prefix}_ci95_low"] = ci_low
            item[f"{output_prefix}_ci95_high"] = ci_high

        summary.append(item)

    return summary


def build_ablation_deltas(summary_rows, variants):
    rows = []
    summary_index = {(row["benchmark"], row["ablation"]): row for row in summary_rows}

    for benchmark in sorted({row["benchmark"] for row in summary_rows}):
        full = summary_index.get((benchmark, "full"))
        if not full:
            continue

        for ablation_name in variants.keys():
            if ablation_name == "full":
                continue

            variant = summary_index.get((benchmark, ablation_name))
            if not variant:
                continue

            full_conv = full.get("convergence_step_mean", "")
            variant_conv = variant.get("convergence_step_mean", "")
            convergence_delta = ""
            if full_conv not in ("", None) and variant_conv not in ("", None):
                convergence_delta = float(variant_conv) - float(full_conv)

            rows.append({
                "benchmark": benchmark,
                "ablation": ablation_name,
                "delta_accuracy": float(variant["accuracy_mean"]) - float(full["accuracy_mean"]),
                "delta_brier": float(variant["brier_mean"]) - float(full["brier_mean"]),
                "delta_convergence_step": convergence_delta,
                "delta_entropy_slope": float(variant["entropy_slope_mean"]) - float(full["entropy_slope_mean"]),
                "delta_correction_count": float(variant["correction_count_mean"]) - float(full["correction_count_mean"]),
            })

    return rows


def plot_metrics(out_dir, run_rows, by_benchmark_entropy):
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

    for key, title, filename in metrics:
        names = []
        means = []
        stds = []

        for benchmark, rows in grouped.items():
            if key == "convergence_step":
                values = [float(row[key]) if row[key] not in (None, "") else 0.0 for row in rows]
            else:
                values = [float(row[key]) for row in rows if row[key] not in (None, "")]

            if not values:
                continue

            names.append(benchmark)
            means.append(statistics.mean(values))
            stds.append(statistics.pstdev(values) if len(values) > 1 else 0.0)

        if not names:
            continue

        plt.figure(figsize=(10, 5))
        plt.bar(names, means, yerr=stds, capsize=4)
        plt.title(title)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=140)
        plt.close()

    plt.figure(figsize=(10, 5))
    for benchmark_label, trajectories in by_benchmark_entropy.items():
        if not benchmark_label.endswith("(full)"):
            continue

        benchmark = benchmark_label.rsplit(" (", 1)[0]
        max_len = max(len(traj) for traj in trajectories)
        averaged = []
        for idx in range(max_len):
            values = [traj[idx] for traj in trajectories if idx < len(traj)]
            averaged.append(sum(values) / len(values))

        plt.plot(range(1, len(averaged) + 1), averaged, label=benchmark)

    plt.title("Entropy Trend (mean across seeds)")
    plt.xlabel("Example index")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_trend_by_benchmark.png"), dpi=140)
    plt.close()


def plot_ablation_deltas(out_dir, ablation_delta_rows):
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
            values = []
            for row in rows:
                value = row.get(key, "")
                if value in ("", None):
                    continue
                values.append(float(value))

            if not values:
                continue

            labels.append(ablation_name)
            means.append(statistics.mean(values))
            stds.append(statistics.pstdev(values) if len(values) > 1 else 0.0)

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