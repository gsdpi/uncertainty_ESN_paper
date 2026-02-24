#!/usr/bin/env python3

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENTS = ("icann", "imwsha", "synth")
METHODS = ("esn", "knn", "minirocket", "pca")

METRICS = {
    "roc_auc": "ROC AUC",
    "auprc": "AUPRC",
    "recall_at_1pct_fpr": "Recall@FPR<=1%",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1_score": "F1-score",
}

METRIC_ALIASES = {
    "roc_auc": ["roc_auc", "rocauc", "aucroc", "roc_auc_score"],
    "auprc": ["auprc", "aucpr", "pr_auc", "average_precision"],
    "recall_at_1pct_fpr": [
        "recall_at_1pct_fpr",
        "recall_1pct_fpr",
        "recallfpr1",
        "recallfpr001",
        "recallat1pctfpr",
    ],
    "sensitivity": ["sensitivity", "recall", "tpr"],
    "specificity": ["specificity", "tnr"],
    "precision": ["precision", "ppv"],
    "f1_score": ["f1_score", "f1", "f1score", "f1-score"],
}

SIGNAL_GROUPS = ("ax", "ay", "combined")
SIGNAL_DISPLAY_NAMES = {
    "ax": r"$a_x$",
    "ay": r"$a_y$",
    "combined": r"$a_x + a_y$",
}


def _normalize_metric_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _build_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical_name, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            lookup[_normalize_metric_name(alias)] = canonical_name
    return lookup


ALIAS_LOOKUP = _build_alias_lookup()


def _canonicalize(name: str) -> Optional[str]:
    return ALIAS_LOOKUP.get(_normalize_metric_name(name))


def _canonicalize_signal_group(name: str) -> Optional[str]:
    normalized = _normalize_metric_name(name)

    has_ax = "ax" in normalized
    has_ay = "ay" in normalized

    if has_ax and has_ay:
        return "combined"

    if normalized in {"allchannels", "allchannel", "all", "combined", "both", "axay"}:
        return "combined"

    if has_ax:
        return "ax"

    if has_ay:
        return "ay"

    return None


def find_result_file(base_dir: Path, experiment: str, method: str) -> Optional[Path]:
    preferred = base_dir / f"results_{experiment}_{method}.xlsx"
    if preferred.exists():
        return preferred

    if method == "esn":
        legacy = base_dir / f"results_{experiment}.xlsx"
        if legacy.exists():
            return legacy

    return None


def _extract_metrics_table_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(METRICS.keys()))

    canonical_columns = {
        column: _canonicalize(column)
        for column in df.columns
        if _canonicalize(column) in METRICS
    }
    canonical_index = {
        idx: _canonicalize(idx)
        for idx in df.index
        if _canonicalize(idx) in METRICS
    }

    if len(canonical_columns) >= len(canonical_index):
        metrics_df = pd.DataFrame(index=df.index)
        for column, canonical_metric in canonical_columns.items():
            metrics_df[canonical_metric] = pd.to_numeric(df[column], errors="coerce")
    else:
        metrics_df = pd.DataFrame(index=df.columns)
        for idx, canonical_metric in canonical_index.items():
            row_values = pd.to_numeric(df.loc[idx], errors="coerce")
            metrics_df[canonical_metric] = row_values.values

    for metric_name in METRICS:
        if metric_name not in metrics_df.columns:
            metrics_df[metric_name] = np.nan

    return metrics_df[list(METRICS.keys())]


def extract_metrics_table(result_file: Path) -> pd.DataFrame:
    df_raw = pd.read_excel(result_file)

    candidates: list[pd.DataFrame] = [df_raw]
    if len(df_raw.columns) > 0:
        candidates.append(df_raw.set_index(df_raw.columns[0]))

    best_table: Optional[pd.DataFrame] = None
    best_score: tuple[int, int] = (-1, -1)

    for candidate in candidates:
        table = _extract_metrics_table_from_frame(candidate)
        non_na_score = int(table.notna().sum().sum())
        signal_index_score = sum(
            1 for idx in table.index if _canonicalize_signal_group(idx) is not None
        )
        score = (non_na_score, signal_index_score)

        if score > best_score:
            best_score = score
            best_table = table

    if best_table is None:
        return pd.DataFrame(columns=list(METRICS.keys()))

    return best_table


def extract_metric_samples(result_file: Path) -> dict[str, np.ndarray]:
    metrics_df = extract_metrics_table(result_file)
    metric_samples = {metric: np.array([], dtype=float) for metric in METRICS}

    for metric_name in METRICS:
        samples = pd.to_numeric(metrics_df[metric_name], errors="coerce").dropna().to_numpy(dtype=float)
        metric_samples[metric_name] = samples

    return metric_samples


def extract_single_metrics(result_file: Path) -> dict[str, float]:
    metric_samples = extract_metric_samples(result_file)
    single_metrics: dict[str, float] = {}
    for metric_name in METRICS:
        samples = metric_samples[metric_name]
        single_metrics[metric_name] = float(samples[0]) if samples.size > 0 else np.nan
    return single_metrics


def extract_icann_signal_metrics(result_file: Path) -> dict[str, dict[str, float]]:
    metrics_df = extract_metrics_table(result_file)

    grouped_values: dict[str, dict[str, list[float]]] = {
        group: {metric: [] for metric in METRICS}
        for group in SIGNAL_GROUPS
    }

    for signal_name in metrics_df.index:
        signal_group = _canonicalize_signal_group(signal_name)
        if signal_group is None:
            continue

        for metric_name in METRICS:
            value = pd.to_numeric(metrics_df.loc[signal_name, metric_name], errors="coerce")
            if pd.notna(value):
                grouped_values[signal_group][metric_name].append(float(value))

    grouped_metrics: dict[str, dict[str, float]] = {}
    for signal_group in SIGNAL_GROUPS:
        grouped_metrics[signal_group] = {}
        for metric_name in METRICS:
            values = grouped_values[signal_group][metric_name]
            grouped_metrics[signal_group][metric_name] = float(np.mean(values)) if values else np.nan

    return grouped_metrics


def summarize_method_metrics(result_file: Path) -> dict[str, dict[str, float]]:
    metric_samples = extract_metric_samples(result_file)
    summary: dict[str, dict[str, float]] = {}
    for metric_name in METRICS:
        samples = metric_samples[metric_name]
        if samples.size == 0:
            summary[metric_name] = {"mean": np.nan, "std": np.nan}
            continue

        summary[metric_name] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0,
        }

    return summary


def plot_icann_experiment(
    experiment_data: dict[str, dict[str, dict[str, dict[str, float]]]],
    output_dir: Path,
) -> None:
    available_methods = [method for method in METHODS if method in experiment_data]
    if not available_methods:
        print("[WARN] No results for icann.")
        return

    selected_metrics = ["roc_auc", "auprc", "f1_score"]
    n_metrics = len(selected_metrics)
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 5.0 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    x_positions = np.arange(len(available_methods), dtype=float)
    bar_width = 0.24
    signal_offsets = {
        "ax": -bar_width,
        "ay": 0.0,
        "combined": bar_width,
    }

    for axis, metric_name in zip(axes, selected_metrics):
        for signal_group in SIGNAL_GROUPS:
            values = [
                experiment_data[method]["signals"][signal_group][metric_name]
                for method in available_methods
            ]
            bars = axis.bar(
                x_positions + signal_offsets[signal_group],
                values,
                width=bar_width,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.5,
                label=SIGNAL_DISPLAY_NAMES[signal_group],
            )

            for rect, value in zip(bars, values):
                if np.isnan(value):
                    continue
                axis.text(
                    rect.get_x() + rect.get_width() / 2,
                    value / 2,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    rotation=90,
                    color="white",
                    weight="bold",
                )

        axis.set_title(METRICS[metric_name], fontsize=11)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([method.upper() for method in available_methods], rotation=0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", linestyle="--", alpha=0.3)

    for axis in axes[n_metrics:]:
        axis.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=3)
    #fig.suptitle("Method comparison - ICANN", fontsize=14, x=0.5)
    fig.suptitle("  ", fontsize=14, x=0.5) # Generate extra space for legend

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison_icann.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Figure saved: {output_path}")
    plt.close(fig)


def plot_imwsha_experiment(
    experiment_data: dict[str, dict[str, dict[str, dict[str, float]]]],
    output_dir: Path,
) -> None:
    available_methods = [method for method in METHODS if method in experiment_data]
    if not available_methods:
        print("[WARN] No results for imwsha.")
        return

    selected_metrics = ["roc_auc", "auprc", "f1_score"]

    n_metrics = len(selected_metrics)
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for axis, metric_name in zip(axes, selected_metrics):
        means = [experiment_data[m]["summary"][metric_name]["mean"] for m in available_methods]
        stds = [experiment_data[m]["summary"][metric_name]["std"] for m in available_methods]
        x_positions = np.arange(len(available_methods))

        bars = axis.bar(
            x_positions,
            means,
            yerr=stds,
            capsize=4,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )

        axis.set_title(METRICS[metric_name], fontsize=11)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([m.upper() for m in available_methods], rotation=0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", linestyle="--", alpha=0.3)

        for rect, value in zip(bars, means):
            if np.isnan(value):
                continue
            axis.text(
                rect.get_x() + rect.get_width() / 2,
                value / 2,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                rotation=90,
                color="white",
                weight="bold",
            )

    for axis in axes[n_metrics:]:
        axis.set_visible(False)

    #fig.suptitle("Method comparison - IMWSHA", fontsize=14, ha="center", x=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison_imwsha.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Figure saved: {output_path}")
    plt.close(fig)


def plot_synth_experiment(
    experiment_data: dict[str, dict[str, dict[str, dict[str, float]]]],
    output_dir: Path,
) -> None:
    available_methods = [method for method in METHODS if method in experiment_data]
    if not available_methods:
        print("[WARN] No results for synth.")
        return

    selected_metrics = ["roc_auc", "auprc", "f1_score"]

    n_metrics = len(selected_metrics)
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for axis, metric_name in zip(axes, selected_metrics):
        values = [experiment_data[m]["single"][metric_name] for m in available_methods]
        x_positions = np.arange(len(available_methods))

        bars = axis.bar(
            x_positions,
            values,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )

        axis.set_title(METRICS[metric_name], fontsize=11)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([m.upper() for m in available_methods], rotation=0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", linestyle="--", alpha=0.3)

        for rect, value in zip(bars, values):
            if np.isnan(value):
                continue
            axis.text(
                rect.get_x() + rect.get_width() / 2,
                value / 2,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                rotation=90,
                color="white",
                weight="bold",
            )

    for axis in axes[n_metrics:]:
        axis.set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison_synth.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Figure saved: {output_path}")
    plt.close(fig)


def main() -> None:
    base_dir = Path(".").resolve()
    output_dir = (base_dir / "figures").resolve()

    all_data: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {
        experiment: {} for experiment in EXPERIMENTS
    }

    for experiment in EXPERIMENTS:
        for method in METHODS:
            result_file = find_result_file(base_dir, experiment, method)
            if result_file is None:
                print(f"[WARN] Not found: results_{experiment}_{method}.xlsx")
                continue

            print(f"[INFO] Loading {result_file.name}")
            method_data: dict[str, dict[str, dict[str, float]]] = {
                "summary": summarize_method_metrics(result_file)
            }

            if experiment == "icann":
                method_data["signals"] = extract_icann_signal_metrics(result_file)
            elif experiment == "synth":
                method_data["single"] = extract_single_metrics(result_file)

            all_data[experiment][method] = method_data

    plot_icann_experiment(all_data["icann"], output_dir)
    plot_imwsha_experiment(all_data["imwsha"], output_dir)
    plot_synth_experiment(all_data["synth"], output_dir)


if __name__ == "__main__":
    main()