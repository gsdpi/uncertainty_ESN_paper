#!/usr/bin/env python3
"""Bar plots comparing our ESN method against the internal-state detectors
(results_<experiment>_internal.xlsx, one row per detector)."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_result_graphs import (
    EXPERIMENTS,
    METRICS,
    SIGNAL_GROUPS,
    SIGNAL_DISPLAY_NAMES,
    _canonicalize_signal_group,
    _extract_metrics_table_from_frame,
    extract_metrics_table,
    find_result_file,
)

SELECTED_METRICS = ["roc_auc", "auprc", "f1_score"]


def table_to_data(table: pd.DataFrame, experiment: str) -> dict:
    """Reduce a metrics table (rows = signals/subjects/single run) to the values to plot."""
    data: dict = {}
    if experiment == "icann":
        data["signals"] = {}
        for group in SIGNAL_GROUPS:
            rows = [i for i in table.index if _canonicalize_signal_group(i) == group]
            data["signals"][group] = {
                m: float(table.loc[rows, m].mean()) if rows else np.nan for m in METRICS
            }
    else:
        data["summary"] = {}
        for m in METRICS:
            s = table[m].dropna().to_numpy(dtype=float)
            data["summary"][m] = {
                "mean": float(s.mean()) if s.size else np.nan,
                "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
            }
    return data


def load_internal_data(result_file: Path, experiment: str) -> dict[str, dict]:
    """One entry per detector found in an internal results file."""
    df = pd.read_excel(result_file)
    det_col = df.columns[0]
    df[det_col] = df[det_col].ffill()  # merged cells from the MultiIndex export

    out = {}
    for detector, group in df.groupby(det_col, sort=False):
        group = group.drop(columns=det_col)
        # icann/imwsha keep a second label column (signal/subject); synth does not
        group = group.set_index(group.columns[0]) if not pd.api.types.is_numeric_dtype(group.iloc[:, 0]) else group
        out[str(detector)] = table_to_data(_extract_metrics_table_from_frame(group), experiment)
    return out


def plot_experiment(experiment: str, data: dict[str, dict], output_dir: Path) -> None:
    methods = list(data)
    if not methods:
        print(f"[WARN] No results for {experiment}.")
        return

    fig, axes = plt.subplots(1, len(SELECTED_METRICS), figsize=(4.2 * len(SELECTED_METRICS), 5.0),
                             constrained_layout=True)
    x = np.arange(len(methods), dtype=float)

    def annotate(axis, bars, values, fontsize):
        for rect, v in zip(bars, values):
            if not np.isnan(v):
                axis.text(rect.get_x() + rect.get_width() / 2, v / 2, f"{v:.3f}", ha="center",
                          va="center", fontsize=fontsize, rotation=90, color="white", weight="bold")

    for axis, metric in zip(np.atleast_1d(axes), SELECTED_METRICS):
        if experiment == "icann":
            width = 0.24
            for k, group in enumerate(SIGNAL_GROUPS):
                values = [data[m]["signals"][group][metric] for m in methods]
                bars = axis.bar(x + (k - 1) * width, values, width=width, alpha=0.9, edgecolor="black",
                                linewidth=0.5, label=SIGNAL_DISPLAY_NAMES[group])
                annotate(axis, bars, values, 7)
        else:
            means = [data[m]["summary"][metric]["mean"] for m in methods]
            stds = [data[m]["summary"][metric]["std"] for m in methods]
            # std is 0 for single-run experiments (synth), so no error bars are drawn there
            bars = axis.bar(x, means, yerr=stds if experiment == "imwsha" else None, capsize=4,
                            alpha=0.9, edgecolor="black", linewidth=0.5)
            annotate(axis, bars, means, 8)

        axis.set_title(METRICS[metric], fontsize=11)
        axis.set_xticks(x)
        axis.set_xticklabels([m.upper() for m in methods])
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", linestyle="--", alpha=0.3)

    if experiment == "icann":
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", ncol=3)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"comparison_internal_{experiment}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Figure saved: {output_path}")
    plt.close(fig)


def main() -> None:
    base_dir = Path(".").resolve()
    output_dir = base_dir / "figures"

    for experiment in EXPERIMENTS:
        data: dict[str, dict] = {}

        # Reference: our SVD+KDE method
        esn_file: Optional[Path] = find_result_file(base_dir, experiment, "esn")
        if esn_file is None:
            print(f"[WARN] Not found: results_{experiment}_esn.xlsx")
        else:
            print(f"[INFO] Loading {esn_file.name}")
            data["esn"] = table_to_data(extract_metrics_table(esn_file), experiment)

        internal_file = base_dir / f"results_{experiment}_internal.xlsx"
        if not internal_file.exists():
            print(f"[WARN] Not found: {internal_file.name}")
        else:
            print(f"[INFO] Loading {internal_file.name}")
            data.update(load_internal_data(internal_file, experiment))

        plot_experiment(experiment, data, output_dir)


if __name__ == "__main__":
    main()
