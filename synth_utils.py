"""
Utilities for synthetic slow-frequency signal dataset.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


N_POINTS = 10000
WINDOW_SIZE = 500
STEP = 50

ANOMALY_1_RANGE = (0.30, 0.40)  # phase inversion
ANOMALY_2_RANGE = (0.70, 0.80)  # amplitude breakdown


def get_anomaly_ranges(n_points: int) -> List[Tuple[int, int, str]]:
    """
    Return anomaly index ranges for a synthetic signal.

    Parameters
    ----------
    n_points : int
        Total signal length.

    Returns
    -------
    ranges : list of tuple
        Each tuple is (start_idx, end_idx, anomaly_type).
    """
    idx1 = int(ANOMALY_1_RANGE[0] * n_points)
    idx2 = int(ANOMALY_1_RANGE[1] * n_points)
    idx3 = int(ANOMALY_2_RANGE[0] * n_points)
    idx4 = int(ANOMALY_2_RANGE[1] * n_points)
    return [
        (idx1, idx2, "phase_inversion"),
        (idx3, idx4, "amplitude_breakdown"),
    ]


def generate_complex_signal(
    n_points: int,
    mode: str = "normal",
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic signal with slow dynamics.

    Parameters
    ----------
    n_points : int
        Number of samples.
    mode : {'normal', 'anomalous'}
        Signal generation mode.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    signal : np.ndarray
        Generated 1D signal.
    """
    rng = np.random.default_rng(random_state)
    t = np.arange(n_points)

    f_t = 0.02 + 0.005 * np.sin(0.002 * t)
    a_t = 1.0 + 0.3 * np.cos(0.001 * t)

    noise = rng.normal(0.0, 0.03, n_points)

    if mode == "normal":
        signal = a_t * np.sin(2 * np.pi * f_t * t) + noise
    elif mode == "anomalous":
        signal = a_t * np.sin(2 * np.pi * f_t * t)

        for start, end, anomaly_type in get_anomaly_ranges(n_points):
            if anomaly_type == "phase_inversion":
                signal[start:end] = -signal[start:end]
            elif anomaly_type == "amplitude_breakdown":
                signal[start:end] = 0.3 * np.sin(2 * np.pi * f_t[start:end] * t[start:end])

        signal += noise
    else:
        raise ValueError("mode must be 'normal' or 'anomalous'")

    return signal


def create_signal_dataframe(
    signal: np.ndarray,
    anomaly_ranges: Optional[List[Tuple[int, int, str]]] = None,
) -> pd.DataFrame:
    """
    Create sample-level dataframe with normal/anomaly labels.

    Parameters
    ----------
    signal : np.ndarray
        1D signal.
    anomaly_ranges : list of tuple, optional
        Anomaly index ranges as (start, end, anomaly_type).

    Returns
    -------
    df : pandas.DataFrame
        Columns: sample_idx, signal, label, anomaly_type.
        Label convention: 1=normal, 0=anomaly.
    """
    n_points = len(signal)
    labels = np.ones(n_points, dtype=int)
    anomaly_type = np.full(n_points, "normal", dtype=object)

    if anomaly_ranges is not None:
        for start, end, a_type in anomaly_ranges:
            labels[start:end] = 0
            anomaly_type[start:end] = a_type

    df = pd.DataFrame(
        {
            "sample_idx": np.arange(n_points),
            "signal": signal,
            "label": labels,
            "anomaly_type": anomaly_type,
        }
    )
    return df


def create_window_dataset(
    signal: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding-window dataset and window labels.

    Parameters
    ----------
    signal : np.ndarray
        1D signal.
    labels : np.ndarray
        Sample labels (1=normal, 0=anomaly).
    window_size : int
        Window length.
    step : int
        Sliding step.

    Returns
    -------
    X : np.ndarray
        Windowed signal, shape (n_windows, window_size).
    y : np.ndarray
        Window labels, shape (n_windows,). Label=1 only if all samples in window are normal.
    window_starts : np.ndarray
        Start index of each window.
    """
    if len(signal) != len(labels):
        raise ValueError("signal and labels must have same length")

    starts = np.arange(0, len(signal) - window_size, step)
    X = np.array([signal[i : i + window_size] for i in starts])
    y = np.array([int(np.all(labels[i : i + window_size] == 1)) for i in starts])

    return X, y, starts


def build_synthetic_dataset(
    n_points: int = N_POINTS,
    window_size: int = WINDOW_SIZE,
    step: int = STEP,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Build synthetic train/test splits and associated DataFrames.

    Returns
    -------
    data : dict
        Contains raw signals, sample DataFrames, windowed arrays and labels.
    """
    s_train = generate_complex_signal(n_points, mode="normal", random_state=random_seed)
    s_test_normal = generate_complex_signal(n_points, mode="normal", random_state=random_seed + 1)
    s_test_anomalous = generate_complex_signal(n_points, mode="anomalous", random_state=random_seed + 2)

    anomaly_ranges = get_anomaly_ranges(n_points)

    df_train = create_signal_dataframe(s_train, anomaly_ranges=None)
    df_test_normal = create_signal_dataframe(s_test_normal, anomaly_ranges=None)
    df_test_anomalous = create_signal_dataframe(s_test_anomalous, anomaly_ranges=anomaly_ranges)

    X_train, y_train, idx_train = create_window_dataset(
        s_train,
        df_train["label"].values,
        window_size=window_size,
        step=step,
    )
    X_test_normal, y_test_normal, idx_test_normal = create_window_dataset(
        s_test_normal,
        df_test_normal["label"].values,
        window_size=window_size,
        step=step,
    )
    X_test_anomalous, y_test_anomalous, idx_test_anomalous = create_window_dataset(
        s_test_anomalous,
        df_test_anomalous["label"].values,
        window_size=window_size,
        step=step,
    )

    return {
        "signals": {
            "train": s_train,
            "test_normal": s_test_normal,
            "test_anomalous": s_test_anomalous,
        },
        "anomaly_ranges": anomaly_ranges,
        "dataframes": {
            "train": df_train,
            "test_normal": df_test_normal,
            "test_anomalous": df_test_anomalous,
        },
        "windows": {
            "X_train": X_train,
            "y_train": y_train,
            "idx_train": idx_train,
            "X_test_normal": X_test_normal,
            "y_test_normal": y_test_normal,
            "idx_test_normal": idx_test_normal,
            "X_test_anomalous": X_test_anomalous,
            "y_test_anomalous": y_test_anomalous,
            "idx_test_anomalous": idx_test_anomalous,
        },
        "params": {
            "n_points": n_points,
            "window_size": window_size,
            "step": step,
            "random_seed": random_seed,
        },
    }
