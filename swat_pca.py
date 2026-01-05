##################################################################
# PCA-based anomaly detection for SWaT dataset
##################################################################

import time
from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

from esn_uncertainty import calc_metrics
from swat_utils import (
    WINDOW_LENGTH,
    STRIDE,
    SAMPLING_PERIOD,
    load_swat_data,
    get_features,
    create_label_mask,
    prepare_train_df,
)

plt.rcParams.update({"font.size": 18})

##################################################################
# GLOBAL PARAMETERS
##################################################################

N_COMPONENTS: Union[float, int] = 0.95
SCORE_TYPE = "reconstruction_error"

##################################################################
# HELPERS
##################################################################

def window_signal(X: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    if X.ndim != 2:
        raise ValueError("X must have shape (T, C)")
    T, _ = X.shape
    if T < window_size:
        raise ValueError("window_size larger than signal length")

    windows = sliding_window_view(X, window_size, axis=0)
    return windows[::stride]


def vectorize_windows(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(windows.shape[0], -1)


##################################################################
# MODEL
##################################################################

class PCAAnomalyDetector:
    """PCA-based anomaly detector for multichannel time series."""

    def __init__(
        self,
        n_components: Union[float, int] = 0.95,
        window_size: int = 50,
        stride: int = 1,
        score_type: str = "reconstruction_error",
    ):
        self.n_components = n_components
        self.window_size = window_size
        self.stride = stride
        self.score_type = score_type
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X_train: np.ndarray):
        print(
            f"Windowing training data (window size {self.window_size}, stride {self.stride})..."
        )
        windows = window_signal(X_train, self.window_size, self.stride)
        print(f"Generated {windows.shape[0]} training windows")
        X_vec = vectorize_windows(windows)
        self.pca.fit(X_vec)
        print("PCA model fitted.")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        windows = window_signal(X, self.window_size, self.stride)
        X_vec = vectorize_windows(windows)
        X_proj = self.pca.transform(X_vec)
        X_rec = self.pca.inverse_transform(X_proj)
        if self.score_type == "reconstruction_error":
            scores = np.linalg.norm(X_vec - X_rec, axis=1)
        else:
            raise ValueError(f"Unsupported score_type: {self.score_type}")
        return scores


##################################################################
# PIPELINE
##################################################################

def process_swat_pca(df: pd.DataFrame, features: List[str], show_roc_plot: bool = False):
    print(f"\n{'='*70}")
    print("PROCESSING: SWaT (PCA)")
    print(f"{'='*70}\n")

    df_train = prepare_train_df(df, attack_is_one=True, trim=0)

    X_train = df_train[features].values
    X_full = df[features].values

    print(f"Training data shape: {X_train.shape}")
    print(f"Full data shape: {X_full.shape}")

    print("\nTraining PCA anomaly detector...")
    start_time = time.time()
    detector = PCAAnomalyDetector(
        n_components=N_COMPONENTS,
        window_size=WINDOW_LENGTH,
        stride=STRIDE,
        score_type=SCORE_TYPE,
    )
    detector.fit(X_train)
    pca_time = time.time() - start_time
    print(f"PCA training completed in {pca_time:.3f} seconds")

    print("\nEvaluating on full signal...")
    start_time = time.time()
    scores = detector.score(X_full)
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.3f} seconds")

    scores_exp = np.kron(scores, np.ones(STRIDE))
    mask = create_label_mask(df, attack_is_one=True)
    mask_ = mask[: len(scores_exp)]

    scores_inverted = -scores_exp
    metrics = calc_metrics(mask_, scores_inverted, plot_roc=False)

    roc_auc = metrics["roc_auc"]
    th_optimal = metrics["threshold"]
    sensitivity = metrics["sensitivity"]
    specificity = metrics["specificity"]
    precision = metrics["precision"]
    f1 = metrics["f1_score"]

    print("\nMetrics:")
    print(f"  AUC: {roc_auc:.3f}")
    print(f"  Optimal threshold: {th_optimal:.3f}")
    print(f"  Sensitivity: {sensitivity:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  F1-score: {f1:.3f}")

    if show_roc_plot:
        fpr, tpr, _ = roc_curve(mask_, scores_inverted)
        roc_auc_plot = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_plot:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - SWaT (PCA)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    results = {
        "pca_training_time": pca_time,
        "evaluation_time": eval_time,
        "roc_auc": roc_auc,
        "threshold": th_optimal,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
    }
    return results


def run_swat_pca(csv_path: str = None, auto_download: bool = True):
    df = load_swat_data(csv_path, auto_download=auto_download)
    features = get_features(df)
    results = process_swat_pca(df, features, show_roc_plot=False)

    metrics_order = [
        "roc_auc",
        "sensitivity",
        "specificity",
        "precision",
        "f1_score",
        "threshold",
        "evaluation_time",
    ]

    results_df = pd.DataFrame({"SWaT_PCA": results}).T[metrics_order]
    output_file = "results_swat_pca.xlsx"
    results_df.to_excel(output_file)
    print(f"\nResults saved to {output_file}")

    return results


##################################################################
# MAIN
##################################################################

if __name__ == "__main__":
    print("=" * 70)
    print("SWaT PCA Anomaly Detection Pipeline")
    print("=" * 70)
    print("\nParameters:")
    print(f"  N_COMPONENTS: {N_COMPONENTS}")
    print(f"  WINDOW_LENGTH: {WINDOW_LENGTH}")
    print(f"  STRIDE: {STRIDE}")
    print(f"  SCORE_TYPE: {SCORE_TYPE}")

    run_swat_pca()

    print("\n" + "=" * 70)
    print("Processing completed!")
    print("=" * 70)

    input("\nPress ENTER to exit...")
