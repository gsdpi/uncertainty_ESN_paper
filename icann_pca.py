##################################################################
# Main script for ICANN dataset processing with PCA
# anomaly detection (no ESN reservoir)
##################################################################

import time
from typing import Union
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

from esn_uncertainty import calc_metrics
from icann_utils import (
    WINDOW_LENGTH,
    STRIDE,
    SAMPLING_PERIOD,
    load_icann_data,
    get_features,
    get_train_experiments,
    get_anomaly_experiments,
    get_normal_experiments,
    prepare_train_df,
    prepare_test_df,
    create_label_mask,
)

# Figure configuration
plt.rcParams.update({'font.size': 18})

##################################################################
# GLOBAL PARAMETERS
##################################################################

# PCA parameters
N_COMPONENTS: Union[float, int] = 0.95  # keep 95% variance or set explicit component count
SCORE_TYPE = "reconstruction_error"

##################################################################
# PCA HELPER FUNCTIONS
##################################################################

def window_signal(X: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Create sliding windows from a multichannel time series.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    if X.ndim != 2:
        raise ValueError("X must have shape (T, C)")

    T, _ = X.shape
    if T < window_size:
        raise ValueError("window_size larger than signal length")

    windows = sliding_window_view(X, window_size, axis=0)
    return windows[::stride]


def vectorize_windows(windows: np.ndarray) -> np.ndarray:
    """
    Convert windows to feature vectors by flattening.
    """
    return windows.reshape(windows.shape[0], -1)


##################################################################
# PCA ANOMALY DETECTOR
##################################################################

class PCAAnomalyDetector:
    """
    PCA-based anomaly detector for multichannel time series.
    """

    def __init__(
        self,
        n_components: Union[float, int] = 0.95,
        window_size: int = 50,
        stride: int = 1,
        score_type: str = "reconstruction_error"
    ):
        self.n_components = n_components
        self.window_size = window_size
        self.stride = stride
        self.score_type = score_type
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X_train: np.ndarray):
        """
        Fit the PCA model using normal (training) data.
        """
        print(f'Windowing training data (window size {self.window_size}, stride {self.stride})...')
        windows = window_signal(
            X_train,
            self.window_size,
            self.stride
        )
        print(f'Generated {windows.shape[0]} training windows')
        X_vec = vectorize_windows(windows)
        self.pca.fit(X_vec)
        print('PCA model fitted.')
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for a time series.
        """
        windows = window_signal(
            X,
            self.window_size,
            self.stride
        )
        X_vec = vectorize_windows(windows)

        X_proj = self.pca.transform(X_vec)
        X_rec = self.pca.inverse_transform(X_proj)

        if self.score_type == "reconstruction_error":
            scores = np.linalg.norm(X_vec - X_rec, axis=1)
        else:
            raise ValueError(f"Unsupported score_type: {self.score_type}")

        return scores


def process_signal_pca(df, features, signal_label, show_roc_plot=False):
    """
    Process a single signal using PCA anomaly detection.
    """
    print(f'\n{"="*70}')
    print(f'PROCESSING SIGNAL: {signal_label}')
    print(f'{"="*70}\n')

    train_experiments = get_train_experiments()
    df_train = prepare_train_df(df, train_experiments)
    test_experiments = get_normal_experiments()+get_anomaly_experiments()
    df_test = prepare_test_df(df, test_experiments)

    X_train = df_train[features].values
    X_test = df_test[features].values

    print(f'Training data shape: {X_train.shape}')
    print(f'Test data shape: {X_test.shape}')

    print('\nTraining PCA anomaly detector...')
    start_time = time.time()

    detector = PCAAnomalyDetector(
        n_components=N_COMPONENTS,
        window_size=WINDOW_LENGTH,
        stride=STRIDE,
        score_type=SCORE_TYPE
    )

    detector.fit(X_train)
    pca_time = time.time() - start_time
    print(f'PCA training completed in {pca_time:.3f} seconds')

    print('\nEvaluating on test signal...')
    start_time = time.time()
    scores = detector.score(X_test)
    eval_time = time.time() - start_time
    print(f'Evaluation completed in {eval_time:.3f} seconds')

    scores_exp = np.kron(scores, np.ones(STRIDE))

    mask = create_label_mask(df_test, get_normal_experiments(), get_anomaly_experiments())
    mask_ = mask[:len(scores_exp)]

    # Higher score = more anomalous, invert for calc_metrics which expects higher values for the positive class (normal=1)
    scores_inverted = -scores_exp
    metrics = calc_metrics(mask_, scores_inverted, plot_roc=False)

    roc_auc = metrics['roc_auc']
    auprc = metrics['auprc']
    recall_at_1pct = metrics['recall_at_1pct_fpr']
    th_optimal = metrics['threshold']
    sensitivity = metrics['sensitivity']
    specificity = metrics['specificity']
    precision = metrics['precision']
    f1 = metrics['f1_score']
    
    print(f'\nMetrics:')
    print(f'  AUC: {roc_auc:.3f}')
    print(f'  AUPRC: {auprc:.3f}')
    print(f'  Recall @ FPR<=1%: {recall_at_1pct:.3f}')
    print(f'  Precision: {precision:.3f}')
    print(f'  F1-score: {f1:.3f}')

    if show_roc_plot:
        fpr, tpr, _ = roc_curve(mask_, scores_inverted)
        roc_auc_plot = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_plot:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {signal_label} (PCA)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    results = {
        'pca_training_time': pca_time,
        'evaluation_time': eval_time,
        'roc_auc': roc_auc,
        'auprc': auprc,
        'recall_at_1pct_fpr': recall_at_1pct,
        'threshold': th_optimal,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
    }

    return results


def process_all_signals_pca(df):
    """
    Process different signal combinations using PCA anomaly detection.
    """
    all_features = get_features()

    signal_configs = [
        (['ax'], 'ax'),
        (['ay'], 'ay'),
        (all_features, 'all_channels'),
    ]

    all_results = []

    for features, signal_label in signal_configs:
        results = process_signal_pca(
            df, features,
            signal_label=signal_label,
            show_roc_plot=False
        )

        all_results.append({
            'Signal': signal_label,
            **results
        })

    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    for res in all_results:
        print(f"\n{res['Signal']}:")
        print(f"  ROC AUC: {res['roc_auc']:.4f}")
        print(f"  AUPRC: {res['auprc']:.4f}")
        print(f"  Recall @ FPR<=1%: {res['recall_at_1pct_fpr']:.3f}")
        print(f"  Sensitivity: {res['sensitivity']:.3f}")
        print(f"  Specificity: {res['specificity']:.3f}")
        print(f"  Precision: {res['precision']:.3f}")
        print(f"  F1-score: {res['f1_score']:.3f}")

    metrics_order = [
        'roc_auc',
        'auprc',
        'recall_at_1pct_fpr',
        'sensitivity',
        'specificity',
        'precision',
        'f1_score',
        'threshold',
        'pca_training_time',
        'evaluation_time'
    ]

    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index('Signal')[metrics_order]

    output_file = 'results_icann_pca.xlsx'
    results_df.to_excel(output_file)
    print(f'\nResults saved to {output_file}')

    return all_results


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':
    print('='*70)
    print('ICANN PCA Anomaly Detection Pipeline')
    print('='*70)
    print(f'\nParameters:')
    print(f'  N_COMPONENTS: {N_COMPONENTS}')
    print(f'  WINDOW_LENGTH: {WINDOW_LENGTH}')
    print(f'  STRIDE: {STRIDE}')
    print(f'  SCORE_TYPE: {SCORE_TYPE}')
    print(f'  Training experiments: {get_train_experiments()}')
    print(f'  Anomaly experiments: {get_anomaly_experiments()}')

    print('\nLoading ICANN dataset...')
    df = load_icann_data('./dataicann/dataicann.mat')
    print(f'Dataset loaded: {df.shape[0]} samples x {df.shape[1]} features')

    results = process_all_signals_pca(df)

    print('\n' + '='*70)
    print('Processing completed!')
    print('='*70)

    input('\nPress ENTER to close plots and exit...')
