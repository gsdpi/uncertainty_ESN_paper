##################################################################
# Main script for ICANN dataset processing with KNN
# anomaly detection (no ESN reservoir)
##################################################################

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# Import KNN detector
from sklearn.neighbors import NearestNeighbors

# Shared utilities
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

# KNN parameters
N_NEIGHBORS = 10
KNN_METRIC = "euclidean"
SCORE_REDUCTION = "mean"  # 'mean', 'max', 'median'

##################################################################
# KNN HELPER FUNCTIONS
##################################################################

def window_signal(X: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Create sliding windows from a multichannel time series.

    Parameters
    ----------
    X : ndarray, shape (T, C)
        Time series with C channels
    window_size : int
        Number of samples per window
    stride : int
        Step between consecutive windows

    Returns
    -------
    windows : ndarray, shape (N, window_size, C)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    if X.ndim != 2:
        raise ValueError("X must have shape (T, C)")

    T, C = X.shape
    if T < window_size:
        raise ValueError("window_size larger than signal length")

    windows = sliding_window_view(X, window_size, axis=0)
    windows = windows[::stride]
    return windows


def vectorize_windows(windows: np.ndarray) -> np.ndarray:
    """
    Convert windows to feature vectors by flattening.

    Parameters
    ----------
    windows : ndarray, shape (N, W, C)

    Returns
    -------
    features : ndarray, shape (N, W*C)
    """
    return windows.reshape(windows.shape[0], -1)


def knn_score(distances: np.ndarray, reduction: str = "mean") -> np.ndarray:
    """
    Reduce KNN distances to an anomaly score.

    Parameters
    ----------
    distances : ndarray, shape (N, k)
    reduction : {'mean', 'max', 'median'}

    Returns
    -------
    scores : ndarray, shape (N,)
    """
    if reduction == "mean":
        return distances.mean(axis=1)
    elif reduction == "max":
        return distances.max(axis=1)
    elif reduction == "median":
        return np.median(distances, axis=1)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


##################################################################
# KNN ANOMALY DETECTOR
##################################################################

class KNNAnomalyDetector:
    """
    KNN-based anomaly detector for multichannel time series.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        window_size: int = 50,
        stride: int = 1,
        score_reduction: str = "mean",
        transition_window: int = None
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.window_size = window_size
        self.stride = stride
        self.score_reduction = score_reduction
        self.transition_window = transition_window if transition_window is not None else window_size

        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric
        )

    def fit(self, X_train: np.ndarray):
        """
        Fit the KNN model using normal (training) data.

        Parameters
        ----------
        X_train : ndarray, shape (T, C)
        """
        print(f'Windowing training data (window size {self.window_size}, stride {self.stride})...')
        windows = window_signal(
            X_train,
            self.window_size,
            self.stride
        )
        print(f'Generated {windows.shape[0]} training windows')
        X_vec = vectorize_windows(windows)
        self.knn.fit(X_vec)
        print('KNN model fitted.')
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for a time series.

        Parameters
        ----------
        X : ndarray, shape (T, C)

        Returns
        -------
        scores : ndarray, shape (N_windows,)
            Higher score => more anomalous
        """
        windows = window_signal(
            X,
            self.window_size,
            self.stride
        )
        X_vec = vectorize_windows(windows)

        distances, _ = self.knn.kneighbors(X_vec)
        scores = knn_score(distances, self.score_reduction)
        return scores


def process_signal_knn(df, features, signal_label, show_roc_plot=False):
    """
    Process a single signal using KNN anomaly detection.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset with all signals
    features : list
        List of feature column names
    signal_label : str
        Signal identifier (e.g., 'all', 'ax', 'ay')
    show_roc_plot : bool
        Whether to display ROC curve
        
    Returns
    -------
    results : dict
        Dictionary containing metrics and timing information
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
    
    # Train KNN model
    print('\nTraining KNN anomaly detector...')
    start_time = time.time()
    
    detector = KNNAnomalyDetector(
        n_neighbors=N_NEIGHBORS,
        window_size=WINDOW_LENGTH,
        stride=STRIDE,
        metric=KNN_METRIC,
        score_reduction=SCORE_REDUCTION,
        transition_window=WINDOW_LENGTH
    )
    
    detector.fit(X_train)
    knn_time = time.time() - start_time
    print(f'KNN training completed in {knn_time:.3f} seconds')
    
    # Evaluate on test signal
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
        plt.title(f'ROC Curve - {signal_label} (KNN)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    results = {
        'knn_training_time': knn_time,
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


def process_all_signals_knn(df):
    """
    Process different signal combinations using KNN anomaly detection.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset
    
    Returns
    -------
    all_results : list
        List of result dictionaries
    """
    all_features = get_features()
    
    # Process individual signals and combinations
    signal_configs = [
        (['ax'], 'ax'),
        (['ay'], 'ay'),
        (all_features, 'all_channels'),
    ]
    
    all_results = []
    
    for features, signal_label in signal_configs:
        results = process_signal_knn(
            df, features,
            signal_label=signal_label,
            show_roc_plot=False
        )
        
        # Store results
        all_results.append({
            'Signal': signal_label,
            **results
        })
    
    # Print summary
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
    
    # Save to Excel
    metrics_order = [
        'roc_auc',
        'auprc',
        'recall_at_1pct_fpr',
        'sensitivity',
        'specificity',
        'precision',
        'f1_score',
        'threshold',
        'knn_training_time',
        'evaluation_time'
    ]
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index('Signal')[metrics_order]
    
    output_file = 'results_icann_knn.xlsx'
    results_df.to_excel(output_file)
    print(f'\nResults saved to {output_file}')
    
    return all_results


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':
    print('='*70)
    print('ICANN KNN Anomaly Detection Pipeline')
    print('='*70)
    print(f'\nParameters:')
    print(f'  N_NEIGHBORS: {N_NEIGHBORS}')
    print(f'  WINDOW_LENGTH: {WINDOW_LENGTH}')
    print(f'  STRIDE: {STRIDE}')
    print(f'  METRIC: {KNN_METRIC}')
    print(f'  SCORE_REDUCTION: {SCORE_REDUCTION}')
    print(f'  Training experiments: {get_train_experiments()}')
    print(f'  Anomaly experiments: {get_anomaly_experiments()}')
    
    # Load data
    print('\nLoading ICANN dataset...')
    df = load_icann_data('./dataicann/dataicann.mat')
    print(f'Dataset loaded: {df.shape[0]} samples x {df.shape[1]} features')
    
    results = process_all_signals_knn(df)
    
    print('\n' + '='*70)
    print('Processing completed!')
    print('='*70)
    
    input('\nPress ENTER to close plots and exit...')
