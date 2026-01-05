##################################################################
# Main script for IM-WSHA dataset processing with PCA
# anomaly detection (no ESN reservoir)
##################################################################

import time
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

from sklearn.decomposition import PCA

# Shared utilities
from esn_uncertainty import calc_metrics
from imwsha_utils import (
    NT,
    WINDOW_LENGTH,
    STRIDE,
    SAMPLING_PERIOD,
    clean_subject_data,
    load_subject_df,
    get_features,
    get_train_activities,
    prepare_train_df,
)

# Figure configuration
plt.rcParams.update({'font.size': 18})

##################################################################
# GLOBAL PARAMETERS
##################################################################

# PCA parameters
N_COMPONENTS = 0.95  # keep 95% variance or set explicit component count
SCORE_TYPE = "reconstruction_error"

##################################################################
# PCA HELPER FUNCTIONS
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
        self.pca.fit(X_vec)
        print('PCA model fitted.')
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

        X_proj = self.pca.transform(X_vec)
        X_rec = self.pca.inverse_transform(X_proj)

        if self.score_type == "reconstruction_error":
            scores = np.linalg.norm(X_vec - X_rec, axis=1)
        else:
            raise ValueError(f"Unsupported score_type: {self.score_type}")

        return scores


def process_subject_pca(df, features, subject_label, show_roc_plot=False):
    """
    Process a single subject using PCA anomaly detection.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Subject data with features and activity_label
    features : list
        List of feature column names
    subject_label : str
        Subject identifier
    show_roc_plot : bool
        Whether to display ROC curve
        
    Returns
    -------
    results : dict
        Dictionary containing metrics and timing information
    """
    print(f'\n{"="*70}')
    print(f'PROCESSING: {subject_label}')
    print(f'{"="*70}\n')
    
    train_activities = get_train_activities()
    df_train = prepare_train_df(df, train_activities, trim=150)
    
    # Extract features as numpy arrays
    X_train = df_train[features].values
    X_full = df[features].values
    
    print(f'Training data shape: {X_train.shape}')
    print(f'Full data shape: {X_full.shape}')
    
    # Train PCA model
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
    
    # Evaluate on full signal
    print('\nEvaluating on full signal...')
    start_time = time.time()
    scores = detector.score(X_full)
    eval_time = time.time() - start_time
    print(f'Evaluation completed in {eval_time:.3f} seconds')
    
    # Calculate metrics
    # Expand score to adjust lengths
    scores_exp = np.kron(scores, np.ones(STRIDE))

    # Create mask for train activities
    mask = np.isin(df['activity_label'], train_activities).astype(int)
    mask_ = mask[:len(scores_exp)]

    # Higher score = more anomalous, invert for calc_metrics which expects
    # higher scores for the positive (train) class
    scores_inverted = -scores_exp
    metrics = calc_metrics(mask_, scores_inverted, plot_roc=False)
    
    roc_auc = metrics['roc_auc']
    th_optimal = metrics['threshold']
    sensitivity = metrics['sensitivity']
    specificity = metrics['specificity']
    precision = metrics['precision']
    f1 = metrics['f1_score']
    
    print(f'\nMetrics:')
    print(f'  AUC: {roc_auc:.3f}')
    print(f'  Optimal threshold: {th_optimal:.3f}')
    print(f'  Sensitivity: {sensitivity:.3f}')
    print(f'  Specificity: {specificity:.3f}')
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
        plt.title(f'ROC Curve - {subject_label} (PCA)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    results = {
        'pca_training_time': pca_time,
        'evaluation_time': eval_time,
        'roc_auc': roc_auc,
        'threshold': th_optimal,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
    }
    
    return results


def process_all_subjects_pca():
    """
    Process all subjects using PCA anomaly detection.
    """
    dataset_path = './IM-WSHA_Dataset/IMSHA_Dataset'
    subject_dirs = sorted(
        [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('Subject')]
    )
    print(f'Found {len(subject_dirs)} subjects')

    all_results = []

    for subject_dir in subject_dirs:
        df = load_subject_df(dataset_path, subject_dir)
        features = get_features(df)

        # Process subject
        results = process_subject_pca(
            df, features,
            subject_label=subject_dir,
            show_roc_plot=False
        )

        # Store results
        all_results.append({
            'subject': subject_dir,
            **results
        })

    # Print summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    for res in all_results:
        print(f"\n{res['subject']}:")
        print(f"  ROC AUC: {res['roc_auc']:.4f}")
        print(f"  Sensitivity: {res['sensitivity']:.3f}")
        print(f"  Specificity: {res['specificity']:.3f}")
        print(f"  Precision: {res['precision']:.3f}")
        print(f"  F1-score: {res['f1_score']:.3f}")

    # Save to Excel

    metrics_order = [
        'roc_auc',
        'sensitivity',
        'specificity',
        'precision',
        'f1_score',
        'threshold',
        'evaluation_time'
    ]

    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index('subject')[metrics_order].T
    
    output_file = 'results_imwsha_pca.xlsx'
    results_df.to_excel(output_file)
    print(f'\nResults saved to {output_file}')
    
    return all_results


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':
    print('='*70)
    print('IM-WSHA PCA Anomaly Detection Pipeline')
    print('='*70)
    print(f'\nParameters:')
    print(f'  N_COMPONENTS: {N_COMPONENTS}')
    print(f'  WINDOW_LENGTH: {WINDOW_LENGTH}')
    print(f'  STRIDE: {STRIDE}')
    print(f'  SCORE_TYPE: {SCORE_TYPE}')
    print(f'  Training activities: 1-{NT}')
    
    results = process_all_subjects_pca()
    
    print('\n' + '='*70)
    print('Processing completed!')
    print('='*70)
    
    input('\nPress ENTER to close plots and exit...')
