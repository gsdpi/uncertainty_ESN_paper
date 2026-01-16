##################################################################
# Main script for IM-WSHA dataset processing with MiniRocket
# anomaly detection (no ESN reservoir)
##################################################################

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# MiniRocket from sktime
from sktime.transformations.panel.rocket import MiniRocket

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

# MiniRocket parameters
N_KERNELS = 10000  # Default for MiniRocket
SCORE_TYPE = 'euclidean' # 'euclidean', 'mahalanobis'
SEGMENT_SIZE = WINDOW_LENGTH  # Segment size for MiniRocket (must be >= 9)

##################################################################
# MINIROCKET HELPER FUNCTIONS
##################################################################

def segment_signal(X: np.ndarray, segment_size: int, stride: int = None) -> tuple:
    """
    Divide signal into segments with sliding window.

    Parameters
    ----------
    X : ndarray, shape (T, C)
        Time series with C channels
    segment_size : int
        Number of samples per segment
    stride : int, optional
        Step between consecutive segments. If None, uses non-overlapping (stride=segment_size)

    Returns
    -------
    segments : list of ndarray, each shape (segment_size, C)
    segment_starts : list of int
        Starting index of each segment
    """
    if stride is None:
        stride = segment_size
    
    T, C = X.shape
    
    segments = []
    segment_starts = []
    
    start = 0
    while start + segment_size <= T:
        end = start + segment_size
        segments.append(X[start:end, :])
        segment_starts.append(start)
        start += stride
    
    return segments, segment_starts


def convert_to_sktime_format(segments: list) -> pd.DataFrame:
    """
    Convert list of segments to sktime nested DataFrame format.
    
    Parameters
    ----------
    segments : list of ndarray, each shape (W, C)
    
    Returns
    -------
    df : pandas.DataFrame
        Nested dataframe with shape (N, C) where each cell contains a pd.Series
    """
    if len(segments) == 0:
        raise ValueError("No segments provided")
    
    W, C = segments[0].shape
    N = len(segments)
    
    # Create nested dataframe
    data = {}
    for c in range(C):
        data[f'dim_{c}'] = [pd.Series(segments[i][:, c]) for i in range(N)]
    
    return pd.DataFrame(data)


##################################################################
# MINIROCKET ANOMALY DETECTOR
##################################################################

class MiniRocketAnomalyDetector:
    """
    MiniRocket-based anomaly detector for multichannel time series.
    Uses segmentation instead of sliding windows for efficiency.
    """

    def __init__(
        self,
        n_kernels: int = 10000,
        segment_size: int = 200,
        score_type: str = "euclidean"
    ):
        self.n_kernels = n_kernels
        self.segment_size = segment_size
        self.score_type = score_type
        
        self.minirocket = MiniRocket(num_kernels=self.n_kernels, random_state=42)
        self.centroid = None
        self.inv_cov = None  # For Mahalanobis distance

    def fit(self, X_train: np.ndarray, stride: int = None):
        """
        Fit the MiniRocket model using normal (training) data.

        Parameters
        ----------
        X_train : ndarray, shape (T, C)
        stride : int, optional
            Stride for creating overlapping segments
        """
        print(f'Segmenting training data (segment size {self.segment_size}, stride {stride or self.segment_size})...')
        segments, _ = segment_signal(X_train, self.segment_size, stride=stride)
        print(f'Generated {len(segments)} training segments')
        
        if len(segments) == 0:
            raise ValueError("Not enough data to create segments")
        
        # Convert to sktime format
        X_sktime = convert_to_sktime_format(segments)
        
        # Fit and transform
        print('Fitting MiniRocket...')
        self.minirocket.fit(X_sktime)
        X_transformed = self.minirocket.transform(X_sktime)
        
        # Calculate centroid
        self.centroid = np.mean(X_transformed, axis=0)
        
        # Calculate inverse covariance for Mahalanobis distance
        if self.score_type == "mahalanobis":
            cov = np.cov(X_transformed.T)
            # Add small regularization to avoid singular matrix
            cov += np.eye(cov.shape[0]) * 1e-6
            try:
                self.inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                print("Warning: Covariance matrix is singular, using euclidean distance")
                self.score_type = "euclidean"
        
        print('MiniRocket model fitted.')
        return self

    def score(self, X: np.ndarray, stride: int = None) -> tuple:
        """
        Compute anomaly scores for a time series.

        Parameters
        ----------
        X : ndarray, shape (T, C)
        stride : int, optional
            Stride for creating overlapping segments

        Returns
        -------
        scores : ndarray, shape (N_segments,)
            Higher score => more anomalous
        segment_starts : ndarray, shape (N_segments,)
            Starting indices of each segment
        """
        segments, segment_starts = segment_signal(X, self.segment_size, stride=stride)
        
        if len(segments) == 0:
            return np.array([]), np.array([])
        
        # Convert to sktime format
        X_sktime = convert_to_sktime_format(segments)
        
        # Transform
        X_transformed = self.minirocket.transform(X_sktime)
        
        # Calculate distance to centroid
        if self.score_type == "euclidean":
            scores = np.linalg.norm(X_transformed - self.centroid, axis=1)
        elif self.score_type == "mahalanobis":
            diff = X_transformed - self.centroid
            scores = np.sqrt(np.sum(diff @ self.inv_cov * diff, axis=1))
        else:
            raise ValueError(f"Unsupported score_type: {self.score_type}")
        
        return scores, np.array(segment_starts)


def process_subject_minirocket(df, features, subject_label, show_roc_plot=False):
    """
    Process a single subject using MiniRocket anomaly detection.
    
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
    
    # Train MiniRocket model
    print('\nTraining MiniRocket anomaly detector...')
    start_time = time.time()
    
    detector = MiniRocketAnomalyDetector(
        n_kernels=N_KERNELS,
        segment_size=SEGMENT_SIZE,
        score_type=SCORE_TYPE
    )
    
    detector.fit(X_train, stride=STRIDE)
    minirocket_time = time.time() - start_time
    print(f'MiniRocket training completed in {minirocket_time:.3f} seconds')
    
    # Evaluate on full signal
    print('\nEvaluating on full signal...')
    start_time = time.time()
    scores, segment_starts = detector.score(X_full, stride=STRIDE)
    eval_time = time.time() - start_time
    print(f'Evaluation completed in {eval_time:.3f} seconds')
    
    # Calculate metrics
    # Expand scores to match signal length
    scores_exp = np.zeros(len(df))
    for i, (score, start) in enumerate(zip(scores, segment_starts)):
        end = min(start + SEGMENT_SIZE, len(df))
        scores_exp[start:end] = score
    
    # Create mask for train activities
    mask = np.isin(df['activity_label'], train_activities).astype(int)
    mask_ = mask[:len(scores_exp)]

    # For MiniRocket, higher score = more anomalous, so we need to invert for calc_metrics
    # which expects higher values for the positive class (train activities)
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
        plt.title(f'ROC Curve - {subject_label} (MiniRocket)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    results = {
        'minirocket_training_time': minirocket_time,
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


def process_all_subjects_minirocket():
    """
    Process all subjects using MiniRocket anomaly detection.
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
        results = process_subject_minirocket(
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
        'evaluation_time'
    ]

    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index('subject')[metrics_order].T
    
    output_file = 'results_imwsha_minirocket.xlsx'
    results_df.to_excel(output_file)
    print(f'\nResults saved to {output_file}')
    
    return all_results


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':
    print('='*70)
    print('IM-WSHA MiniRocket Anomaly Detection Pipeline')
    print('='*70)
    print(f'\nParameters:')
    print(f'  N_KERNELS: {N_KERNELS}')
    print(f'  SEGMENT_SIZE: {SEGMENT_SIZE}')
    print(f'  SCORE_TYPE: {SCORE_TYPE}')
    print(f'  Training activities: 1-{NT}')
    
    results = process_all_subjects_minirocket()
    
    print('\n' + '='*70)
    print('Processing completed!')
    print('='*70)
    
    input('\nPress ENTER to close plots and exit...')
