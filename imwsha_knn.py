##################################################################
# Main script for IM-WSHA dataset processing with KNN
# anomaly detection (no ESN reservoir)
##################################################################

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import glob

# Import KNN detector from test-knn.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.neighbors import NearestNeighbors

# Import utilities
from esn_uncertainty import calc_metrics

# Figure configuration
plt.rcParams.update({'font.size': 18})

##################################################################
# GLOBAL PARAMETERS
##################################################################

# Processing parameters
NT = 7  # number of activities to train
WINDOW_LENGTH = 140  # L
STRIDE = 20  # S
SAMPLING_PERIOD = 1 / 20.  # tm

# KNN parameters
N_NEIGHBORS = 10
KNN_METRIC = "euclidean"
SCORE_REDUCTION = "mean"  # 'mean', 'max', 'median'

##################################################################
# KNN HELPER FUNCTIONS (from test-knn.py)
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


##################################################################
# SUBJECT CLEANING RULES (from imwsha_main.py)
##################################################################

SUBJECT_CLEANING = {
    'Subject 1': [
        (0, 200, 12),
        (1150, 1375, 1),
        (2390, 2510, 2),
        (3300, 3840, 3),
        (3840, 4000, 12),
        (4000, 4800, 4),
        (4800, 4950, 12),
        (4950, 6050, 5),
        (6050, 6300, 12),
        (6300, 7350, 6),
        (7350, 7500, 12),
        (7500, 8500, 7),
        (8500, 8700, 12),
        (8700, 9700, 8),
        (9700, 9850, 12),
        (9850, 10950, 9),
        (10950, 11050, 12),
        (11050, 12000, 10),
        (12000, 12100, 12),
        (12100, 12600, 11),
    ],
    'Subject 2': [
        (0, 200, 12),
        (1200, 1400, 12),
        (2400, 2550, 12),
        (3500, 3650, 3),
        (3650, 3850, 12),
        (3850, 4800, 4),
        (4800, 4950, 12),
        (4950, 6030, 5),
        (6030, 6150, 12),
        (6150, 7230, 6),
        (7230, 7300, 12),
        (7300, 8400, 7),
        (8400, 8500, 12),
        (8500, 9620, 8),
        (9620, 9800, 12),
        (9800, 10850, 9),
        (10850, 10930, 12),
        (10930, 11890, 10),
        (11890, 12080, 12),
        (12080, 12400, 11),
    ],
    'Subject 3': [
        (0, 200, 12),
        (1190, 1370, 12),
        (2385, 2600, 12),
        (3500, 3740, 3),
        (3740, 3790, 12),
        (4800, 5020, 12),
        (5020, 6000, 5),
        (6000, 6150, 12),
        (6150, 7170, 6),
        (7170, 7250, 12),
        (7250, 8350, 7),
        (8350, 8450, 12),
        (8450, 9650, 8),
        (9650, 9710, 12),
        (9710, 10800, 9),
        (10800, 10875, 12),
        (10875, 11820, 10),
        (11820, 11900, 12),
        (11900, 12000, 11),
    ],
    'Subject 4': [
        (0, 200, 12),
        (1200, 1400, 12),
        (2400, 2500, 12),
        (3200, 3700, 3),
        (3700, 3850, 12),
        (4800, 5000, 12),
        (5000, 6000, 5),
        (6000, 6150, 12),
        (6150, 7200, 6),
        (7200, 7300, 12),
        (7300, 8450, 7),
        (8450, 8510, 12),
        (8510, 9630, 8),
        (9630, 9750, 12),
        (9750, 10820, 9),
        (10820, 10935, 12),
        (10935, 11820, 10),
        (11820, 11940, 12),
        (11940, 12500, 11),
    ],
    'Subject 5': [
        (0, 200, 12),
        (200, 1200, 1),
        (1200, 1500, 12),
        (2380, 2550, 12),
        (2550, 3715, 3),
        (3715, 4000, 12),
        (4000, 4800, 4),
        (4800, 4950, 12),
        (4950, 6000, 5),
        (6000, 6150, 12),
        (6150, 7200, 6),
        (7200, 7300, 12),
        (7300, 8500, 7),
        (8500, 8650, 12),
        (8650, 9750, 8),
        (9750, 9860, 12),
        (9860, 10950, 9),
        (10950, 11050, 12),
        (11050, 12000, 10),
        (12000, 12100, 12),
        (12100, 12500, 11),
    ],
    'Subject 6': [
        (0, 200, 12),
        (1200, 1400, 12),
        (2400, 2550, 12),
        (3500, 3700, 3),
        (3700, 3850, 12),
        (3850, 4800, 4),
        (4800, 4950, 12),
        (4950, 6030, 5),
        (6030, 6150, 12),
        (6150, 7200, 6),
        (7200, 7350, 12),
        (7350, 8450, 7),
        (8450, 8550, 12),
        (8550, 9700, 8),
        (9700, 9800, 12),
        (9800, 10900, 9),
        (10900, 11000, 12),
        (11000, 11900, 10),
        (11900, 12050, 12),
        (12050, 12500, 11),
    ],
    'Subject 7': [
        (0, 200, 12),
        (1190, 1370, 12),
        (2385, 2600, 12),
        (3600, 3850, 3),
        (3850, 4000, 12),
        (4000, 4800, 4),
        (4800, 4950, 12),
        (4950, 6030, 5),
        (6030, 6150, 12),
        (6150, 7200, 6),
        (7200, 7350, 12),
        (7350, 8500, 7),
        (8500, 8600, 12),
        (8600, 9700, 8),
        (9700, 9850, 12),
        (9850, 10950, 9),
        (10950, 11050, 12),
        (11050, 12000, 10),
        (12000, 12100, 12),
        (12100, 12500, 11),
    ],
    'Subject 8': [
        (0, 200, 12),
        (1190, 1370, 12),
        (2385, 2600, 12),
        (3500, 3740, 3),
        (3740, 3850, 12),
        (3850, 4800, 4),
        (4800, 4950, 12),
        (4950, 6030, 5),
        (6030, 6150, 12),
        (6150, 7200, 6),
        (7200, 7300, 12),
        (7300, 8450, 7),
        (8450, 8550, 12),
        (8550, 9650, 8),
        (9650, 9750, 12),
        (9750, 10850, 9),
        (10850, 10950, 12),
        (10950, 11900, 10),
        (11900, 12050, 12),
        (12050, 12500, 11),
    ],
    'Subject 9': [
        (0, 200, 12),
        (1190, 1370, 12),
        (2385, 2600, 12),
        (3500, 3740, 3),
        (3740, 3850, 12),
        (3850, 4800, 4),
        (4800, 4950, 12),
        (4950, 6030, 5),
        (6030, 6150, 12),
        (6150, 7200, 6),
        (7200, 7300, 12),
        (7300, 8450, 7),
        (8450, 8550, 12),
        (8550, 9650, 8),
        (9650, 9750, 12),
        (9750, 10850, 9),
        (10850, 10950, 12),
        (10950, 11900, 10),
        (11900, 12050, 12),
        (12050, 12500, 11),
    ],
    'Subject 10': [
        (0, 200, 12),
        (1190, 1370, 12),
        (2385, 2600, 12),
        (3500, 3740, 3),
        (3740, 3850, 12),
        (3850, 4800, 4),
        (4800, 4950, 12),
        (4950, 6030, 5),
        (6030, 6150, 12),
        (6150, 7200, 6),
        (7200, 7300, 12),
        (7300, 8450, 7),
        (8450, 8550, 12),
        (8550, 9650, 8),
        (9650, 9750, 12),
        (9750, 10850, 9),
        (10850, 10950, 12),
        (10950, 11900, 10),
        (11900, 12050, 12),
        (12050, 12500, 11),
    ],
}

##################################################################
# UTILITY FUNCTIONS
##################################################################

def clean_subject_data(df, subject_label):
    """
    Clean activity labels for a specific subject based on predefined ranges.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with activity data to clean
    subject_label : str
        Subject identifier (e.g., 'Subject 1')
        
    Returns
    -------
    df : pandas.DataFrame
        Cleaned dataframe with corrected activity labels
    """
    if subject_label not in SUBJECT_CLEANING:
        print(f'WARNING: No cleaning rules found for {subject_label}, returning unchanged data.')
        return df
    
    print(f'Cleaning activity labels for {subject_label}...')
    cleaning_rules = SUBJECT_CLEANING[subject_label]
    
    for start, end, label in cleaning_rules:
        if end == -1:
            end = len(df) - 1
        df.loc[start:end, 'activity_label'] = label
    
    print(f'Cleaning completed for {subject_label}')
    return df


def process_subject_knn(df, features, subject_label, show_roc_plot=False):
    """
    Process a single subject using KNN anomaly detection.
    
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
    
    # Prepare training data (activities 1-NT)
    train_activities = np.arange(1, NT + 1)
    
    df_train = pd.DataFrame()
    for aa in train_activities:
        df_tmp = df.loc[df['activity_label'] == aa]
        # Trim 150 samples from start and end
        df_train = pd.concat([df_train, df_tmp[150:-150]])
    
    # Extract features as numpy arrays
    X_train = df_train[features].values
    X_full = df[features].values
    
    print(f'Training data shape: {X_train.shape}')
    print(f'Full data shape: {X_full.shape}')
    
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

    # For KNN, higher score = more anomalous, so we need to invert for calc_metrics
    # which expects higher values for the positive class (train activities)
    scores_inverted = -scores_exp
    metrics = calc_metrics(mask_, scores_inverted, plot_roc=False)
    # plt.plot(scores_inverted, label='Anomaly Score (inverted)', color='blue')
    # plt.plot(X_full[:, 0:3], label='Signal (channels 0:3)', color='orange', alpha=0.5)
    # plt.show()
    
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
        fpr, tpr, _ = roc_curve(mask_aligned, scores_inverted)
        roc_auc_plot = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_plot:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {subject_label} (KNN)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    results = {
        'knn_training_time': knn_time,
        'evaluation_time': eval_time,
        'roc_auc': roc_auc,
        'threshold': th_optimal,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
    }
    
    return results


def process_all_subjects_knn():
    """
    Process all subjects using KNN anomaly detection.
    """
    dataset_path = './IM-WSHA_Dataset/IMSHA_Dataset'
    subject_dirs = sorted([d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('Subject')])
    print(f'Found {len(subject_dirs)} subjects')

    all_results = []

    for subject_dir in subject_dirs:
        subject_path = os.path.join(dataset_path, subject_dir)
        csv_files = glob.glob(os.path.join(subject_path, '*.csv'))
        if not csv_files:
            print(f'\nWARNING: No CSV file found for {subject_dir}, skipping...')
            continue

        csv_file = csv_files[0]
        print(f'\nLoading data for {subject_dir} from {os.path.basename(csv_file)}...')
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['activity_label'])

        # Clean data
        df = clean_subject_data(df, subject_dir)

        # Extract features (all columns except activity_label)
        features = [col for col in df.columns if col != 'activity_label']

        # Process subject
        results = process_subject_knn(
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
    
    output_file = 'results_imwsha_knn.xlsx'
    results_df.to_excel(output_file)
    print(f'\nResults saved to {output_file}')
    
    return all_results


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':
    print('='*70)
    print('IM-WSHA KNN Anomaly Detection Pipeline')
    print('='*70)
    print(f'\nParameters:')
    print(f'  N_NEIGHBORS: {N_NEIGHBORS}')
    print(f'  WINDOW_LENGTH: {WINDOW_LENGTH}')
    print(f'  STRIDE: {STRIDE}')
    print(f'  METRIC: {KNN_METRIC}')
    print(f'  SCORE_REDUCTION: {SCORE_REDUCTION}')
    print(f'  Training activities: 1-{NT}')
    
    results = process_all_subjects_knn()
    
    print('\n' + '='*70)
    print('Processing completed!')
    print('='*70)
    
    input('\nPress ENTER to close plots and exit...')
