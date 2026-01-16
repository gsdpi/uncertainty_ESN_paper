##################################################################
# Main script for ECG5000 dataset processing with KNN
# anomaly detection (no ESN reservoir)
##################################################################

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors
import os

# Shared utilities
from esn_uncertainty import calc_metrics
from ecg5000_utils import load_ecg5000_data, prepare_train_test_split, get_ecg_features

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
# KNN ANOMALY DETECTOR
##################################################################

class KNNAnomalyDetector:
    """
    KNN-based anomaly detector for ECG time series.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        score_reduction: str = "mean"
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.score_reduction = score_reduction
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric
        )

    def fit(self, X_train: np.ndarray):
        """
        Fit the KNN model using normal (training) data.

        Parameters
        ----------
        X_train : ndarray, shape (N, 140)
        """
        print(f'Fitting KNN model with {len(X_train)} samples...')
        self.knn.fit(X_train)
        print('KNN model fitted.')
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Parameters
        ----------
        X : ndarray, shape (M, 140)

        Returns
        -------
        scores : ndarray, shape (M,)
            Higher score => more anomalous
        """
        distances, _ = self.knn.kneighbors(X)
        
        if self.score_reduction == "mean":
            scores = distances.mean(axis=1)
        elif self.score_reduction == "max":
            scores = distances.max(axis=1)
        elif self.score_reduction == "median":
            scores = np.median(distances, axis=1)
        else:
            raise ValueError(f"Unsupported reduction: {self.score_reduction}")
        
        return scores


##################################################################
# PROCESSING
##################################################################

def process_ecg5000_knn():
    """
    Process ECG5000 dataset using KNN anomaly detection.
    """
    print('='*70)
    print('ECG5000 KNN Anomaly Detection')
    print('='*70)
    print(f'\nParameters:')
    print(f'  N_NEIGHBORS: {N_NEIGHBORS}')
    print(f'  METRIC: {KNN_METRIC}')
    print(f'  SCORE_REDUCTION: {SCORE_REDUCTION}')
    
    # Load and prepare data
    print('\nLoading ECG5000 dataset...')
    df = load_ecg5000_data()
    df_train, df_test = prepare_train_test_split(df, train_ratio=0.75)
    
    features = [col for col in range(1, 141)]
    X_train = df_train[features].values
    X_test = df_test[features].values
    y_test = df_test['label'].values
    
    print(f'\nTraining data shape: {X_train.shape}')
    print(f'Test data shape: {X_test.shape}')
    
    # Train KNN model
    print('\nTraining KNN anomaly detector...')
    start_time = time.time()
    
    detector = KNNAnomalyDetector(
        n_neighbors=N_NEIGHBORS,
        metric=KNN_METRIC,
        score_reduction=SCORE_REDUCTION
    )
    
    detector.fit(X_train)
    knn_time = time.time() - start_time
    print(f'KNN training completed in {knn_time:.3f} seconds')
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    start_time = time.time()
    scores = detector.score(X_test)
    eval_time = time.time() - start_time
    print(f'Evaluation completed in {eval_time:.3f} seconds')
    
    # Create binary labels for scoring (1 = normal, 0 = anomaly)
    mask = (y_test == 1.0).astype(int)
    
    # Invert scores: higher score = more anomalous, but calc_metrics expects
    # higher values for the positive class (normal)
    scores_inverted = -scores
    
    # Calculate metrics
    metrics = calc_metrics(mask, scores_inverted, plot_roc=False)
    
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
    
    results = {
        'Method': 'KNN',
        'n_neighbors': N_NEIGHBORS,
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

    # Summary and save to Excel
    print('\n' + '='*70)
    print('SUMMARY - ECG5000 KNN')
    print('='*70)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"Recall @ FPR<=1%: {recall_at_1pct:.4f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"F1-score: {f1:.3f}")

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

    results_df = pd.DataFrame([results])
    results_df = results_df[metrics_order + ['Method', 'n_neighbors']]
    output_file = 'results_ecg5000_knn.xlsx'
    results_df.to_excel(output_file, index=False)
    print(f'\nResults saved to {output_file}')

    return results


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':
    results = process_ecg5000_knn()
    
    print('\n' + '='*70)
    print('Processing completed!')
    print('='*70)
    
    input('\nPress ENTER to exit...')
