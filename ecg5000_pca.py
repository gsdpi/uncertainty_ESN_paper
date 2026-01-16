##################################################################
# Main script for ECG5000 dataset processing with PCA
# anomaly detection (no ESN reservoir)
##################################################################

import time
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import os

# Shared utilities
from esn_uncertainty import calc_metrics
from ecg5000_utils import load_ecg5000_data, prepare_train_test_split, get_ecg_features

# Figure configuration
plt.rcParams.update({'font.size': 18})


##################################################################
# GLOBAL PARAMETERS
##################################################################

# PCA parameters
N_COMPONENTS: Union[float, int] = 0.95  # keep 95% variance

##################################################################
# PCA ANOMALY DETECTOR
##################################################################

class PCAAnomalyDetector:
    """
    PCA-based anomaly detector for ECG time series.
    """

    def __init__(self, n_components: Union[float, int] = 0.95):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X_train: np.ndarray):
        """
        Fit the PCA model using normal (training) data.

        Parameters
        ----------
        X_train : ndarray, shape (N, 140)
        """
        print(f'Fitting PCA model with {len(X_train)} samples...')
        self.pca.fit(X_train)
        print(f'PCA model fitted. Components kept: {self.pca.n_components_}')
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores based on reconstruction error.

        Parameters
        ----------
        X : ndarray, shape (M, 140)

        Returns
        -------
        scores : ndarray, shape (M,)
            Higher score => more anomalous
        """
        X_proj = self.pca.transform(X)
        X_rec = self.pca.inverse_transform(X_proj)
        
        scores = np.linalg.norm(X - X_rec, axis=1)
        return scores


##################################################################
# PROCESSING
##################################################################

def process_ecg5000_pca():
    """
    Process ECG5000 dataset using PCA anomaly detection.
    """
    print('='*70)
    print('ECG5000 PCA Anomaly Detection')
    print('='*70)
    print(f'\nParameters:')
    print(f'  N_COMPONENTS: {N_COMPONENTS}')
    
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
    
    # Train PCA model
    print('\nTraining PCA anomaly detector...')
    start_time = time.time()
    
    detector = PCAAnomalyDetector(n_components=N_COMPONENTS)
    detector.fit(X_train)
    pca_time = time.time() - start_time
    print(f'PCA training completed in {pca_time:.3f} seconds')
    
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
        'Method': 'PCA',
        'n_components': N_COMPONENTS,
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

    # Summary and save to Excel
    print('\n' + '='*70)
    print('SUMMARY - ECG5000 PCA')
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
        'pca_training_time',
        'evaluation_time'
    ]

    results_df = pd.DataFrame([results])
    results_df = results_df[metrics_order + ['Method', 'n_components']]
    output_file = 'results_ecg5000_pca.xlsx'
    results_df.to_excel(output_file, index=False)
    print(f'\nResults saved to {output_file}')

    return results


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':
    results = process_ecg5000_pca()
    
    print('\n' + '='*70)
    print('Processing completed!')
    print('='*70)
    
    input('\nPress ENTER to exit...')
