##################################################################
# Main script for ECG5000 dataset processing with MiniRocket
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
from ecg5000_utils import load_ecg5000_data, prepare_train_test_split, get_ecg_features

# Figure configuration
plt.rcParams.update({'font.size': 18})


def convert_to_sktime_format(X: np.ndarray) -> pd.DataFrame:
    """
    Convert numpy array to sktime nested DataFrame format.
    
    Parameters
    ----------
    X : ndarray, shape (N, 140)
    
    Returns
    -------
    df : pandas.DataFrame
        Nested dataframe with shape (N, 1) where each cell contains a pd.Series
    """
    N, L = X.shape
    
    # Create nested dataframe (single dimension)
    data = {'dim_0': [pd.Series(X[i, :]) for i in range(N)]}
    
    return pd.DataFrame(data)


##################################################################
# GLOBAL PARAMETERS
##################################################################

# MiniRocket parameters
N_KERNELS = 10000  # Default for MiniRocket
SCORE_TYPE = "euclidean"  # 'euclidean', 'mahalanobis'

##################################################################
# MINIROCKET ANOMALY DETECTOR
##################################################################

class MiniRocketAnomalyDetector:
    """
    MiniRocket-based anomaly detector for ECG time series.
    """

    def __init__(
        self,
        n_kernels: int = 10000,
        score_type: str = "euclidean"
    ):
        self.n_kernels = n_kernels
        self.score_type = score_type
        
        self.minirocket = MiniRocket(num_kernels=self.n_kernels, random_state=42)
        self.centroid = None
        self.inv_cov = None  # For Mahalanobis distance

    def fit(self, X_train: np.ndarray):
        """
        Fit the MiniRocket model using normal (training) data.

        Parameters
        ----------
        X_train : ndarray, shape (N, 140)
        """
        print(f'Fitting MiniRocket with {len(X_train)} samples...')
        
        # Convert to sktime format
        X_sktime = convert_to_sktime_format(X_train)
        
        # Fit and transform
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

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for ECG samples.

        Parameters
        ----------
        X : ndarray, shape (M, 140)

        Returns
        -------
        scores : ndarray, shape (M,)
            Higher score => more anomalous
        """
        # Convert to sktime format
        X_sktime = convert_to_sktime_format(X)
        
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
        
        return scores


##################################################################
# PROCESSING
##################################################################

def process_ecg5000_minirocket():
    """
    Process ECG5000 dataset using MiniRocket anomaly detection.
    """
    print('='*70)
    print('ECG5000 MiniRocket Anomaly Detection')
    print('='*70)
    print(f'\nParameters:')
    print(f'  N_KERNELS: {N_KERNELS}')
    print(f'  SCORE_TYPE: {SCORE_TYPE}')
    
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
    
    # Train MiniRocket model
    print('\nTraining MiniRocket anomaly detector...')
    start_time = time.time()
    
    detector = MiniRocketAnomalyDetector(
        n_kernels=N_KERNELS,
        score_type=SCORE_TYPE
    )
    
    detector.fit(X_train)
    minirocket_time = time.time() - start_time
    print(f'MiniRocket training completed in {minirocket_time:.3f} seconds')
    
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
        'Method': 'MiniRocket',
        'n_kernels': N_KERNELS,
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

    # Summary and save to Excel
    print('\n' + '='*70)
    print('SUMMARY - ECG5000 MiniRocket')
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
        'minirocket_training_time',
        'evaluation_time'
    ]

    results_df = pd.DataFrame([results])
    results_df = results_df[metrics_order + ['Method', 'n_kernels']]
    output_file = 'results_ecg5000_minirocket.xlsx'
    results_df.to_excel(output_file, index=False)
    print(f'\nResults saved to {output_file}')

    return results


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':
    results = process_ecg5000_minirocket()
    
    print('\n' + '='*70)
    print('Processing completed!')
    print('='*70)
    
    input('\nPress ENTER to exit...')
