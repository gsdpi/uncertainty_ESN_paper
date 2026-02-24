##################################################################
# Main script for synthetic signal processing with MiniRocket
# anomaly detection (no ESN reservoir)
##################################################################

import time
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from esn_uncertainty import calc_metrics
from synth_utils import (
    N_POINTS,
    WINDOW_SIZE,
    STEP,
    build_synthetic_dataset,
)

# Figure configuration
plt.rcParams.update({'font.size': 18})


##################################################################
# GLOBAL PARAMETERS
##################################################################

N_KERNELS = 10_000
SCORE_TYPE = "euclidean"  # 'euclidean', 'mahalanobis'
RANDOM_SEED = 42


def get_minirocket_class():
    """
    Dynamically load MiniRocket class from sktime.
    """
    try:
        rocket_module = importlib.import_module("sktime.transformations.panel.rocket")
    except ImportError as exc:
        raise ImportError(
            "sktime is required for synth_minirocket.py. "
            "Install it with: pip install sktime"
        ) from exc

    return rocket_module.MiniRocket


def convert_to_sktime_format(X: np.ndarray) -> pd.DataFrame:
    """
    Convert numpy array to sktime nested DataFrame format.

    Parameters
    ----------
    X : ndarray, shape (N, L)

    Returns
    -------
    df : pandas.DataFrame
        Nested dataframe with shape (N, 1) where each cell contains a pd.Series.
    """
    n_samples, _ = X.shape
    data = {'dim_0': [pd.Series(X[i, :]) for i in range(n_samples)]}
    return pd.DataFrame(data)


##################################################################
# MINIROCKET ANOMALY DETECTOR
##################################################################

class MiniRocketAnomalyDetector:
    """
    MiniRocket-based anomaly detector for synthetic windowed signals.
    """

    def __init__(
        self,
        n_kernels: int = 10_000,
        score_type: str = "euclidean",
    ):
        self.n_kernels = n_kernels
        self.score_type = score_type

        MiniRocket = get_minirocket_class()
        self.minirocket = MiniRocket(num_kernels=self.n_kernels, random_state=RANDOM_SEED)
        self.centroid = None
        self.inv_cov = None

    def fit(self, X_train: np.ndarray):
        """
        Fit MiniRocket model using normal training windows.
        """
        print(f'Fitting MiniRocket with {len(X_train)} windows...')
        X_sktime = convert_to_sktime_format(X_train)

        self.minirocket.fit(X_sktime)
        X_transformed = self.minirocket.transform(X_sktime)

        self.centroid = np.mean(X_transformed, axis=0)

        if self.score_type == "mahalanobis":
            cov = np.cov(X_transformed.T)
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
        Compute anomaly scores.

        Higher score => more anomalous.
        """
        X_sktime = convert_to_sktime_format(X)
        X_transformed = self.minirocket.transform(X_sktime)

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

def process_synthetic_minirocket(show_plot: bool = True):
    """
    Process synthetic dataset using MiniRocket anomaly detection.
    """
    print('=' * 70)
    print('Synthetic Signal MiniRocket Anomaly Detection')
    print('=' * 70)
    print(f'\nParameters:')
    print(f'  N_KERNELS: {N_KERNELS}')
    print(f'  SCORE_TYPE: {SCORE_TYPE}')
    print(f'  N_POINTS: {N_POINTS}')
    print(f'  WINDOW_SIZE: {WINDOW_SIZE}')
    print(f'  STEP: {STEP}')

    print('\nBuilding synthetic dataset...')
    data = build_synthetic_dataset(
        n_points=N_POINTS,
        window_size=WINDOW_SIZE,
        step=STEP,
        random_seed=RANDOM_SEED,
    )

    X_train = data['windows']['X_train']
    X_test = data['windows']['X_test_anomalous']
    y_test = data['windows']['y_test_anomalous']

    print(f'Training windows: {X_train.shape}')
    print(f'Test windows: {X_test.shape}')
    print(f'Test labels distribution (1=normal,0=anomaly):')
    print(pd.Series(y_test).value_counts().sort_index())

    print('\nTraining MiniRocket anomaly detector...')
    start_time = time.time()

    detector = MiniRocketAnomalyDetector(
        n_kernels=N_KERNELS,
        score_type=SCORE_TYPE,
    )

    detector.fit(X_train)
    minirocket_time = time.time() - start_time
    print(f'MiniRocket training completed in {minirocket_time:.3f} seconds')

    print('\nEvaluating on anomalous synthetic test signal...')
    start_time = time.time()
    scores = detector.score(X_test)
    eval_time = time.time() - start_time
    print(f'Evaluation completed in {eval_time:.3f} seconds')

    scores_inverted = -scores
    metrics = calc_metrics(y_test, scores_inverted, plot_roc=False)

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
        'dataset': 'synthetic_signal',
        'n_kernels': N_KERNELS,
        'score_type': SCORE_TYPE,
        'n_points': N_POINTS,
        'window_size': WINDOW_SIZE,
        'step': STEP,
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
        'evaluation_time',
    ]

    results_df = pd.DataFrame([results])
    results_df = results_df[
        metrics_order + [
            'Method', 'dataset', 'n_kernels', 'score_type',
            'n_points', 'window_size', 'step'
        ]
    ]
    output_file = 'results_synth_minirocket.xlsx'
    results_df.to_excel(output_file, index=False)
    print(f'\nResults saved to {output_file}')

    if show_plot:
        signal = data['signals']['test_anomalous']
        anomaly_ranges = data['anomaly_ranges']
        window_starts = data['windows']['idx_test_anomalous']

        n_plot = min(len(signal), len(data['dataframes']['test_anomalous']))
        signal_plot = signal[:n_plot]

        window_pred_normal = (-scores) > th_optimal
        votes_normal = np.zeros(n_plot, dtype=float)
        votes_total = np.zeros(n_plot, dtype=float)

        for i, start in enumerate(window_starts):
            end = min(start + WINDOW_SIZE, n_plot)
            if end <= start:
                continue
            votes_total[start:end] += 1.0
            if window_pred_normal[i]:
                votes_normal[start:end] += 1.0

        pred_normal = np.zeros(n_plot, dtype=bool)
        valid = votes_total > 0
        pred_normal[valid] = votes_normal[valid] >= (0.5 * votes_total[valid])
        pred_anomaly = ~pred_normal

        plt.figure(figsize=(15, 5))
        plt.plot(signal_plot, color='gray', alpha=0.35, lw=1.0, label='Señal base')

        colors = {
            'phase_inversion': 'red',
            'amplitude_breakdown': 'orange',
        }
        labels_map = {
            'phase_inversion': 'Anomaly: Phase',
            'amplitude_breakdown': 'Anomaly: Amplitude',
        }

        plotted = set()
        for start, end, a_type in anomaly_ranges:
            plt.axvspan(
                start,
                end,
                color=colors.get(a_type, 'gray'),
                alpha=0.20,
                label=labels_map.get(a_type, a_type) if a_type not in plotted else None,
            )
            plotted.add(a_type)

        signal_pred_normal = signal_plot.copy()
        signal_pred_normal[pred_anomaly] = np.nan
        plt.plot(
            np.arange(n_plot),
            signal_pred_normal,
            color='green',
            lw=3.4,
            alpha=0.9,
            label='MiniRocket: detectado normal',
        )

        signal_pred_anomaly = signal_plot.copy()
        signal_pred_anomaly[pred_normal] = np.nan
        plt.plot(
            np.arange(n_plot),
            signal_pred_anomaly,
            color='red',
            lw=3.4,
            alpha=0.95,
            label='MiniRocket: detectado anómalo',
        )

        plt.title('Synthetic Test Signal with Anomalies (Slow Frequencies)')
        plt.xlabel('Sample index')
        plt.ylabel('Signal value')
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(15, 4))
        plt.plot(window_starts, scores, color='darkgreen', lw=1.8)
        plt.title('Window MiniRocket Score (Higher = More Anomalous)')
        plt.xlabel('Window start index')
        plt.ylabel('Distance to centroid')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    print('\n' + '=' * 70)
    print('SUMMARY - SYNTHETIC MINIROCKET')
    print('=' * 70)
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'AUPRC: {auprc:.4f}')
    print(f'Recall @ FPR<=1%: {recall_at_1pct:.4f}')
    print(f'Sensitivity: {sensitivity:.3f}')
    print(f'Specificity: {specificity:.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'F1-score: {f1:.3f}')

    return results


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':
    process_synthetic_minirocket(show_plot=True)
    input('\nPress ENTER to exit...')
