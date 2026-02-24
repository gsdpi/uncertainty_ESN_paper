##################################################################
# Main script for synthetic signal processing with KNN
# anomaly detection (no ESN reservoir)
##################################################################

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

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

N_NEIGHBORS = 10
KNN_METRIC = "euclidean"
SCORE_REDUCTION = "mean"  # 'mean', 'max', 'median'
RANDOM_SEED = 42


##################################################################
# KNN ANOMALY DETECTOR
##################################################################

class KNNAnomalyDetector:
    """
    KNN-based anomaly detector for synthetic windowed signals.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        metric: str = "euclidean",
        score_reduction: str = "mean",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.score_reduction = score_reduction
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
        )

    def fit(self, X_train: np.ndarray):
        """
        Fit the KNN model using normal training windows.
        """
        print(f'Fitting KNN model with {len(X_train)} windows...')
        self.knn.fit(X_train)
        print('KNN model fitted.')
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Higher score => more anomalous.
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

def process_synthetic_knn(show_plot: bool = True):
    """
    Process synthetic dataset using KNN anomaly detection.
    """
    print('=' * 70)
    print('Synthetic Signal KNN Anomaly Detection')
    print('=' * 70)
    print(f'\nParameters:')
    print(f'  N_NEIGHBORS: {N_NEIGHBORS}')
    print(f'  KNN_METRIC: {KNN_METRIC}')
    print(f'  SCORE_REDUCTION: {SCORE_REDUCTION}')
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

    print('\nTraining KNN anomaly detector...')
    start_time = time.time()

    detector = KNNAnomalyDetector(
        n_neighbors=N_NEIGHBORS,
        metric=KNN_METRIC,
        score_reduction=SCORE_REDUCTION,
    )

    detector.fit(X_train)
    knn_time = time.time() - start_time
    print(f'KNN training completed in {knn_time:.3f} seconds')

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
        'Method': 'KNN',
        'dataset': 'synthetic_signal',
        'n_neighbors': N_NEIGHBORS,
        'knn_metric': KNN_METRIC,
        'score_reduction': SCORE_REDUCTION,
        'n_points': N_POINTS,
        'window_size': WINDOW_SIZE,
        'step': STEP,
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
        'evaluation_time',
    ]

    results_df = pd.DataFrame([results])
    results_df = results_df[
        metrics_order + [
            'Method', 'dataset', 'n_neighbors', 'knn_metric', 'score_reduction',
            'n_points', 'window_size', 'step'
        ]
    ]
    output_file = 'results_synth_knn.xlsx'
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
            label='KNN: detectado normal',
        )

        signal_pred_anomaly = signal_plot.copy()
        signal_pred_anomaly[pred_normal] = np.nan
        plt.plot(
            np.arange(n_plot),
            signal_pred_anomaly,
            color='red',
            lw=3.4,
            alpha=0.95,
            label='KNN: detectado anómalo',
        )

        plt.title('Synthetic Test Signal with Anomalies (Slow Frequencies)')
        plt.xlabel('Sample index')
        plt.ylabel('Signal value')
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(15, 4))
        plt.plot(window_starts, scores, color='teal', lw=1.8)
        plt.title('Window KNN Score (Higher = More Anomalous)')
        plt.xlabel('Window start index')
        plt.ylabel('KNN distance score')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    print('\n' + '=' * 70)
    print('SUMMARY - SYNTHETIC KNN')
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
    process_synthetic_knn(show_plot=True)
    input('\nPress ENTER to exit...')
