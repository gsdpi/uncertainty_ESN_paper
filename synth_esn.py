##################################################################
# Main script for synthetic signal processing with ESN
# and epistemic uncertainty estimation
##################################################################

import time

import reservoirpy as rpy
from packaging.version import Version
from reservoirpy.nodes import Reservoir, Ridge, Input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from esn_uncertainty import train_uncertainty_model, evaluate_uncertainty_on_signal, calc_metrics
from synth_utils import N_POINTS, WINDOW_SIZE, STEP, build_synthetic_dataset

# Removed in version v0.0.4
# rpy.verbosity(0)
rpy.set_seed(42)

# Figure configuration
plt.rcParams.update({'font.size': 18})


##################################################################
# GLOBAL PARAMETERS (copied from ICANN)
##################################################################

# ESN hyperparameters
N_STATES = 300
RHO = 0.99
SPARSITY = 0.01
LR = 0.27031482024950293
WIN_SCALE = 0.8696730804425951
INPUT_SCALE = 1
WARMUP = 20
SET_BIAS = True
RIDGE = 5.530826061879047e-08

# Uncertainty parameters (from synth_utils)
WINDOW_LENGTH = WINDOW_SIZE
STRIDE = STEP
LATENT_DIMENSIONS = 5

RANDOM_SEED = 42


##################################################################
# ESN UTILITY FUNCTIONS
##################################################################

def create_esn_model():
    """
    Create ESN model structure (without training).
    """
    print('Creating ESN architecture...')

    data = Input()
    reservoir = Reservoir(
        N_STATES,
        lr=LR,
        sr=RHO,
        input_scaling=INPUT_SCALE,
        rc_connectivity=SPARSITY,
        Win=rpy.mat_gen.bernoulli(input_scaling=WIN_SCALE),
    )

    if Version(rpy.__version__) >= Version("0.4"):
        readout = Ridge(ridge=RIDGE, fit_bias=SET_BIAS)
    else:
        readout = rpy.nodes.Ridge(ridge=RIDGE, input_bias=SET_BIAS)

    esn_model = data >> reservoir >> readout

    if Version(rpy.__version__) < Version("0.4"):
        print(f'ESN created with nodes: {esn_model.node_names}')
    else:
        print(f'ESN created with nodes: {esn_model.nodes}')

    return esn_model


def train_esn_model(esn_model, df_train, features, target_column='signal'):
    """
    Train ESN readout on normal synthetic data.
    """
    X_train = df_train[features].values.reshape(-1, len(features))
    Y_train = df_train[target_column].values.reshape(-1, 1)

    print('Training ESN...')
    start_time = time.time()
    esn_model.fit(X_train, Y_train, warmup=WARMUP)
    training_time = time.time() - start_time
    print(f'ESN training completed in {training_time:.2f} seconds')

    return esn_model, training_time


##################################################################
# PROCESSING
##################################################################

def process_synthetic_esn(show_plot: bool = True):
    """
    Process synthetic dataset with ESN uncertainty estimation.
    """
    print('=' * 70)
    print('Synthetic Signal ESN Uncertainty Estimation')
    print('=' * 70)
    print(f'\nParameters (ICANN-based):')
    print(f'  N_STATES: {N_STATES}')
    print(f'  RHO: {RHO}')
    print(f'  SPARSITY: {SPARSITY}')
    print(f'  LR: {LR}')
    print(f'  WIN_SCALE: {WIN_SCALE}')
    print(f'  WARMUP: {WARMUP}')
    print(f'  RIDGE: {RIDGE}')
    print(f'  WINDOW_LENGTH: {WINDOW_LENGTH}')
    print(f'  STRIDE: {STRIDE}')
    print(f'  LATENT_DIMENSIONS (r): {LATENT_DIMENSIONS}')

    print('\nBuilding synthetic dataset...')
    data = build_synthetic_dataset(
        n_points=N_POINTS,
        window_size=WINDOW_LENGTH,
        step=STRIDE,
        random_seed=RANDOM_SEED,
    )

    df_train = data['dataframes']['train'].copy()
    df_test = data['dataframes']['test_anomalous'].copy()

    # Add constant train segment label to avoid transition masking issues
    # in train_uncertainty_model for continuous-valued targets.
    df_train['segment_label'] = 1

    features = ['signal']

    print(f'Training samples: {len(df_train)}')
    print(f'Test samples: {len(df_test)}')
    print('Test labels distribution (1=normal,0=anomaly):')
    print(df_test['label'].value_counts().sort_index())

    esn_model = create_esn_model()

    # Train readout with normal data only
    esn_model, esn_training_time = train_esn_model(
        esn_model,
        df_train,
        features=features,
        target_column='signal',
    )

    reservoir = esn_model.nodes[1]

    print(f'\nTraining uncertainty model (r={LATENT_DIMENSIONS})...')
    start_time = time.time()
    kde_model = train_uncertainty_model(
        df_train=df_train,
        features=features,
        target_column='segment_label',
        r=LATENT_DIMENSIONS,
        window_length=WINDOW_LENGTH,
        stride=STRIDE,
        train_activities=[1],
        reservoir=reservoir,
        transition_window=0,
    )
    kde_time = time.time() - start_time

    if kde_model is None:
        raise RuntimeError('KDE model training failed: no valid windows for synthetic train set.')

    print('\nEvaluating uncertainty on anomalous synthetic test signal...')
    start_time = time.time()
    logprob = evaluate_uncertainty_on_signal(
        df=df_test,
        features=features,
        reservoir=reservoir,
        kde_model=kde_model,
        r=LATENT_DIMENSIONS,
        window_length=WINDOW_LENGTH,
        stride=STRIDE,
    )
    eval_time = time.time() - start_time

    actual_labels = df_test['label'].values[: len(logprob)]
    metrics = calc_metrics(actual_labels, logprob, plot_roc=False)

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
        'Method': 'ESN',
        'dataset': 'synthetic_signal',
        'n_states': N_STATES,
        'rho': RHO,
        'sparsity': SPARSITY,
        'lr': LR,
        'win_scale': WIN_SCALE,
        'ridge': RIDGE,
        'warmup': WARMUP,
        'window_length': WINDOW_LENGTH,
        'stride': STRIDE,
        'r': LATENT_DIMENSIONS,
        'esn_training_time': esn_training_time,
        'kde_training_time': kde_time,
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
        'esn_training_time',
        'kde_training_time',
        'evaluation_time',
    ]

    results_df = pd.DataFrame([results])
    results_df = results_df[
        metrics_order
        + [
            'Method', 'dataset', 'n_states', 'rho', 'sparsity',
            'lr', 'win_scale', 'ridge', 'warmup',
            'window_length', 'stride', 'r'
        ]
    ]
    output_file = 'results_synth_esn.xlsx'
    results_df.to_excel(output_file, index=False)
    print(f'\nResults saved to {output_file}')

    if show_plot:
        signal = data['signals']['test_anomalous']
        anomaly_ranges = data['anomaly_ranges']
        n_plot = min(len(signal), len(logprob))

        signal_plot = signal[:n_plot]
        score_plot = logprob[:n_plot]
        pred_normal = score_plot > th_optimal
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
            label='ESN: detectado normal',
        )

        signal_pred_anomaly = signal_plot.copy()
        signal_pred_anomaly[pred_normal] = np.nan
        plt.plot(
            np.arange(n_plot),
            signal_pred_anomaly,
            color='red',
            lw=3.4,
            alpha=0.95,
            label='ESN: detectado anómalo',
        )

        plt.title('Synthetic Test Signal with Anomalies (Slow Frequencies)')
        plt.xlabel('Sample index')
        plt.ylabel('Signal value')
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(15, 4))
        t = np.arange(len(logprob))
        plt.plot(t, logprob, color='purple', lw=1.8)
        plt.axhline(th_optimal, color='black', linestyle='--', alpha=0.7, label='Optimal threshold')
        plt.title('ESN Uncertainty Score (Higher = More Normal)')
        plt.xlabel('Sample index')
        plt.ylabel('Log-likelihood score')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    print('\n' + '=' * 70)
    print('SUMMARY - SYNTHETIC ESN')
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
    process_synthetic_esn(show_plot=True)
    input('\nPress ENTER to exit...')
