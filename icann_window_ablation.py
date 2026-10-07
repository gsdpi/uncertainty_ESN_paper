"""Evaluate ICANN ax uncertainty metrics across window lengths."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import reservoirpy as rpy

from esn_uncertainty import calc_metrics, evaluate_uncertainty_on_signal, train_uncertainty_model
from icann_esn import LATENT_DIMENSIONS, create_esn_model, train_esn_model
from icann_utils import (
    SAMPLING_PERIOD,
    STRIDE,
    WINDOW_LENGTH,
    create_label_mask,
    get_anomaly_experiments,
    get_normal_experiments,
    get_train_experiments,
    load_icann_data,
    prepare_test_df,
    prepare_train_df,
)


FEATURES = ['ax']
WINDOW_LENGTHS = [50, 100, 250, 500, 1000, 1500, 2000, 3000]
RANDOM_SEED = 42
DATA_PATH = Path('./dataicann/dataicann.mat')
RESULTS_PATH = Path('./results/icann_window_ablation.xlsx')
FIGURE_PATH = Path('./figures/icann_window_ablation.png')


def run_window_ablation():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'ICANN dataset not found: {DATA_PATH}')

    rpy.set_seed(RANDOM_SEED)
    df = load_icann_data(str(DATA_PATH))
    df_train = prepare_train_df(df)
    df_test = prepare_test_df(df)

    esn_model = create_esn_model()
    esn_model, _ = train_esn_model(esn_model, df_train, FEATURES)
    reservoir = esn_model.nodes[1]

    actual_labels = create_label_mask(
        df_test,
        get_normal_experiments(),
        get_anomaly_experiments(),
    )

    scores_by_window = {}
    for window_length in WINDOW_LENGTHS:
        print(f'\nEvaluating window length {window_length} samples...')
        kde_model = train_uncertainty_model(
            df_train=df_train,
            features=FEATURES,
            target_column='resistance',
            r=LATENT_DIMENSIONS,
            window_length=window_length,
            stride=STRIDE,
            train_activities=get_train_experiments(),
            reservoir=reservoir,
            transition_window=0,
        )
        if kde_model is None:
            raise RuntimeError(f'KDE training failed for window length {window_length}.')

        scores, _ = evaluate_uncertainty_on_signal(
            df=df_test,
            features=FEATURES,
            reservoir=reservoir,
            kde_model=kde_model,
            r=LATENT_DIMENSIONS,
            window_length=window_length,
            stride=STRIDE,
        )
        scores_by_window[window_length] = scores

    common_length = min(len(actual_labels), *(len(scores) for scores in scores_by_window.values()))
    labels_common = actual_labels[:common_length]
    result_rows = []
    for window_length, scores in scores_by_window.items():
        metrics = calc_metrics(labels_common, scores[:common_length], plot_roc=False)
        result_rows.append({
            'window_length_samples': window_length,
            'window_length_ms': window_length * SAMPLING_PERIOD * 1000,
            'stride_samples': STRIDE,
            'r': LATENT_DIMENSIONS,
            'roc_auc': metrics['roc_auc'],
            'auprc': metrics['auprc'],
            'f1_score': metrics['f1_score'],
            'threshold_test': metrics['threshold'],
            'evaluated_samples': common_length,
        })

    results = pd.DataFrame(result_rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_excel(RESULTS_PATH, index=False)
    print(f'\nAblation results saved to {RESULTS_PATH}')
    print('F1 uses the threshold optimized on this evaluation set, matching icann_esn.')

    figure, axis = plt.subplots(figsize=(10, 6))
    for metric, label in (
        ('roc_auc', 'ROC-AUC'),
        ('auprc', 'AUPRC'),
        ('f1_score', 'F1-score'),
    ):
        axis.plot(
            results['window_length_ms'],
            results[metric],
            marker='o',
            linewidth=2,
            label=label,
        )

    baseline_ms = WINDOW_LENGTH * SAMPLING_PERIOD * 1000
    axis.axvline(
        baseline_ms,
        color='black',
        linestyle='--',
        alpha=0.7,
        label=f'Current window ({baseline_ms:g} ms)',
    )
    axis.set_xlabel('Window length (ms)')
    axis.set_ylabel('Metric score')
    axis.set_ylim(0, 1.02)
    axis.set_title('ICANN ax: sensitivity to window length')
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    print(f'Ablation plot saved to {FIGURE_PATH}')

    if 'agg' not in plt.get_backend().lower():
        plt.show()

    return results


if __name__ == '__main__':
    run_window_ablation()