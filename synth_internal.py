##################################################################
# Synthetic dataset: OOD detection from the INSTANTANEOUS reservoir
# state (no sliding windows) as a baseline for the SVD+KDE method.
# Detectors live in internal_detectors.py.
##################################################################
import time

import pandas as pd

from synth_esn import create_esn_model, train_esn_model, WARMUP, RANDOM_SEED
from synth_utils import N_POINTS, WINDOW_SIZE, STEP, get_cached_synthetic_dataset
from esn_uncertainty import calc_metrics
from internal_detectors import DETECTORS

FEATURES = ['signal']


def process_internal(detector='mahalanobis'):
    fit_fn, score_fn = DETECTORS[detector]

    data = get_cached_synthetic_dataset(
        n_points=N_POINTS, window_size=WINDOW_SIZE, step=STEP, random_seed=RANDOM_SEED,
    )
    df_train = data['dataframes']['train']
    df_test = data['dataframes']['test_anomalous']

    esn_model = create_esn_model()
    esn_model, esn_time = train_esn_model(esn_model, df_train, FEATURES, target_column='signal')
    reservoir = esn_model.nodes[1]  # data >> reservoir >> readout

    # Instantaneous reservoir states (washout discarded for training)
    states_train = reservoir.run(df_train[FEATURES].values)[WARMUP:]
    states_test = reservoir.run(df_test[FEATURES].values)

    start = time.time()
    model = fit_fn(states_train)
    fit_time = time.time() - start

    start = time.time()
    scores = score_fn(model, states_test)
    eval_time = time.time() - start

    # label: 1 = normal, 0 = anomaly
    metrics = calc_metrics(df_test['label'].values[:len(scores)], scores, plot_roc=False)
    print(f'{detector}: ' + ', '.join(f'{k}={v:.3f}' for k, v in metrics.items()))

    return {
        'detector': detector,
        'esn_training_time': esn_time,
        'fit_time': fit_time,
        'evaluation_time': eval_time,
        **metrics,
    }


if __name__ == '__main__':
    results = [process_internal(d) for d in DETECTORS]

    metrics_order = [
        'roc_auc', 'auprc', 'recall_at_1pct_fpr', 'sensitivity', 'specificity',
        'precision', 'f1_score', 'threshold', 'esn_training_time', 'fit_time', 'evaluation_time',
    ]
    df_results = pd.DataFrame(results).set_index('detector')[metrics_order]
    df_results.to_excel('results_synth_internal.xlsx', sheet_name='metrics')
    print('\nResults saved to results_synth_internal.xlsx')
