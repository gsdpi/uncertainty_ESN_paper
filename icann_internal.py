##################################################################
# DATAICANN dataset: OOD detection from the INSTANTANEOUS reservoir
# state (no sliding windows) as a baseline for the SVD+KDE method.
# Detectors live in internal_detectors.py.
##################################################################
import time

import numpy as np
import pandas as pd

from icann_esn import create_esn_model, train_esn_model, WARMUP
from esn_uncertainty import calc_metrics
from internal_detectors import DETECTORS
from icann_utils import (
    load_icann_data,
    get_features,
    get_normal_experiments,
    get_anomaly_experiments,
    create_label_mask,
    prepare_test_df,
    prepare_train_df,
)

##################################################################
# PIPELINE
##################################################################


def process_internal(df, esn_model, features, detector='mahalanobis', target='resistance'):
    print(f'\n{"=" * 70}\nPROCESSING {features} with detector "{detector}"\n{"=" * 70}\n')
    fit_fn, score_fn = DETECTORS[detector]

    df_train = prepare_train_df(df)
    df_test = prepare_test_df(df)

    esn_model, esn_time = train_esn_model(esn_model, df_train, features, target)
    reservoir = esn_model.nodes[1]  # data >> reservoir >> readout

    # Instantaneous reservoir states (washout discarded for training)
    states_train = reservoir.run(df_train[features].values)[WARMUP:]
    states_test = reservoir.run(df_test[features].values)

    start = time.time()
    model = fit_fn(states_train)
    fit_time = time.time() - start

    start = time.time()
    scores = score_fn(model, states_test)
    eval_time = time.time() - start

    # 1 = in-distribution, 0 = anomaly
    labels = create_label_mask(df_test, get_normal_experiments(), get_anomaly_experiments())
    metrics = calc_metrics(labels[:len(scores)], scores, plot_roc=False)

    print('Metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.3f}')

    return {
        'features': features,
        'detector': detector,
        'esn_training_time': esn_time,
        'fit_time': fit_time,
        'evaluation_time': eval_time,
        **metrics,
    }


if __name__ == '__main__':
    df = load_icann_data('./dataicann/dataicann.mat')
    print(f'Dataset loaded: {df.shape[0]} samples x {df.shape[1]} features')

    all_features = get_features()
    signal_configs = [
        (['ax'], 'ax'),
        (['ay'], 'ay'),
        (all_features, 'all_channels'),
    ]

    all_results = []
    for detector in DETECTORS:
        for features, label in signal_configs:
            esn_model = create_esn_model()  # fresh ESN per run
            res = process_internal(df, esn_model, features, detector=detector)
            all_results.append({'Detector': detector, 'Signal': label, **res})

    metrics_order = [
        'roc_auc', 'auprc', 'recall_at_1pct_fpr', 'sensitivity', 'specificity',
        'precision', 'f1_score', 'threshold', 'esn_training_time', 'fit_time', 'evaluation_time',
    ]
    df_results = pd.DataFrame(all_results).set_index(['Detector', 'Signal'])[metrics_order]
    df_results.to_excel('results_icann_internal.xlsx', sheet_name='metrics')
    print('\nResults saved to results_icann_internal.xlsx')
