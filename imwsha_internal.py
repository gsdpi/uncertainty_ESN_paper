##################################################################
# IM-WSHA dataset: OOD detection from the INSTANTANEOUS reservoir
# state (no sliding windows) as a baseline for the SVD+KDE method.
# Detectors live in internal_detectors.py.
##################################################################
import os
import time

import numpy as np
import pandas as pd

from imwsha_esn import create_esn_model, train_esn_model, WARMUP
from esn_uncertainty import calc_metrics
from internal_detectors import DETECTORS
from imwsha_utils import (
    TRIM,
    load_subject_df,
    get_features,
    get_train_activities,
    prepare_train_df,
)

DATASET_PATH = './IM-WSHA_Dataset/IMSHA_Dataset'
TRAIN_SUBJECT = 'Subject 1'  # same as imwsha_esn: ESN and detector fitted once, applied to all subjects


def evaluate_subject(df, features, reservoir, fit_model, score_fn, subject_label, detector):
    states = reservoir.run(df[features].values)

    start = time.time()
    scores = score_fn(fit_model, states)
    eval_time = time.time() - start

    # 1 = seen activity (in-distribution), 0 = unseen
    labels = np.isin(df['activity_label'], get_train_activities()).astype(int)
    metrics = calc_metrics(labels[:len(scores)], scores, plot_roc=False)

    print(f'{subject_label}: ' + ', '.join(f'{k}={v:.3f}' for k, v in metrics.items()))
    return {'subject': subject_label, 'detector': detector, 'evaluation_time': eval_time, **metrics}


if __name__ == '__main__':
    train_activities = get_train_activities()

    print(f'Loading {TRAIN_SUBJECT} for training...')
    df_ref = load_subject_df(DATASET_PATH, TRAIN_SUBJECT)
    features = get_features(df_ref)
    df_train = prepare_train_df(df_ref, train_activities, trim=TRIM)

    esn_model = create_esn_model()
    esn_model, esn_time = train_esn_model(esn_model, df_train, features)
    reservoir = esn_model.nodes[1]  # data >> reservoir >> readout

    # Instantaneous reservoir states (washout discarded)
    states_train = reservoir.run(df_train[features].values)[WARMUP:]

    subject_dirs = sorted(d for d in os.listdir(DATASET_PATH)
                          if os.path.isdir(os.path.join(DATASET_PATH, d)) and d.startswith('Subject'))

    all_results = []
    for detector, (fit_fn, score_fn) in DETECTORS.items():
        start = time.time()
        model = fit_fn(states_train)
        fit_time = time.time() - start

        for subject_dir in subject_dirs:
            try:
                df = load_subject_df(DATASET_PATH, subject_dir)
            except Exception as e:
                print(f'  ERROR loading {subject_dir}: {e}')
                continue
            res = evaluate_subject(df, get_features(df), reservoir, model, score_fn, subject_dir, detector)
            all_results.append({**res, 'esn_training_time': esn_time, 'fit_time': fit_time})

    metrics_order = [
        'roc_auc', 'auprc', 'recall_at_1pct_fpr', 'sensitivity', 'specificity',
        'precision', 'f1_score', 'threshold', 'esn_training_time', 'fit_time', 'evaluation_time',
    ]
    df_results = pd.DataFrame(all_results).set_index(['detector', 'subject'])[metrics_order]
    os.makedirs('results', exist_ok=True)
    df_results.to_excel('results/results_imwsha_internal.xlsx', sheet_name='metrics')
    print('\nResults saved to results/results_imwsha_internal.xlsx')
