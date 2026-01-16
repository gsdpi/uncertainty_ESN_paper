##################################################################
# Main script for ECG5000 dataset processing with ESN
# and epistemic uncertainty estimation
##################################################################

import reservoirpy as rpy
import time
import os

rpy.set_seed(42)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
from reservoirpy.nodes import Reservoir, Ridge, Input

from packaging.version import Version

from esn_uncertainty import train_uncertainty_model, evaluate_uncertainty_on_signal, calc_metrics
from ecg5000_utils import load_ecg5000_data, prepare_train_test_split, get_ecg_features

# Figure configuration
plt.rcParams.update({'font.size': 18})


##################################################################
# GLOBAL PARAMETERS (Fixed)
##################################################################

# ESN hyperparameters
N_STATES = 300
RHO = 0.9977765104808194
SPARSITY = 0.01
LR = 0.053814290145298004
WIN_SCALE = 0.744831763674846
INPUT_SCALE = 1
WARMUP = 5  # Reduced for short series
SET_BIAS = True
RIDGE = 4.6801882228427845e-08

# ECG-specific parameters
WINDOW_LENGTH = 140  # Full series length
STRIDE = 140  # Non-overlapping
SAMPLING_PERIOD = 1.0

# Global model
GLOBAL_ESN = None
GLOBAL_RESERVOIR = None
ESN_TRAINING_TIME = 0


##################################################################
# UTILITY FUNCTIONS (ESN and metrics)
##################################################################

def create_esn_model():
    """
    Create ESN model structure (without training).
    
    Returns
    -------
    esn_model : reservoirpy model
        Untrained ESN model
    reservoir : reservoirpy.nodes.Reservoir
        Reservoir node
    """
    print('Creating ESN architecture...')
    data = Input()
    reservoir = Reservoir(
        N_STATES, lr=LR, sr=RHO, input_scaling=INPUT_SCALE,
        rc_connectivity=SPARSITY,
        Win=rpy.mat_gen.bernoulli(input_scaling=WIN_SCALE)
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

    return esn_model, reservoir


def train_esn_model(esn_model, df_train, features):
    """
    Train ESN model with training data.
    
    Parameters
    ----------
    esn_model : reservoirpy model
        Untrained ESN model
    df_train : pandas.DataFrame
        Training dataframe
    features : list
        List of feature column names
        
    Returns
    -------
    esn_model : reservoirpy model
        Trained ESN model
    training_time : float
        Training time in seconds
    """
    X_train = df_train[features].values.reshape(-1, len(features))
    Y_train = df_train['label'].values.reshape(-1, 1)
    
    print('Training ESN...')
    start_time = time.time()
    esn_model = esn_model.fit(X_train, Y_train, warmup=WARMUP)
    training_time = time.time() - start_time
    
    print(f'ESN training completed in {training_time:.2f} seconds')
    
    return esn_model, training_time


def process_ecg5000(df_train, df_test, features, esn_model, reservoir,
                    r_values=[8], train_readout=False, show_roc_plot=False):
    """
    Process ECG5000 dataset with ESN model.
    
    Parameters
    ----------
    df_train : pandas.DataFrame
        Training data (normal samples only)
    df_test : pandas.DataFrame
        Test data (normal + anomalous)
    features : list
        List of feature column names
    esn_model : reservoirpy model
        Pre-trained ESN model
    reservoir : reservoirpy.nodes.Reservoir
        Reservoir node
    r_values : list, optional
        List of r values for uncertainty evaluation. Default: [8]
    train_readout : bool, optional
        If True, trains the readout layer. Default: False
    show_roc_plot : bool, optional
        If True, shows ROC curve. Default: False
        
    Returns
    -------
    results : dict
        Dictionary containing metrics for each r value
    """
    print(f'\n{"="*70}')
    print(f'PROCESSING: ECG5000 Dataset')
    print(f'{"="*70}\n')
    
    # Train readout if requested
    training_time = 0
    if train_readout:
        esn_model, training_time = train_esn_model(esn_model, df_train, features)
    
    # Process uncertainty for each r value
    results = {}
    
    print(f'\nEvaluating uncertainty with r values: {r_values}')
    
    for r in r_values:
        print(f'\n{"-"*70}')
        print(f'Processing with r={r}')
        print(f'{"-"*70}')
        
        # Train uncertainty model
        start_time = time.time()
        kde_model = train_uncertainty_model(
            df_train=df_train,
            features=features,
            target_column='label',
            r=r,
            window_length=WINDOW_LENGTH,
            stride=STRIDE,
            train_activities=[1.0],  # Only normal samples
            reservoir=reservoir,
            transition_window=WINDOW_LENGTH
        )
        kde_time = time.time() - start_time
        
        # Evaluate uncertainty
        if kde_model is None:
            print('KDE model training failed, skipping uncertainty evaluation.')
            logprobX_exp = np.zeros(len(df_test)) - 1
            eval_time = 0
            roc_auc = 0
            th_optimal = 0
            sensitivity = 0
            specificity = 0
            precision = 0
            f1 = 0
        else:
            start_time = time.time()
            logprobX_exp = evaluate_uncertainty_on_signal(
                df=df_test,
                features=features,
                reservoir=reservoir,
                kde_model=kde_model,
                r=r,
                window_length=WINDOW_LENGTH,
                stride=STRIDE
            )
            eval_time = time.time() - start_time
            
            # Calculate metrics
            # 1 = normal (train activities), 0 = anomaly
            mask = (df_test['label'] == 1.0).astype(int)

            # Align lengths to avoid shape mismatch (windowing shortens the score array)
            min_len = min(len(mask), len(logprobX_exp))
            actual_labels = mask[:min_len]
            scores_aligned = logprobX_exp[:min_len]

            metrics = calc_metrics(actual_labels, scores_aligned, plot_roc=False)
            
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

            if show_roc_plot and kde_model is not None:
                fpr, tpr, _ = roc_curve(actual_labels, logprobX_exp)
                roc_auc_plot = auc(fpr, tpr)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_plot:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - ECG5000 (ESN, r={r})')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
        
        results[r] = {
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
    
    return results


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':
    print('='*70)
    print('ECG5000 ESN Anomaly Detection with Uncertainty Estimation')
    print('='*70)
    print(f'\nESN Parameters:')
    print(f'  N_STATES: {N_STATES}')
    print(f'  RHO: {RHO}')
    print(f'  SPARSITY: {SPARSITY}')
    print(f'  LR: {LR}')
    print(f'  WIN_SCALE: {WIN_SCALE}')
    print(f'  RIDGE: {RIDGE}')
    print(f'  WARMUP: {WARMUP}')
    print(f'\nDataset Parameters:')
    print(f'  WINDOW_LENGTH: {WINDOW_LENGTH}')
    print(f'  STRIDE: {STRIDE}')
    
    # Load data
    print('\nLoading ECG5000 dataset...')
    df = load_ecg5000_data()
    df_train, df_test = prepare_train_test_split(df, train_ratio=0.75)
    
    features = get_ecg_features()
    print(f'Features: {len(features)} ECG channels')
    
    # Create ESN model
    esn_model, reservoir = create_esn_model()
    
    # Train ESN
    print('\nTraining ESN model...')
    start_time = time.time()
    esn_model, esn_train_time = train_esn_model(esn_model, df_train, features)
    ESN_TRAINING_TIME = esn_train_time
    
    # Process with different r values
    r_values = [8]
    results = process_ecg5000(
        df_train, df_test, features, esn_model, reservoir,
        r_values=r_values,
        train_readout=False,
        show_roc_plot=False
    )
    
    # Print summary
    print('\n' + '='*70)
    print('SUMMARY - ECG5000 ESN')
    print('='*70)
    
    summary_data = []
    for r, metrics in results.items():
        print(f'\nResults with r={r}:')
        print(f'  AUC: {metrics["roc_auc"]:.4f}')
        print(f'  AUPRC: {metrics["auprc"]:.4f}')
        print(f'  Recall @ FPR<=1%: {metrics["recall_at_1pct_fpr"]:.4f}')
        print(f'  Sensitivity: {metrics["sensitivity"]:.3f}')
        print(f'  Specificity: {metrics["specificity"]:.3f}')
        print(f'  Precision: {metrics["precision"]:.3f}')
        print(f'  F1-score: {metrics["f1_score"]:.3f}')
        
        summary_data.append({
            'r': r,
            'roc_auc': metrics['roc_auc'],
            'auprc': metrics['auprc'],
            'recall_at_1pct_fpr': metrics['recall_at_1pct_fpr'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score'],
            'threshold': metrics['threshold'],
            'kde_training_time': metrics['kde_training_time'],
            'evaluation_time': metrics['evaluation_time'],
        })
    
    # Save results
    results_df = pd.DataFrame(summary_data)
    output_file = 'results_ecg5000_esn.xlsx'
    results_df.to_excel(output_file, index=False)
    print(f'\nResults saved to {output_file}')
    
    print('\n' + '='*70)
    print('Processing completed!')
    print('='*70)
