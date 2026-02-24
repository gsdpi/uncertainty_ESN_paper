##################################################################
# Main script for IM-WSHA dataset processing with ESN
# and epistemic uncertainty estimation
##################################################################

import reservoirpy as rpy
import time
import os

# Removed in version v0.0.4
# rpy.verbosity(0)
rpy.set_seed(42)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
from reservoirpy.nodes import Reservoir, Ridge, Input

from packaging.version import Version

from esn_uncertainty import train_uncertainty_model, evaluate_uncertainty_on_signal, calc_metrics

# Figure configuration
plt.rcParams.update({'font.size': 18})

##################################################################
# GLOBAL PARAMETERS (Fixed across all subjects)
##################################################################

# ESN hyperparameters
N_STATES = 300
RHO = 0.9977765104808194
SPARSITY = 0.01
LR = 0.053814290145298004
WIN_SCALE = 0.744831763674846
INPUT_SCALE = 1
WARMUP = 20
SET_BIAS = True
RIDGE = 4.6801882228427845e-08


# IM-WSHA utilities (loading, cleaning, features, splits)
from imwsha_utils import (
    NT, WINDOW_LENGTH, STRIDE, SAMPLING_PERIOD, TRIM,
    load_subject_df, get_features, get_train_activities, prepare_train_df
)

# Global model (reused for all subjects)
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

    if( Version(rpy.__version__) >= Version("0.4")):
        readout = Ridge(ridge=RIDGE, fit_bias=SET_BIAS)
    else:
        readout = rpy.nodes.Ridge(ridge=RIDGE, input_bias=SET_BIAS)

    esn_model = data >> reservoir >> readout
    
    if( Version(rpy.__version__) < Version("0.4")):
        print(f'ESN created with nodes: {esn_model.node_names}')
    else:
        print(f'ESN created with nodes: {esn_model.nodes}')

    return esn_model


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
    Y_train = df_train['activity_label'].values.reshape(-1, 1)
    
    print('Training ESN...')
    start_time = time.time()
    esn_model = esn_model.fit(X_train, Y_train, warmup=WARMUP)
    training_time = time.time() - start_time
    
    print(f'ESN training completed in {training_time:.2f} seconds')
    
    return esn_model, training_time


def process_subject(df, features, esn_model, subject_label='Subject', 
                    r_values=[8], train_readout=False,
                    show_roc_plot=False):
    """
    Process a single subject with ESN model.
    Generates results and plots.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataframe with all data
    features : list
        List of feature column names
    esn_model : reservoirpy model
        Pre-trained ESN model (can be retrained if train_readout=True)
    subject_label : str, optional
        Label for plots (e.g., 'Subject 1', 'Patient A'). Default: 'Subject'
    r_values : list, optional
        List of r values for uncertainty evaluation. Default: [8]
    train_readout : bool, optional
        If True, trains the readout layer with this subject's data. Default: False
        and shows classification plot.
    show_roc_plot : bool, optional
        If True, shows the ROC curve with AUC. Default: False
        
    Returns
    -------
    results : dict
        Dictionary containing metrics and timing information for each r value
    """
    print(f'\n{"="*70}')
    print(f'PROCESSING: {subject_label}')
    print(f'{"="*70}\n')
    
    # Prepare training data using shared utilities
    train_activities = get_train_activities()
    df_train = prepare_train_df(df, train_activities, trim=TRIM)
    
    # Train readout if requested
    training_time = 0
    if train_readout:
        esn_model, training_time = train_esn_model(esn_model, df_train, features)
    
    # Get reservoir from model (it's the second node in the pipeline)
    reservoir = esn_model.nodes[1]  # data >> reservoir >> readout
   
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
            target_column='activity_label',
            r=r,
            window_length=WINDOW_LENGTH,
            stride=STRIDE,
            train_activities=train_activities,
            reservoir=reservoir,
            transition_window=WINDOW_LENGTH
        )
        kde_time = time.time() - start_time
        
        # Evaluate uncertainty
        if kde_model is None:
            print('KDE model training failed (no valid windows), skipping uncertainty evaluation.')
            logprobX_exp = np.zeros(Y_out.shape[0]//STRIDE) - 1
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
                df=df,
                features=features,
                reservoir=reservoir,
                kde_model=kde_model,
                r=r,
                window_length=WINDOW_LENGTH,
                stride=STRIDE
            )
            eval_time = time.time() - start_time
            
            # Calculate metrics
            mask = np.isin(df['activity_label'], train_activities).astype(int)
            mask_ = mask[:len(logprobX_exp)]
            
            actual_labels = mask_
            metrics = calc_metrics(actual_labels, logprobX_exp, plot_roc=False)
            
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
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(actual_labels, logprobX_exp)
                roc_auc_plot = auc(fpr, tpr)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_plot:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
                plt.grid()
                plt.tight_layout()
                plt.show()
        
        # Store results
        results[r] = {
            'subject': subject_label,
            'r': r,
            'esn_training_time': training_time,
            'kde_training_time': kde_time,
            'evaluation_time': eval_time,
            'roc_auc': roc_auc,            'auprc': auprc,
            'recall_at_1pct_fpr': recall_at_1pct,            'threshold': th_optimal,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1
        }

        # Plot classification results if requested
        if train_readout:
            # Get predictions
            print('\nRunning signals through ESN...')
            X = df[features].values
            Y = df['activity_label'].values
            Y_out = esn_model.run(X)
            Y_out = np.clip(Y_out, 0, 12)
            
            # Prepare time vector
            t = np.arange(len(df)).reshape(-1, 1) * SAMPLING_PERIOD
            t_adj = t[:Y_out.shape[0]]
            Y_adj = Y[:Y_out.shape[0]]
    
            plt.figure(figsize=(12, 8))
            
            # Subplot 1: IMU signals
            plt.subplot(2, 1, 1)
            plt.plot(t, X[:, 0:3] / np.max(np.abs(X[:, 0:3])) / 3)
            plt.plot(t, X[:, 3:6] / np.max(np.abs(X[:, 3:6])) / 3 + 1)
            plt.plot(t, X[:, 6:9] / np.max(np.abs(X[:, 6:9])) / 3 + 2)
            plt.grid()
            
            ax = plt.gca()
            ax.set_xlim(0, t_adj[-1])
            ax.set_yticks([0, 1, 2])
            ax.yaxis.set_ticklabels(["IMU1", "IMU2", "IMU3"])
            plt.yticks(rotation=90)
            
            plt.title(f'{subject_label} - IMU Signals')
            plt.ylabel('acceleration (normalized)')
            
            # Subplot 2: Activity classification
            plt.subplot(2, 1, 2)
            
            cc = np.array([1 if i > th_optimal else 0 for i in
                           logprobX_exp[::STRIDE][:len(logprobX_exp)//STRIDE]])
            cc_exp = np.kron(cc, np.ones(STRIDE))
            
            plt.plot(t_adj, Y_adj, label='Real', linewidth=2)
            
            mask_green = cc_exp == 0
            mask_red = cc_exp == 1
            
            def plot_segments(T, Y, mask, color):
                Y_aux = Y.copy()[:mask.shape[0]]
                Y_aux[mask] = np.nan
                plt.plot(T[:mask.shape[0]], Y_aux, color=color, alpha=0.8, linewidth=2)
            
            plot_segments(t_adj, Y_out, mask_red, 'red')
            plot_segments(t_adj, Y_out, mask_green, 'green')
            plt.plot(t_adj, Y_out, color='gray', alpha=0.3)
            plt.grid()
            
            ax = plt.gca()
            ax.set_xlim(0, t_adj[-1])
            ax.set_ylim(0, 12)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            plt.title(f'{subject_label} - Activity Classification (r={r}, AUC={roc_auc:.3f})')
            plt.xlabel('time (s)')
            plt.ylabel('activity class')
            
            plt.tight_layout()
            plt.show()
    
    return results


##################################################################
# EXAMPLE USAGE
##################################################################


def single_subject_example(esn_model):
    """
    Example for processing a single subject using shared loading/cleaning utilities.
    """
    dataset_path = './IM-WSHA_Dataset/IMSHA_Dataset'
    print('Loading data for Subject 1...')
    df = load_subject_df(dataset_path, 'Subject 1')
    features = get_features(df)

    results = process_subject(
        df, features, esn_model,
        subject_label='Subject 1',
        r_values=[8],
        train_readout=True,
        show_roc_plot=True
    )

    print('\n' + '='*70)
    print('RESULTS')
    print('='*70)
    for r, metrics in results.items():
        print(f"\nr={r}:")
        print(f"  ESN training time: {metrics['esn_training_time']:.2f}s")
        print(f"  KDE training time: {metrics['kde_training_time']:.2f}s")
        print(f"  Evaluation time: {metrics['evaluation_time']:.2f}s")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  Optimal threshold: {metrics['threshold']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  F1-score: {metrics['f1_score']:.3f}")

    return esn_model, results



def process_all_subjects(esn_model):
    """
    Process all subjects automatically using IM-WSHA utilities.
    """
    dataset_path = './IM-WSHA_Dataset/IMSHA_Dataset'
    subject_dirs = sorted([d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('Subject')])
    print(f'Found {len(subject_dirs)} subjects')
    all_results = []
    for subject_dir in subject_dirs:
        print(f'\nLoading data for {subject_dir}...')
        try:
            df = load_subject_df(dataset_path, subject_dir)
        except Exception as e:
            print(f'  ERROR: {e}')
            continue
        features = get_features(df)
        results = process_subject(
            df, features, esn_model,
            subject_label=subject_dir,
            r_values=[8],
            train_readout=False,
        )
        for r, metrics in results.items():
            all_results.append(metrics)
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    for metrics in all_results:
        print(f"\n{metrics['subject']} (r={metrics['r']}):")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        print(f"  Recall @ FPR<=1%: {metrics['recall_at_1pct_fpr']:.3f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  F1-score: {metrics['f1_score']:.3f}")
    return all_results


##################################################################
# MAIN EXECUTION
##################################################################

if __name__ == '__main__':
    # Create the ESN model once in main
    esn_model = create_esn_model()

    # Option 1: process single subject
    esn_model, _ = single_subject_example(esn_model)

    # Option 2: process all subjects
    all_results = process_all_subjects(esn_model)

    # Save results to Excel (rows=metrics, columns=subjects)
    metrics_order = [
        'roc_auc',        'auprc',
        'recall_at_1pct_fpr',        'sensitivity',
        'specificity',
        'precision',
        'f1_score',
        'threshold',
        'esn_training_time',
        'kde_training_time',
        'evaluation_time'
    ]
    df_results = pd.DataFrame(all_results)
    # Keep only metrics of interest and transpose
    df_metrics = df_results.set_index('subject')[metrics_order].T
    df_metrics.to_excel('results_imwsha.xlsx', sheet_name='metrics')
    print('\nResults saved to results_imwsha.xlsx')

    input('\nPress ENTER to close plots and exit...')