##################################################################
# Main script for DATAICANN dataset processing with ESN
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
# GLOBAL PARAMETERS (Fixed across all cases)
##################################################################

# ESN hyperparameters
N_STATES = 300
RHO = 0.99 #1.270074061545781 
SPARSITY = 0.01
LR = 0.27031482024950293
WIN_SCALE = 0.8696730804425951
INPUT_SCALE = 1
WARMUP = 20
SET_BIAS = True
RIDGE = 5.530826061879047e-08

# Windowing parameters
WINDOW_LENGTH = 1000
STRIDE = 200   
SAMPLING_PERIOD = 1 / 5000.

# Global model (reused for all cases)
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

def train_esn_model(esn_model, df_train, features, target_column='resistance'):
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
    Y_train = df_train[target_column].values.reshape(-1, 1)

    print('Training ESN...')
    start_time = time.time()
    esn_model.fit(X_train, Y_train, warmup=WARMUP)
    training_time = time.time() - start_time
   
    print(f'ESN training completed in {training_time:.2f} seconds')
    
    return esn_model, training_time


def process_vibration(df, esn_model, features, r_values=[20], train_readout=False,
                    show_roc_plot=False, target='resistance'):


    print(f'\n{"="*70}')
    print(f'PROCESSING ACCELEROMETER: {features}')
    print(f'{"="*70}\n')
    
    train_activities   = [2,6,3,4,5]
    df_train_parts = []
    for exp_id in train_activities:
        df_exp = df[df['experiment'] == exp_id]
        df_train_parts.append(df_exp.head(10000))
    df_train = pd.concat(df_train_parts, ignore_index=True)

    # Train readout if requested
    training_time = 0
    if train_readout:
        esn_model, training_time = train_esn_model(esn_model, df_train, features, target)

    
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
        #esn_model.run(df_train['ax'].values.reshape(-1,1))  # Warm-up reservoir
        
        kde_model = train_uncertainty_model(
            df_train=df_train,
            features=features,
            target_column='resistance',
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
            mask = np.isin(df['experiment'], train_activities+[7,8]).astype(int)
            mask_ = mask[:len(logprobX_exp)]
            
            actual_labels = mask_
            metrics = calc_metrics(actual_labels, logprobX_exp, plot_roc=False)
            
            roc_auc = metrics['roc_auc']
            th_optimal = metrics['threshold']
            sensitivity = metrics['sensitivity']
            specificity = metrics['specificity']
            precision = metrics['precision']
            f1 = metrics['f1_score']
            
            print(f'\nMetrics:')
            print(f'  AUC: {roc_auc:.3f}')
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
            'features': features,
            'r': r,
            'esn_training_time': training_time,
            'kde_training_time': kde_time,
            'evaluation_time': eval_time,
            'roc_auc': roc_auc,
            'threshold': th_optimal,
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
            Y = df[target].values
            Y_out = esn_model.run(X)
                        
            # Prepare time vector
            t = np.arange(len(df)).reshape(-1, 1) * SAMPLING_PERIOD
            t_adj = t[:Y_out.shape[0]]
            Y_adj = Y[:Y_out.shape[0]]
    
            plt.figure(figsize=(12, 8))
            
            # Subplot 1: acceleration signals
            plt.subplot(2, 1, 1)
            n_features = X.shape[1]
            colors = ['C0', 'C1', 'C2']
            for j in range(n_features):
                offset = j * 1.0
                plt.plot(t, X[:, j] / np.max(np.abs(X[:, j])) / 3 + offset, color=colors[j % len(colors)])
            plt.grid()
            
            ax = plt.gca()
            ax.set_xlim(0, t_adj[-1])
            ax.set_yticks(np.arange(n_features))
            ax.yaxis.set_ticklabels(features[:n_features])
            plt.yticks(rotation=90)
            
            plt.title('Acceleration Signals')
            plt.ylabel('acceleration (g)')
            
            # Subplot 2: Resistance
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
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            plt.title(f'{target} - Resistance estimation (r={r}, AUC={roc_auc:.3f})')
            plt.xlabel('time (s)')
            plt.ylabel('resistance (ohms)')
            
            plt.tight_layout()
            plt.show()
    
    return results




##################################################################
# MAIN EXECUTION
##################################################################

if __name__ == '__main__':
    from scipy.io import loadmat

    # Read data
    PATH = './dataicann/'
    d = loadmat(PATH+'dataicann.mat')

    # Create dataframe with all signals
    ohm     = [0, 5, 10, 15, 20] + [np.nan, np.nan, np.nan, np.nan]
    exp   = [2, 6, 3, 4, 5] + [7, 8, 0, 1]
    X = []
    Y = []
    X_label = []

    for i in range(len(exp)):
        paq = d['z'][0][exp[i]]
        X.append(paq)
        Y.append(np.repeat(float(ohm[i]), paq.shape[0]))
        X_label.append(np.repeat(exp[i], paq.shape[0]))

    X = np.vstack(X)
    Y = np.vstack([np.array(y, dtype=float).reshape(-1, 1) for y in Y])
    X_label = np.vstack([np.array(xl, dtype=int).reshape(-1, 1) for xl in X_label])
    df = pd.DataFrame(X, columns=["ac", "ax", "ay", "ir", "is"])
    df.insert(0, 'experiment', X_label.flatten())
    df['resistance'] = Y.flatten()

    # Create the ESN model once in main
    esn_model = create_esn_model()
    process_vibration(df,esn_model,['ax'],train_readout=True,show_roc_plot=True)

    # Option 1: process single subject
    #esn_model, _ = single_subject_example(esn_model)

    # # Option 2: process all subjects
    # all_results = process_all_subjects(esn_model)

    # # Save results to Excel (rows=metrics, columns=subjects)
    # metrics_order = [
    #     'roc_auc',
    #     'sensitivity',
    #     'specificity',
    #     'precision',
    #     'f1_score',
    #     'threshold',
    #     'esn_training_time',
    #     'kde_training_time',
    #     'evaluation_time'
    # ]
    # df_results = pd.DataFrame(all_results)
    # # Keep only metrics of interest and transpose
    # df_metrics = df_results.set_index('experiment')[metrics_order].T
    # df_metrics.to_excel('results_icann.xlsx', sheet_name='metrics')
    # print('\nResults saved to results_icann.xlsx')

    input('\nPress ENTER to close plots and exit...')



