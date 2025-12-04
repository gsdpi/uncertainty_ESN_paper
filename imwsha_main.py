##################################################################
# Main script for IM-WSHA dataset processing with ESN
# and epistemic uncertainty estimation
##################################################################

import reservoirpy as rpy
import time

rpy.verbosity(0)
rpy.set_seed(42)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
from reservoirpy.nodes import Reservoir, Ridge, Input

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

# Processing parameters
NT = 7  # number of activities to train
WINDOW_LENGTH = 140  # L
STRIDE = 20  # S
SAMPLING_PERIOD = 1 / 20.  # tm

# Global model (trained once, reused for all subjects)
GLOBAL_ESN = None
GLOBAL_RESERVOIR = None
ESN_TRAINING_TIME = 0

##################################################################
# UTILITY FUNCTIONS
##################################################################

def train_esn_model(df_train, features):
    """
    Create and train ESN model (called once).
    
    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataframe
    features : list
        List of feature column names
        
    Returns
    -------
    esn_model : reservoirpy model
        Trained ESN model
    reservoir : reservoirpy.nodes.Reservoir
        Reservoir node
    training_time : float
        Training time in seconds
    """
    X_train = df_train[features].values.reshape(-1, len(features))
    Y_train = df_train['activity_label'].values.reshape(-1, 1)
    
    print('Creating ESN...')
    data = Input()
    reservoir = Reservoir(
        N_STATES, lr=LR, sr=RHO, input_scaling=INPUT_SCALE,
        rc_connectivity=SPARSITY,
        Win=rpy.mat_gen.bernoulli(input_scaling=WIN_SCALE)
    )
    readout = Ridge(ridge=RIDGE, input_bias=SET_BIAS)
    esn_model = data >> reservoir >> readout
    print(esn_model.node_names)
    
    print('Training ESN...')
    start_time = time.time()
    esn_model = esn_model.fit(X_train, Y_train, warmup=WARMUP)
    training_time = time.time() - start_time
    
    print(f'ESN training completed in {training_time:.2f} seconds')
    print(f'Reservoir: {reservoir.is_initialized}, '
          f'Readout: {readout.is_initialized}, Fitted: {readout.fitted}')
    
    return esn_model, reservoir, training_time


def process_subject(df, features, subject_label='Subject', r_values=[8]):
    """
    Process a single subject with pre-trained ESN.
    Generates results and plots.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataframe with all data
    features : list
        List of feature column names
    subject_label : str, optional
        Label for plots (e.g., 'Subject 1', 'Patient A'). Default: 'Subject'
    r_values : list, optional
        List of r values for uncertainty evaluation. Default: [8]
    """
    global GLOBAL_ESN, GLOBAL_RESERVOIR, ESN_TRAINING_TIME
    
    print(f'\n{"="*70}')
    print(f'PROCESSING: {subject_label}')
    print(f'{"="*70}\n')
    
    # Prepare training data
    train_activities = np.arange(1, NT + 1)
    
    df_train = pd.DataFrame()
    for aa in train_activities:
        df_tmp = df.loc[df['activity_label'] == aa]
        df_train = pd.concat([df_train, df_tmp[300:-200]])
    
    # Train ESN once (only on first subject)
    if GLOBAL_ESN is None:
        GLOBAL_ESN, GLOBAL_RESERVOIR, ESN_TRAINING_TIME = train_esn_model(df_train, features)
    
    # Get predictions
    print('\nRunning signals through ESN...')
    X = df[features].values
    Y = df['activity_label'].values
    Y_out = GLOBAL_ESN.run(X)
    Y_out = np.clip(Y_out, 0, 12)
    
    # Prepare time vector
    t = np.arange(len(df)).reshape(-1, 1) * SAMPLING_PERIOD
    t_adj = t[:Y_out.shape[0]]
    Y_adj = Y[:Y_out.shape[0]]
    
    # Process uncertainty for each r value
    
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
            reservoir=GLOBAL_RESERVOIR,
            r=r,
            window_length=WINDOW_LENGTH,
            stride=STRIDE,
            train_activities=train_activities,
            transition_window=WINDOW_LENGTH
        )
        kde_time = time.time() - start_time
        
        # Evaluate uncertainty
        start_time = time.time()
        logprobX_exp = evaluate_uncertainty_on_signal(
            df=df,
            features=features,
            reservoir=GLOBAL_RESERVOIR,
            kde_model=kde_model,
            r=r,
            window_length=WINDOW_LENGTH,
            stride=STRIDE
        )
        eval_time = time.time() - start_time
        
        print(f'Uncertainty model training: {kde_time:.2f}s')
        print(f'Uncertainty evaluation: {eval_time:.2f}s')
        
        # Calculate metrics (WARNING: Windows between activities are NOT excluded here)
        mask = np.isin(df['activity_label'], train_activities).astype(int)
        mask_ = mask[:len(logprobX_exp)]
        
        actual_labels = mask_
        roc_auc, th_optimal = calc_metrics(actual_labels, logprobX_exp, plot_roc=True)
        
        # Plot results
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
    
    print(f'\nESN training time (one-time): {ESN_TRAINING_TIME:.2f}s')


##################################################################
# EXAMPLE USAGE
##################################################################

if __name__ == '__main__':
    
    # Load and clean data (user does this manually)
    print('Loading data for Subject 1...')
    df = pd.read_csv('./IM-WSHA_Dataset/IMSHA_Dataset/Subject 1/3-imu-one subject.csv')
    
    # User cleans the data here
    df.loc[1150:1375, 'activity_label'] = 1
    df.loc[2390:2510, 'activity_label'] = 2
    df.loc[3300:3840, 'activity_label'] = 3
    df.loc[6000:6300, 'activity_label'] = 5
    df.loc[7200:7340, 'activity_label'] = 6
    df.loc[8400:8570, 'activity_label'] = 7
    df.loc[9675:9825, 'activity_label'] = 8
    df.loc[10900:11010, 'activity_label'] = 9
    
    features = df.keys()[1:].tolist()
    
    # Simple call to process
    process_subject(df, features, subject_label='Subject 1', r_values=[8])
    
    # Can call again with different subject (ESN reused)
    # process_subject(df2, features, subject_label='Subject 2')
    
    print('\n' + '='*70)
    print('Processing completed')
    print('='*70)
    
    plt.show()
