##################################################################
# Main script for IM-WSHA dataset processing with ESN
# and epistemic uncertainty estimation
##################################################################

import reservoirpy as rpy
import time

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

# Processing parameters
NT = 7  # number of activities to train
WINDOW_LENGTH = 140  # L
STRIDE = 20  # S
SAMPLING_PERIOD = 1 / 20.  # tm

# Global model (reused for all subjects)
GLOBAL_ESN = None
GLOBAL_RESERVOIR = None
ESN_TRAINING_TIME = 0

##################################################################
# UTILITY FUNCTIONS
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
    
    # Prepare training data
    train_activities = np.arange(1, NT + 1)
    
    df_train = pd.DataFrame()
    for aa in train_activities:
        df_tmp = df.loc[df['activity_label'] == aa]
        df_train = pd.concat([df_train, df_tmp[150:-150]])
    
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
            'subject': subject_label,
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


##################################################################
# DATA CLEANING
##################################################################

# Cleaning specifications for each subject
SUBJECT_CLEANING = {
    'Subject 1': [
        (0, 190, 12),
        (1150, 1340, 1),
        (1340, 1580, 12),
        (1580, 2500, 2),
        (2500, 2680, 12),
        (2680, 3725, 3),
        (3725, 3950, 12),
        (4800, 5085, 12),
        (6000, 6130, 5),
        (6130, 6280, 12),
        (6280, 7285, 6),
        (7285, 7358, 12),
        (7358, 8500, 7),
        (8500, 8590, 12),
        (8590, 9745, 8),
        (9745, 9945, 12),
        (10900, 10807, 9),
        (10807, 10938, 12),
        (11810, 11940, 12),
        #(11815, 12100, 12),
    ],
    'Subject 2': [
        (0, 200, 12),
        (1200, 1400, 12),
        (2400, 2550, 12),
        (3500, 3650, 3),
        (3650, 3850, 12),
        (3850, 4800, 4),
        (4800, 4950, 12),
        (4950, 6030, 5),
        (6030, 6150, 12),
        (6150, 7230, 6),
        (7230, 7300, 12),
        (7300, 8400, 7),
        (8400, 8500, 12),
        (8500, 9620, 8),
        (9620, 9800, 12),
        (9800, 10850, 9),
        (10850, 10930, 12),
        (10930, 11890, 10),
        (11890, 12080, 12),
        (12080, 12400, 11),
    ],
    'Subject 3': [
        (0, 200, 12),
        (1190, 1370, 12),
        (2385, 2600, 12),
        (3500, 3740, 3),
        (3740, 3790, 12),
        (4800, 5020, 12),
        (5020, 6000, 5),
        (6000, 6150, 12),
        (6150, 7170, 6),
        (7170, 7250, 12),
        (7250, 8350, 7),
        (8350, 8450, 12),
        (8450, 9650, 8),
        (9650, 9710, 12),
        (9710, 10800, 9),
        (10800, 10875, 12),
        (10875, 11820, 10),
        (11820, 11900, 12),
        (11900, 12000, 11),
    ],
    'Subject 4': [
        (0, 200, 12),
        (1200, 1400, 12),
        (2400, 2500, 12),
        (3200, 3700, 3),
        (3700, 3850, 12),
        (4800, 5000, 12),
        (5000, 6000, 5),
        (6000, 6150, 12),
        (6150, 7200, 6),
        (7200, 7300, 12),
        (7300, 8450, 7),
        (8450, 8510, 12),
        (8510, 9630, 8),
        (9630, 9750, 12),
        (9750, 10820, 9),
        (10820, 10935, 12),
        (10935, 11820, 10),
        (11820, 11940, 12),
        (11940, 12500, 11),
    ],
    'Subject 5': [
        (0, 200, 12),
        (200, 1200, 1),
        (1200, 1500, 12),
        (2380, 2550, 12),
        (2550, 3715, 3),
        (3715, 4000, 12),
        (4765, 5100, 12),
        (6000, 6150, 12),
        (7200, 7300, 12),
        (8500, 8590, 12),
        (9600, 9765, 12),
        (10800, 10890, 12),
        (10890, 11400, 10),
        (11750, 11850, 12),
        (11850, -1, 11),  # -1 represents end of dataframe
    ],
    'Subject 6': [
        (0, 200, 12),
        (1218, 1470, 12),
        (2390, 2530, 12),
        (3000, 3630, 3),
        (3630, 3800, 12),
        (4800, 5015, 12),
        (6050, 6200, 12),
        (7200, 7325, 12),
        (8400, 8485, 12),
        (8485, 9000, 8),
        (9650, 9770, 12),
        (10825, 10970, 12),
        (10970, 11500, 10),
        (11630, 11700, 12),
        (11700, 12300, 11),
    ],
    'Subject 7': [
        (0, 220, 12),
        (220, 1260, 1),
        (1260, 1430, 12),
        (2450, 2560, 12),
        (3000, 3690, 3),
        (3690, 3850, 12),
        (4810, 5050, 12),
        (6050, 6150, 12),
        (7220, 7350, 12),
        (8420, 8550, 12),
        (9650, 9720, 12),
        (10850, 11000, 12),
        (11000, 11500, 10),
        (11700, 11800, 12),
        (11800, 12300, 11),
    ],
    'Subject 8': [
        (0, 200, 12),
        (1250, 1460, 12),
        (2440, 2660, 12),
        (2660, 3700, 3),
        (3690, 4050, 12),
        (4850, 5200, 12),
        (6040, 6300, 12),
        (7250, 7330, 12),
        (8450, 8590, 12),
        (8590, 9760, 8),
        (9760, 9850, 12),
        (10900, 11000, 12),
        (11000, 11650, 10),
        (11650, 11750, 12),
        (11750, -1, 11),
    ],
    'Subject 9': [
        (0, 200, 12),
        (900, 1200, 1),
        (1200, 1500, 12),
        (2100, 2390, 2),
        (2390, 2560, 12),
        (3000, 3615, 3),
        (3615, 3880, 12),
        (4800, 5100, 12),
        (6000, 6150, 12),
        (7225, 7350, 12),
        (8320, 8506, 12),
        (9680, 9800, 12),
        (10830, 10930, 12),
        (10930, 11603, 10),
        (11603, 11656, 12),
        (11656, 12300, 11),
    ],
    'Subject 10': [
        (0, 200, 12),
        (1200, 1440, 12),
        (2380, 2600, 12),
        (2600, 3600, 3),
        (3600, 3950, 12),
        (4800, 5150, 12),
        (6000, 6240, 12),
        (7220, 7440, 12),
        (8360, 8505, 12),
        (9620, 9810, 12),
        (10800, 10965, 12),
        (11630, 11700, 12),
        (11700, -1, 11),
    ],
}


def clean_subject_data(df, subject_label):
    """
    Clean activity labels for a specific subject based on predefined ranges.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with activity data to clean
    subject_label : str
        Subject identifier (e.g., 'Subject 1')
        
    Returns
    -------
    df : pandas.DataFrame
        Cleaned dataframe with corrected activity labels
    """
    if subject_label not in SUBJECT_CLEANING:
        print(f'WARNING: No cleaning rules found for {subject_label}, returning unchanged data.')
        return df
    
    print(f'Cleaning activity labels for {subject_label}...')
    cleaning_rules = SUBJECT_CLEANING[subject_label]
    
    for start, end, label in cleaning_rules:
        if end == -1:
            end = len(df) - 1
        df.loc[start:end, 'activity_label'] = label
    
    print(f'Cleaning completed for {subject_label}')
    return df

def single_subject_example(esn_model):
    """
    Example for processing a single subject with manual data cleaning.
    """
    print('Loading data for Subject 1...')
    df = pd.read_csv('./IM-WSHA_Dataset/IMSHA_Dataset/Subject 1/3-imu-one subject.csv')
    df = df.dropna(subset=['activity_label'])

    # df.loc[1150:1375, 'activity_label'] = 1
    # df.loc[2390:2510, 'activity_label'] = 2
    # df.loc[3300:3840, 'activity_label'] = 3
    # df.loc[6000:6300, 'activity_label'] = 5
    # df.loc[7200:7340, 'activity_label'] = 6
    # df.loc[8400:8570, 'activity_label'] = 7
    # df.loc[9675:9825, 'activity_label'] = 8
    # df.loc[10900:11010, 'activity_label'] = 9
    df = clean_subject_data(df, 'Subject 1')

    features = df.keys()[1:].tolist()

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
    Process all subjects automatically.
    The ESN model is created outside and trained with the first subject.
    """
    import os, glob

    dataset_path = './IM-WSHA_Dataset/IMSHA_Dataset'
    subject_dirs = sorted([d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('Subject')])
    print(f'Found {len(subject_dirs)} subjects')

    all_results = []

    for subject_dir in subject_dirs:
        subject_path = os.path.join(dataset_path, subject_dir)
        csv_files = glob.glob(os.path.join(subject_path, '*.csv'))
        if not csv_files:
            print(f'\nWARNING: No CSV file found for {subject_dir}, skipping...')
            continue

        csv_file = csv_files[0]
        print(f'\nLoading data for {subject_dir} from {os.path.basename(csv_file)}...')
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['activity_label'])
        
        # Clean subject-specific data
        df = clean_subject_data(df, subject_dir)
        
        features = df.keys()[1:].tolist()

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
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  F1-score: {metrics['f1_score']:.3f}")

    # Return results for Excel export
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
        'roc_auc',
        'sensitivity',
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