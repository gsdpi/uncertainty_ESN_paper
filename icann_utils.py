"""
Utilities for the ICANN dataset: loading, features, and splits.
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat

# Dataset-level parameters
WINDOW_LENGTH = 1000
STRIDE = 200
SAMPLING_PERIOD = 1 / 5000.

# Mapping of experiment IDs to resistance values (in ohms)
EXPERIMENT_MAPPING = {
    2: 0,
    6: 5,
    3: 10,
    4: 15,
    5: 20,
    7: np.nan,      # normal operation (no resistance label)
    8: np.nan,      # normal operation (no resistance label)
    0: np.nan,      # anomalies
    1: np.nan,      # anomalies
}

# Training experiments (normal operation)
TRAIN_EXPERIMENTS = [2, 6, 3, 4, 5]

# Test anomalies (to detect)
ANOMALY_EXPERIMENTS = [0, 1]

#NORMAL_EXPERIMENTS = TRAIN_EXPERIMENTS + [7, 8]
NORMAL_EXPERIMENTS = [7, 8]


def load_icann_data(mat_file_path='./dataicann/dataicann.mat'):
    """
    Load ICANN dataset from .mat file and return as pandas DataFrame.
    
    Parameters
    ----------
    mat_file_path : str
        Path to the dataicann.mat file
        
    Returns
    -------
    df : pandas.DataFrame
        Dataframe with columns: 'experiment', 'ac', 'ax', 'ay', 'ir', 'is', 'resistance'
    """
    d = loadmat(mat_file_path)
    
    # Create dataframe with all signals
    ohm = [EXPERIMENT_MAPPING[exp] for exp in EXPERIMENT_MAPPING.keys()]
    exp_ids = list(EXPERIMENT_MAPPING.keys())
    
    X = []
    Y = []
    X_label = []
    
    for i, exp_id in enumerate(exp_ids):
        paq = d['z'][0][exp_id]
        X.append(paq)
        Y.append(np.repeat(float(ohm[i]), paq.shape[0]))
        X_label.append(np.repeat(exp_id, paq.shape[0]))
    
    X = np.vstack(X)
    Y = np.vstack([np.array(y, dtype=float).reshape(-1, 1) for y in Y])
    X_label = np.vstack([np.array(xl, dtype=int).reshape(-1, 1) for xl in X_label])
    
    df = pd.DataFrame(X, columns=["ac", "ax", "ay", "ir", "is"])
    df.insert(0, 'experiment', X_label.flatten())
    df['resistance'] = Y.flatten()
    
    return df


def get_features():
    """
    Return list of feature columns for ICANN dataset.
    
    Returns
    -------
    features : list
        List of accelerometer feature columns
    """
    #return ["ac", "ax", "ay", "ir", "is"]
    # We will return only the two accelerometer channels that contain information
    return ["ax", "ay"]


def get_train_experiments():
    """
    Return the training experiment IDs (normal operation).
    
    Returns
    -------
    experiments : list
        List of training experiment IDs
    """
    return TRAIN_EXPERIMENTS


def get_anomaly_experiments():
    """
    Return the anomaly experiment IDs to detect.
    
    Returns
    -------
    experiments : list
        List of anomaly experiment IDs
    """
    return ANOMALY_EXPERIMENTS

def get_normal_experiments():
    """
    Return the normal experiment IDs (training + normal test).
    
    Returns
    -------
    experiments : list
        List of normal experiment IDs
    """
    return NORMAL_EXPERIMENTS


def prepare_train_df(df, train_experiments=None, trim=None):
    """
    Return a dataframe with only training experiments.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset
    train_experiments : list, optional
        List of experiment IDs to use for training. If None, uses TRAIN_EXPERIMENTS.
    trim : int, optional
        Number of samples to trim at the start and end of each experiment.
        If None, no trimming is applied.
    
    Returns
    -------
    df_train : pandas.DataFrame
        Training dataframe
    """
    if train_experiments is None:
        train_experiments = TRAIN_EXPERIMENTS
    
    df_train = pd.DataFrame()
    for exp_id in train_experiments:
        df_exp = df[df['experiment'] == exp_id]
        
        if trim is not None and trim > 0:
            df_exp = df_exp.iloc[trim:-trim]
        
        df_train = pd.concat([df_train, df_exp], ignore_index=True)
    
    return df_train


def prepare_test_df(df, test_experiments=None, trim=None):
    """
    Return a dataframe with only test experiments.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset
    test_experiments : list, optional
        List of experiment IDs to use for testing. If None, uses ANOMALY_EXPERIMENTS and NORMAL_EXPERIMENTS.
    trim : int, optional
        Number of samples to trim at the start and end of each experiment.
        If None, no trimming is applied.
    
    Returns
    -------
    df_test : pandas.DataFrame
        Test dataframe
    """
    if test_experiments is None:
        test_experiments = get_normal_experiments()+get_anomaly_experiments()
    
    df_test = pd.DataFrame()
    for exp_id in test_experiments:
        df_exp = df[df['experiment'] == exp_id]
        
        if trim is not None and trim > 0:
            df_exp = df_exp.iloc[trim:-trim]
        
        df_test = pd.concat([df_test, df_exp], ignore_index=True)
    
    return df_test


def create_label_mask(df, normal_experiments=None, anomaly_experiments=None):
    """
    Create a binary mask: 1 for normal (training) experiments, 0 for anomalies.
    This matches the convention used in IM-WSHA dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset
    train_experiments : list, optional
        List of training experiment IDs
    anomaly_experiments : list, optional
        List of anomaly experiment IDs
    
    Returns
    -------
    mask : numpy.ndarray
        Binary mask (1=normal, 0=anomaly)
    """
    if normal_experiments is None:
        normal_experiments = NORMAL_EXPERIMENTS
    if anomaly_experiments is None:
        anomaly_experiments = ANOMALY_EXPERIMENTS
    
    # Create mask with 1 for normal (training) experiments
    mask = np.isin(df['experiment'], normal_experiments).astype(int)
    return mask
