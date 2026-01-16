##################################################################
# Utility functions for ECG5000 dataset
##################################################################

import pandas as pd
import os


##################################################################
# DATASET LOADING AND PREPARATION
##################################################################

def load_ecg5000_data(base_path: str = "./ECG5000"):
    """
    Load ECG5000 dataset from train and test files.
    
    Parameters
    ----------
    base_path : str
        Path to ECG5000 directory
        
    Returns
    -------
    df : pandas.DataFrame
        Combined dataset with label and 140 features (columns 1-140)
    """
    train_path = os.path.join(base_path, "ECG5000_TRAIN.txt")
    test_path = os.path.join(base_path, "ECG5000_TEST.txt")
    
    df_train = pd.read_csv(train_path, sep=r'\s+', header=None)
    df_test = pd.read_csv(test_path, sep=r'\s+', header=None)
    
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.rename(columns={0: "label"})
    
    return df


def prepare_train_test_split(df: pd.DataFrame, train_ratio: float = 0.75):
    """
    Split dataset: train on 70-80% of normal samples, test on all data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset
    train_ratio : float
        Ratio of normal samples to use for training (default 0.75 = 75%)
        
    Returns
    -------
    df_train : pandas.DataFrame
        Training dataframe (normal samples only)
    df_test : pandas.DataFrame
        Test dataframe (all samples)
    """
    # Separate normal and anomalous samples
    normal_mask = df['label'] == 1.0
    df_normal = df[normal_mask].reset_index(drop=True)
    df_anomaly = df[~normal_mask].reset_index(drop=True)
    
    print(f"Total samples: {len(df)}")
    print(f"Normal samples: {len(df_normal)}")
    print(f"Anomalous samples: {len(df_anomaly)}")
    
    # Split normal samples for training
    n_train = int(len(df_normal) * train_ratio)
    df_train = df_normal.iloc[:n_train].reset_index(drop=True)
    df_test_normal = df_normal.iloc[n_train:].reset_index(drop=True)
    
    # Test set contains remaining normal + all anomalous
    df_test = pd.concat([df_test_normal, df_anomaly], ignore_index=True)
    
    print(f"\nTrain set (normal only): {len(df_train)}")
    print(f"Test set - normal: {len(df_test_normal)}, anomalous: {len(df_anomaly)}")
    
    return df_train, df_test


def get_ecg_features():
    """Get list of ECG feature columns (columns 1-140)."""
    return [col for col in range(1, 141)]
