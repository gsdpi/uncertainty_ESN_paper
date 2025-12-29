"""
Utilities for the IM-WSHA dataset: cleaning, loading, features, and splits.
"""
import os
import glob
import numpy as np
import pandas as pd

# Dataset-level parameters
NT = 7  # number of training activities
WINDOW_LENGTH = 140
STRIDE = 20
SAMPLING_PERIOD = 1 / 20.

# Cleaning rules per subject (copied from imwsha_main.py)
SUBJECT_CLEANING = {
    'Subject 1': [
        (0, 200, 12),
        (1150, 1375, 1),
        (2390, 2510, 2),
        (3300, 3840, 3),
        (3840, 4000, 12),
        (4000, 4800, 4),
        (4800, 4950, 12),
        (4950, 6050, 5),
        (6050, 6300, 12),
        (6300, 7350, 6),
        (7350, 7500, 12),
        (7500, 8500, 7),
        (8500, 8700, 12),
        (8700, 9700, 8),
        (9700, 9850, 12),
        (9850, 10950, 9),
        (10950, 11050, 12),
        (11050, 12000, 10),
        (12000, 12100, 12),
        (12100, 12600, 11),
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
        (11850, -1, 11),
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
    Clean activity labels according to predefined per-subject rules.
    """
    if subject_label not in SUBJECT_CLEANING:
        print(f'WARNING: No cleaning rules found for {subject_label}, returning unchanged data.')
        return df
    cleaning_rules = SUBJECT_CLEANING[subject_label]
    for start, end, label in cleaning_rules:
        if end == -1:
            end = len(df) - 1
        df.loc[start:end, 'activity_label'] = label
    return df

def load_subject_df(dataset_path, subject_dir):
    """
    Load a subject CSV and apply cleaning.
    """
    subject_path = os.path.join(dataset_path, subject_dir)
    csv_files = glob.glob(os.path.join(subject_path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV file found for {subject_dir}')
    df = pd.read_csv(csv_files[0])
    df = df.dropna(subset=['activity_label'])
    df = clean_subject_data(df, subject_dir)
    return df

def get_features(df):
    """
    Return list of feature columns (excluding activity_label).
    """
    return [col for col in df.columns if col != 'activity_label']

def get_train_activities():
    """
    Return the training activity labels (1..NT).
    """
    return np.arange(1, NT + 1)

def prepare_train_df(df, train_activities, trim=150):
    """
    Return a dataframe with only training activities, trimming `trim` samples at the start and end of each segment.
    """
    df_train = pd.DataFrame()
    for aa in train_activities:
        df_tmp = df.loc[df['activity_label'] == aa]
        df_train = pd.concat([df_train, df_tmp[trim:-trim]])
    return df_train
