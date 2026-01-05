"""
Utilities for the SWaT dataset: download, loading, features, and splits.
The download uses the public Google Drive link provided in the snippet.

https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

"""

import os
import urllib.request
import zipfile
from typing import List, Optional

import numpy as np
import pandas as pd

# Dataset-level parameters (tune as needed)
WINDOW_LENGTH = 120  # samples per window
STRIDE = 10          # step between windows
SAMPLING_PERIOD = 1.0  # adjust if you know the true sampling rate

# Google Drive source (Attack2.csv.zip) from the provided snippet
DEFAULT_DATA_URL = (
    "https://drive.google.com/uc?export=download&id=1klDpUNwhYp_pbUALdpKMbydBTYupIvkH"
)
DEFAULT_ZIP_NAME = "Attack2.csv.zip"
DEFAULT_CSV_NAME = "Attack2.csv"
DEFAULT_CACHE_DIR = "swat_dataset"


def _maybe_download_swat(dest_dir: str = DEFAULT_CACHE_DIR, auto_download: bool = True) -> str:
    """
    Download and extract the SWaT CSV using tf.keras.utils.get_file if available.
    Returns the CSV path. Raise RuntimeError if download is not possible.
    """
    dest_dir = os.path.expanduser(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    target_csv = os.path.join(dest_dir, DEFAULT_CSV_NAME)
    if os.path.exists(target_csv):
        return target_csv

    if not auto_download:
        raise RuntimeError("auto_download is disabled and no path was provided, and no local CSV was found")

    zip_path = os.path.join(dest_dir, DEFAULT_ZIP_NAME)

    # Download zip if missing
    if not os.path.exists(zip_path):
        print(f"Downloading SWaT zip from {DEFAULT_DATA_URL}...")
        urllib.request.urlretrieve(DEFAULT_DATA_URL, zip_path)

    # Extract CSV
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)

    return target_csv if os.path.exists(target_csv) else target_csv


def load_swat_data(csv_path: Optional[str] = None, auto_download: bool = True) -> pd.DataFrame:
    """
    Load the SWaT dataset as a DataFrame.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV. If None, it will attempt to download using TensorFlow's
        keras utils and DEFAULT_DATA_URL.
    auto_download : bool
        Whether to download if csv_path is None.
    """
    if csv_path is None:
        csv_path = _maybe_download_swat(dest_dir=DEFAULT_CACHE_DIR, auto_download=auto_download)

    csv_path = os.path.expanduser(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def _detect_attack_column(df: pd.DataFrame) -> str:
    """Infer the attack/label column name (case-insensitive heuristics)."""
    candidates = [
        "Attack",
        "attack",
        "Label",
        "label",
        "is_attack",
        "anomaly",
    ]
    for col in df.columns:
        if col in candidates:
            return col
        if col.lower() in {c.lower() for c in candidates}:
            return col
    raise ValueError("Could not find an attack/label column in the dataframe")


def get_features(df: pd.DataFrame) -> List[str]:
    """
    Return numeric feature columns, excluding label/timestamp columns.
    """
    label_cols = {"attack", "label", "is_attack"}
    time_cols = {"timestamp", "time", "datetime"}

    features = []
    for col in df.columns:
        col_low = col.lower()
        if col_low in label_cols or col_low in time_cols:
            continue
        if np.issubdtype(df[col].dtype, np.number):
            features.append(col)
    if not features:
        raise ValueError("No numeric feature columns found")
    return features


def create_label_mask(df: pd.DataFrame, attack_is_one: bool = True) -> np.ndarray:
    """
    Create a binary mask: 1 for normal, 0 for attack.
    """
    attack_col = _detect_attack_column(df)
    labels = df[attack_col].values
    if attack_is_one:
        mask = (labels == 0).astype(int)
    else:
        mask = (labels == 1).astype(int)
    return mask


def prepare_train_df(df: pd.DataFrame, attack_is_one: bool = True, trim: int = 0) -> pd.DataFrame:
    """
    Return a dataframe with only normal (non-attack) samples.
    Optionally trim samples from start/end to avoid boundary effects.
    """
    attack_col = _detect_attack_column(df)
    if attack_is_one:
        df_train = df[df[attack_col] == 0]
    else:
        df_train = df[df[attack_col] == 1]

    if trim > 0 and len(df_train) > 2 * trim:
        df_train = df_train.iloc[trim:-trim]
    return df_train.reset_index(drop=True)
