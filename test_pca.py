
"""
pca_anomaly_timeseries.py

PCA-based anomaly detection for multichannel time series.

- Multichannel input: X shape (T, C)
- Sliding window representation
- PCA reconstruction error as anomaly score
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------

def window_signal(X, window_size, stride):
    """
    Create sliding windows from a multichannel time series.

    Parameters
    ----------
    X : ndarray, shape (T, C)
    window_size : int
    stride : int

    Returns
    -------
    windows : ndarray, shape (N, window_size, C)
    """
    if X.ndim != 2:
        raise ValueError("X must have shape (T, C)")

    T, C = X.shape
    if T < window_size:
        raise ValueError("window_size larger than signal length")

    windows = sliding_window_view(X, window_size, axis=0)
    windows = windows[::stride]
    return windows


# ---------------------------------------------------------------------
# Feature extraction (same hook as before)
# ---------------------------------------------------------------------

def vectorize_windows(windows):
    """
    Convert windows (N, W, C) to vectors (N, W*C)
    """
    return windows.reshape(windows.shape[0], -1)


# ---------------------------------------------------------------------
# PCA anomaly detector
# ---------------------------------------------------------------------

class PCAAnomalyDetector:
    """
    PCA-based anomaly detector for multichannel time series.
    """

    def __init__(
        self,
        n_components=0.95,
        window_size=50,
        stride=1,
        score_type="reconstruction_error"
    ):
        """
        Parameters
        ----------
        n_components : int or float
            Number of PCA components or explained variance ratio
        window_size : int
        stride : int
        score_type : {'reconstruction_error'}
        """
        self.n_components = n_components
        self.window_size = window_size
        self.stride = stride
        self.score_type = score_type

        self.pca = PCA(n_components=self.n_components)

    def fit(self, X_train):
        """
        Fit PCA using normal data.
        """
        windows = window_signal(
            X_train,
            self.window_size,
            self.stride
        )
        X_vec = vectorize_windows(windows)
        self.pca.fit(X_vec)
        return self

    def score(self, X):
        """
        Compute anomaly scores.

        Returns
        -------
        scores : ndarray, shape (N_windows,)
        """
        windows = window_signal(
            X,
            self.window_size,
            self.stride
        )
        X_vec = vectorize_windows(windows)

        X_proj = self.pca.transform(X_vec)
        X_rec = self.pca.inverse_transform(X_proj)

        # Reconstruction error (L2 per window)
        errors = np.linalg.norm(X_vec - X_rec, axis=1)
        return errors


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":

    np.random.seed(0)

    X_train = np.random.randn(10000, 5)   # normal data
    X_test = np.random.randn(4000, 5)

    detector = PCAAnomalyDetector(
        n_components=0.95,   # keep 95% variance
        window_size=100,
        stride=10
    )

    detector.fit(X_train)
    scores = detector.score(X_test)

    print("Scores shape:", scores.shape)
    print("First scores:", scores[:5])
