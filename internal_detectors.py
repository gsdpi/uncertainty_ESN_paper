##################################################################
# OOD detectors operating on INSTANTANEOUS reservoir states
# (no sliding windows). Shared by icann_/imwsha_/synth_internal.
# Each detector is a pair (fit, score):
#   fit(states_train)    -> model
#   score(model, states) -> 1D array, HIGHER = more in-distribution
##################################################################
import numpy as np

MAHALANOBIS_REG = 1e-6  # relative ridge on the covariance (reservoir states are highly correlated)


def fit_mahalanobis(states_train):
    """Mean and inverse covariance of the training reservoir states."""
    mu = states_train.mean(axis=0)
    cov = np.cov(states_train, rowvar=False)
    cov += MAHALANOBIS_REG * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    return {'mu': mu, 'cov_inv': np.linalg.pinv(cov)}


def score_mahalanobis(model, states):
    """Negative squared Mahalanobis distance of each state to the training distribution."""
    d = states - model['mu']
    d2 = np.einsum('ij,jk,ik->i', d, model['cov_inv'], d)
    return -d2


DETECTORS = {
    'mahalanobis': (fit_mahalanobis, score_mahalanobis),
    # 'pca': (fit_pca, score_pca),
}
