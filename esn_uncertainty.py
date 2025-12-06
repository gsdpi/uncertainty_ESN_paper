
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def calc_metrics(actual_labels, scores, plot_roc=False):
    """
    Calculate and print metrics for binary classification, optionally plot ROC curve.
    Parameters
    ----------
    actual_labels : array-like
        Ground truth binary labels (0/1)
    scores : array-like
        Model scores (e.g., log-probabilities or probabilities)
    plot_roc : bool, optional
        If True, plot ROC curve. Default: False
    """
    from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
    import matplotlib.pyplot as plt

    fpr, tpr, thresholds = roc_curve(actual_labels, scores)
    roc_auc = auc(fpr, tpr)
    th_optimal = thresholds[np.argmax(tpr - fpr)]

    sensitivity = recall_score(actual_labels, scores > th_optimal)
    specificity = recall_score(np.logical_not(actual_labels),
                              np.logical_not(scores > th_optimal))
    precision = precision_score(actual_labels, scores > th_optimal)
    f1 = f1_score(actual_labels, scores > th_optimal)

    print(f'\nMetrics:')
    print(f'  AUC: {roc_auc:.3f}')
    print(f'  Optimal threshold: {th_optimal:.3f}')
    print(f'  Sensitivity: {sensitivity:.3f}')
    print(f'  Specificity: {specificity:.3f}')
    print(f'  Precision: {precision:.3f}')
    print(f'  F1-score: {f1:.3f}')

    if plot_roc:
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
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
    return roc_auc, th_optimal


def train_uncertainty_model(df_train, features, target_column, r, window_length, 
                             stride, train_activities, reservoir=None, states_train=None,
                             transition_window=None):
    """
    Train KDE model for epistemic uncertainty estimation from training data.
    Automatically excludes transition zones between activities.
    
    Parameters
    ----------
    df_train : pandas.DataFrame
        DataFrame with TRAINING data (only train activities)
    features : list
        List of column names to use as features
    target_column : str
        Name of the target column (e.g. 'activity_label')
    r : int
        Number of dimensions (singular values) to use for PDF
    window_length : int
        Sliding window length (L)
    stride : int
        Sliding window step (S)
    train_activities : array-like
        List or array with activity labels used for training
    reservoir : reservoirpy.nodes.Reservoir, optional
        Reservoir node of the ESN. Either reservoir or states_train must be provided.
    states_train : numpy.ndarray, optional
        Pre-computed reservoir states. Either reservoir or states_train must be provided.
    transition_window : int, optional
        Number of samples to exclude before and after each transition (default: window_length)
        
    Returns
    -------
    kde_model : sklearn.neighbors.KernelDensity
        Trained KDE model with singular values from training set
    """
    
    # Set transition_window to window_length if not provided
    if transition_window is None:
        transition_window = window_length
    
    # Calculate reservoir states for training data
    if states_train is None:
        if reservoir is None:
            raise ValueError("Either 'reservoir' or 'states_train' must be provided")
        X_train = df_train[features].values.reshape(-1, len(features))
        print('Computing reservoir states for training data...')
        states_train = reservoir.run(X_train)
    else:
        print('Using pre-computed reservoir states for training data...')
    
    Y_train = df_train[target_column].values
    
    # Create mask to exclude transitions
    mask_transition = np.zeros(len(df_train)).astype(bool)
    
    # Mark first 'transition_window' samples
    mask_transition[:transition_window] = True
    
    # Detect transitions
    df_train_reset = df_train.reset_index(drop=True)
    transitions = df_train_reset[target_column] != df_train_reset[target_column].shift(1)
    
    for i in transitions[transitions].index:
        start_idx = max(0, i - transition_window)
        end_idx = min(i + transition_window, len(df_train))
        mask_transition[start_idx:end_idx] = True
    
    # Apply sliding window to create reference PDF
    C_pdf = []
    Q_train = len(X_train)
    
    print(f'Decomposing training data (window size {window_length}, stride {stride})...')
    rango_train = np.arange(0, Q_train - window_length, stride)
    
    skipped_svd = 0
    for i in rango_train:
        idx = np.arange(i, i + window_length)
        
        # Check if any sample in window is in transition zone
        if np.any(mask_transition[idx]):
            # Skip this window if it contains transitions
            continue
            
        print(f"\rWindow {i} of {rango_train[-1]}", end='', flush=True)
        
        # Perform SVD with error handling
        try:
            # Check for NaN or Inf values
            window_data = states_train[idx, :].T
            if np.any(~np.isfinite(window_data)):
                skipped_svd += 1
                continue
            
            U, s, VT = np.linalg.svd(window_data, full_matrices=False)
            # Add new high-dimensional point
            C_pdf.append(s)
        except np.linalg.LinAlgError:
            # Skip windows where SVD fails to converge
            skipped_svd += 1
            continue
    
    C_pdf = np.array(C_pdf)
    print(f'\nValid windows: {len(C_pdf)} (transitions excluded, {skipped_svd} SVD failures skipped)')
    
    if len(C_pdf) == 0:
        return None

    # Estimate PDF with KDE using first r singular values
    print(f'Estimating PDF with KDE (r={r})...')
    values = np.stack(C_pdf[:, 0:r])
    bw = len(values) ** (-1. / (r + 4))  # Scott's rule of thumb
    kde_model = KernelDensity(kernel='gaussian', bandwidth=bw).fit(values)
    
    print('Uncertainty model trained.')
    
    return kde_model


def evaluate_uncertainty_on_signal(df, features, reservoir, kde_model, r, window_length, stride):
    """
    Evaluate epistemic uncertainty of signal data using a previously trained KDE model.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with COMPLETE data to evaluate
    features : list
        List of column names to use as features
    reservoir : reservoirpy.nodes.Reservoir
        Reservoir node of the trained ESN
    kde_model : sklearn.neighbors.KernelDensity
        Previously trained KDE model from estimate_pdf_from_train()
    r : int
        Number of dimensions (singular values) - must match KDE r
    window_length : int
        Sliding window length (L)
    stride : int
        Sliding window step (S)
        
    Returns
    -------
    logprobX_exp : numpy.ndarray
        Array with expanded log-likelihood for each sample
    """
    
    # Process all data from complete dataframe
    print('Computing reservoir states for all data...')
    X_all = df[features].values
    states_all = reservoir.run(X_all)
    
    # Apply sliding window to all data
    C = []
    Q_all = X_all.shape[0]
    
    print(f'Decomposing all data (window size {window_length}, stride {stride})...')
    rango_all = np.arange(0, Q_all - window_length, stride)
    
    skipped_svd = 0
    for i in rango_all:
        idx = np.arange(i, i + window_length)
        print(f"\rWindow {i} of {rango_all[-1]}", end='', flush=True)
        
        try:
            # Check for NaN or Inf values
            window_data = states_all[idx, :].T
            if np.any(~np.isfinite(window_data)):
                # Use zeros for problematic windows
                C.append(np.zeros(min(window_data.shape)))
                skipped_svd += 1
                continue
            
            U, s, VT = np.linalg.svd(window_data, full_matrices=False)
            C.append(s)
        except np.linalg.LinAlgError:
            # Use zeros for windows where SVD fails
            C.append(np.zeros(min(states_all[idx, :].T.shape)))
            skipped_svd += 1
            continue
    
    C = np.array(C)
    if skipped_svd > 0:
        print(f'\nDone. ({skipped_svd} SVD failures handled)')
    else:
        print('\nDone.')
    
    # Evaluate all samples with kernel
    print('Evaluating log-probabilities...')
    logprobX = kde_model.score_samples(C[:, 0:r])
    
    # Expand score to adjust lengths
    logprobX_exp = np.kron(logprobX, np.ones(stride))
    
    print('Uncertainty evaluation completed.')
    
    return logprobX_exp
