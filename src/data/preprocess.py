# src/data/preprocess.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def log_transform(X: pd.DataFrame):
    """
    Applies log2(x + 1) transformation.
    """
    return np.log2(X + 1)

def zscore_normalize(X: pd.DataFrame):
    """
    Applies z-score normalization.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)
