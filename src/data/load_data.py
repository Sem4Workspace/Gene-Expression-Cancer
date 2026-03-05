# src/data/load_data.py

import pandas as pd

def load_uci_dataset(data_path: str, label_path: str):
    """
    Loads UCI Cancer Gene Expression dataset.

    Parameters:
    data_path (str): Path to data.csv
    label_path (str): Path to labels.csv

    Returns:
    X (DataFrame): Gene expression matrix
    y (Series): Class labels
    """

    X = pd.read_csv(data_path)
    y = pd.read_csv(label_path).iloc[:, 0]

    return X, y
