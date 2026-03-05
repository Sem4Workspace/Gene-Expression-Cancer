# src/data/split_data.py

import os
import pandas as pd
def save_processed_data(X, y, output_dir):
    """
    Saves processed feature matrix and labels.
    Creates directory if it does not exist.
    """
    os.makedirs(output_dir, exist_ok=True)

    X.to_csv(os.path.join(output_dir, "X_processed.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "labels.csv"), index=False)
