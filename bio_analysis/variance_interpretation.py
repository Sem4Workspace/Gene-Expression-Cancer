
import numpy as np
import matplotlib.pyplot as plt

def compute_variance_explained(singular_values):
    """
    Compute explained variance ratio.
    """

    variance = singular_values ** 2
    total_variance = np.sum(variance)

    explained_ratio = variance / total_variance
    cumulative_ratio = np.cumsum(explained_ratio)

    return explained_ratio, cumulative_ratio


def plot_variance(explained_ratio, cumulative_ratio):
    plt.figure(figsize=(6,4))
    plt.plot(cumulative_ratio, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("SVD Variance Interpretation")
    plt.show()