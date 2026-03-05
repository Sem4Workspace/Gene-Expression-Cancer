# SVD-Based Gene Expression Analysis and Cancer Classification

Analyze high-dimensional gene expression data using Singular Value Decomposition (SVD) and machine learning to identify patterns and classify cancer types.

## Dataset

UCI Gene Expression Cancer RNA-Seq dataset:

- **801** tumor samples × **~20,000** genes
- **5 cancer types:** BRCA, KIRC, COAD, LUAD, PRAD

## Project Phases

| Phase | Notebook | Description |
|-------|----------|-------------|
| 1 | `01_data_exploration.ipynb` | Data loading, log₂ transform, z-score normalization |
| 2 | `02_svd_analysis.ipynb` | Truncated SVD (k=50), scree plot, variance explained |
| 3 | `03_visualization.ipynb` | 2D/3D SVD projections, PCA comparison |
| 4 | `04_clustering.ipynb` | K-Means on SVD features, elbow & silhouette analysis |
| 5 | `05_classification.ipynb` | Cancer classification using Logistic Regression on SVD features |
| 6 | `06_experiments.ipynb` | SVD rank sensitivity, normalization comparison, classifier comparison, gene importance |

## Project Structure

```
src/                  # Core modules
  data/               # Loading, preprocessing, saving
  linear_algebra/     # SVD computation, reconstruction error
  ml/                 # Classification, clustering
  visualization/      # SVD plots, projections, clustering plots
bio_analysis/         # Gene importance, variance interpretation
experiments/          # Rank sensitivity, normalization & classifier comparison
notebooks/            # Jupyter notebooks (Phases 1–6)
Data/                 # Raw and processed datasets
results/              # Figures, tables, logs
```

## Setup

```bash
pip install -r environment/requirements.txt
```

## Key Results

- SVD captures most variance in fewer than 20 components
- Log + z-score normalization yields best classification performance
- All classifiers (Logistic Regression, SVM, Random Forest) achieve high accuracy on SVD features
- Top gene loadings from Vᵀ reveal biologically meaningful contributors