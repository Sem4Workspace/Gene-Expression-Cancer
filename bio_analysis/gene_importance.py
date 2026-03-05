import numpy as np
import pandas as pd

def extract_top_genes(Vt, gene_names, component_index=0, top_n=20):
    """
    Extract top contributing genes for a given SVD component.

    Parameters:
        Vt: (k x genes) matrix from SVD
        gene_names: list of gene column names
        component_index: which SVD component
        top_n: number of top genes to return
    """

    component_loadings = Vt[component_index]

    # Get indices sorted by absolute contribution
    top_indices = np.argsort(np.abs(component_loadings))[::-1][:top_n]

    top_genes = pd.DataFrame({
        "Gene": np.array(gene_names)[top_indices],
        "Loading": component_loadings[top_indices]
    })

    return top_genes