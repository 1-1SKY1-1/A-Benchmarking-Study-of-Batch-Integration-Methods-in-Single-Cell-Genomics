"""
Run Scanorama integration using scanpy.external.pp.scanorama_integrate.
Outputs:
  - UMAP before and after
  - scanorama_metrics.csv
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
from scipy.io import mmread
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import warnings
import importlib

warnings.filterwarnings("ignore")
sc.settings.set_figure_params(dpi=100, facecolor='white')


# ======================================================
# Load GSE132044
# ======================================================
def load_gse132044(cell_tsv, gene_tsv, mtx_file):
    print("Loading raw GSE132044...")
    cells = pd.read_csv(cell_tsv, sep="\t", index_col=0)
    genes = pd.read_csv(gene_tsv, sep="\t")
    counts = mmread(mtx_file).T.tocsr()  # genes × cells → cells × genes

    # trim mismatch
    n = min(counts.shape[0], len(cells))
    counts = counts[:n]
    cells = cells.iloc[:n]

    adata = sc.AnnData(X=counts)
    adata.obs = cells.copy()

    if "gene_ids" in genes.columns:
        adata.var = genes.set_index("gene_ids")
    else:
        adata.var = pd.DataFrame(index=[f"gene_{i}" for i in range(counts.shape[1])])

    # batch = sequencing method
    adata.obs["batch"] = adata.obs.index.str.split(".").str[1]

    print("Loaded:", adata)
    print(adata.obs["batch"].value_counts())
    return adata


# ======================================================
# preprocess
# ======================================================
def preprocess(adata):
    ad = adata.copy()

    # try mitochondrial tag
    try:
        ad.var["mt"] = ad.var_names.str.split("_").str[1].str.startswith("mt-")
    except:
        ad.var["mt"] = False

    sc.pp.calculate_qc_metrics(ad, qc_vars=["mt"], percent_top=None, inplace=True)

    sc.pp.filter_genes(ad, min_cells=3)
    ad = ad[(ad.obs["n_genes_by_counts"] >= 200) &
            (ad.obs["pct_counts_mt"] < 20)].copy()

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    sc.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True)
    sc.pp.scale(ad)
    sc.tl.pca(ad, n_comps=50)

    return ad


# ======================================================
# metrics
# ======================================================
def compute_metrics(adata, batch_key="batch", label_key="leiden"):

    X = adata.obsm["X_pca"][:, :20]

    metrics = {}

    # ASW batch
    try:
        metrics["ASW_batch"] = silhouette_score(X, adata.obs[batch_key])
    except:
        metrics["ASW_batch"] = np.nan

    # ILISI
    try:
        nbrs = NearestNeighbors(n_neighbors=50).fit(X)
        _, idx = nbrs.kneighbors(X)
        lisi_vals = []
        for i in range(len(X)):
            batches = adata.obs[batch_key].iloc[idx[i]].values
            u, c = np.unique(batches, return_counts=True)
            p = c / c.sum()
            simpson = np.sum(p ** 2)
            lisi_vals.append(1 / simpson)
        metrics["ILISI"] = np.mean(lisi_vals)
    except:
        metrics["ILISI"] = np.nan

    # PCR
    try:
        le = LabelEncoder()
        y = le.fit_transform(adata.obs[batch_key])
        r2_all = []
        for i in range(10):
            xi = adata.obsm["X_pca"][:, i].reshape(-1, 1)
            reg = LinearRegression().fit(xi, y)
            y_pred = reg.predict(xi)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2_all.append(1 - ss_res / ss_tot)
        metrics["PCR"] = np.mean(r2_all)
    except:
        metrics["PCR"] = np.nan

    # ASW label
    try:
        metrics["ASW_label"] = silhouette_score(X, adata.obs[label_key])
    except:
        metrics["ASW_label"] = np.nan

    # cLISI
    try:
        nbrs = NearestNeighbors(n_neighbors=50).fit(X)
        _, idx = nbrs.kneighbors(X)
        lisi_vals = []
        for i in range(len(X)):
            labs = adata.obs[label_key].iloc[idx[i]].values
            u, c = np.unique(labs, return_counts=True)
            p = c / c.sum()
            simpson = np.sum(p ** 2)
            lisi_vals.append(1 / simpson)
        metrics["cLISI"] = np.mean(lisi_vals)
    except:
        metrics["cLISI"] = np.nan

    # Graph connectivity
    try:
        conn = adata.obsp["connectivities"]
        conns = []
        for cl in adata.obs[label_key].unique():
            mask = adata.obs[label_key] == cl
            if mask.sum() <= 1:
                continue
            sub = conn[mask][:, mask]
            comp, _ = connected_components(sub)
            conns.append(1 - (comp - 1) / max(mask.sum() - 1, 1))
        metrics["Graph_connectivity"] = np.mean(conns)
    except:
        metrics["Graph_connectivity"] = np.nan

    return metrics


# ======================================================
# UMAP
# ======================================================
def plot_umap(adata, color, out):
    sc.pl.umap(adata, color=color, show=False)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================
# Scanorama integration with version handling
# ======================================================
def run_scanorama_integration(adata, key="batch", basis="X_pca", adjusted_basis="X_scanorama"):
    """
    Run Scanorama integration with version compatibility handling.
    """
    try:
        # First try the standard scanpy external function
        print("Attempting Scanorama integration via scanpy.external.pp.scanorama_integrate...")
        sce.pp.scanorama_integrate(adata, key=key, basis=basis, adjusted_basis=adjusted_basis)
        print("Scanorama integration successful via scanpy external function.")
        
    except AttributeError as e:
        if "'assemble'" in str(e):
            print("Detected Scanorama version compatibility issue. Using direct Scanorama integration...")
            
            # Try importing scanorama directly
            try:
                import scanorama
                
                # Split data by batch for Scanorama
                batches = adata.obs[key].unique()
                adatas = []
                for batch in batches:
                    adatas.append(adata[adata.obs[key] == batch].copy())
                
                # Run Scanorama integration
                print(f"Integrating {len(adatas)} batches with Scanorama...")
                
                # Check Scanorama version and use appropriate function
                scanorama_version = importlib.metadata.version("scanorama") if importlib.metadata else "unknown"
                print(f"Scanorama version: {scanorama_version}")
                
                if hasattr(scanorama, 'integrate_scanpy'):
                    # For newer versions of Scanorama
                    integrated = scanorama.integrate_scanpy(adatas, dimred=50)
                    corrected = np.concatenate([ad.obsm['X_scanorama'] for ad in integrated])
                else:
                    # For older versions
                    datasets = [ad.obsm[basis] for ad in adatas]
                    integrated, corrected = scanorama.assemble(datasets)
                    corrected = np.concatenate(corrected)
                
                # Store corrected embeddings
                adata.obsm[adjusted_basis] = corrected
                
                print("Scanorama integration successful via direct method.")
                
            except Exception as e:
                print(f"Error in direct Scanorama integration: {e}")
                raise
        else:
            # Re-raise if it's a different AttributeError
            raise
    
    except Exception as e:
        print(f"Error in Scanorama integration: {e}")
        raise


# ======================================================
# main
# ======================================================
def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="GSE132044_cortex_mm10_cell.tsv.gz")
    parser.add_argument("--gene", default="GSE132044_cortex_mm10_gene.tsv.gz")
    parser.add_argument("--mtx", default="GSE132044_cortex_mm10_count_matrix.mtx.gz")
    parser.add_argument("--outdir", default="scanorama_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load
    adata = load_gse132044(args.cell, args.gene, args.mtx)

    # Preprocess
    adata = preprocess(adata)

    adata = adata[adata.obs.sort_values("batch").index].copy()
    print("Batches are now contiguous.")
    print(adata.obs["batch"].value_counts())

    # Leiden before
    sc.pp.neighbors(adata, n_pcs=20)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    # Plot before
    plot_umap(adata, "batch", f"{args.outdir}/umap_before_batch.png")
    plot_umap(adata, "leiden", f"{args.outdir}/umap_before_leiden.png")

    # Metrics before
    metrics_before = compute_metrics(adata)

    # ======================================================
    # ⭐️ Scanorama Integration with version handling
    # ======================================================
    run_scanorama_integration(adata, key="batch", basis="X_pca", adjusted_basis="X_scanorama")

    # compute neighbors on scanorama
    sc.pp.neighbors(adata, use_rep="X_scanorama")
    sc.tl.umap(adata)

    # After leiden
    sc.tl.leiden(adata)

    # Plot after
    plot_umap(adata, "batch", f"{args.outdir}/umap_after_batch.png")
    plot_umap(adata, "leiden", f"{args.outdir}/umap_after_leiden.png")

    # metrics after
    metrics_after = compute_metrics(adata)

    # Save
    df = pd.DataFrame({"Before": metrics_before, "After": metrics_after})
    df.to_csv(f"{args.outdir}/scanorama_metrics.csv")

    print("\n=== Scanorama Integration Metrics ===")
    print(df)
    
    # Save the integrated AnnData object
    adata.write(f"{args.outdir}/integrated_adata.h5ad", compression="gzip")
    print(f"\nIntegrated data saved to {args.outdir}/integrated_adata.h5ad")


if __name__ == "__main__":
    main()