import anndata as ad
import os
import pandas as pd
import scanpy as sc
import scipy.io
import numpy as np

data_path = '/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data'

def load_neurips(data_path=data_path, only_train=True, gex=False, batch=None):
    path = os.path.join(data_path, 'neurips', 'phase2-private-data/common/openproblems_bmmc_multiome_phase2', 'openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2.h5ad')
    adata = ad.read(path)
    adata.layers["counts"].data = np.ceil(adata.layers["counts"].data/2)
    adata.layers["counts"] = scipy.sparse.csr_matrix(adata.layers["counts"])
    adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.obs["size_factor"] = adata.layers["counts"].sum(axis =1)
    if gex:
        path = os.path.join(data_path, 'neurips', 'phase2-private-data/common/openproblems_bmmc_multiome_phase2', 'openproblems_bmmc_multiome_phase2.manual_formatting.output_rna.h5ad')
        adata_gex = ad.read(path)
        adata.obsm["X_gex"] = adata_gex.layers['counts'].A.copy()
    if only_train:   
        adata = adata[adata.obs.is_train]
    if batch is not None:
        batch = ([batch] if isinstance(batch, str) else batch)
        adata = adata[adata.obs["batch"].isin(batch)].copy()
    return adata

# Cell types from https://satijalab.org/signac/articles/monocle.html
def load_hematopoiesis(data_path=data_path):
    cache_path = os.path.join(data_path, "GSE129785_scATAC-Hematopoiesis", "GSE129785_scATAC-Hematopoiesis.h5ad")
    cached = os.path.exists(cache_path)
    if cached:
        adata = ad.read(cache_path)
        adata.obs_names_make_unique()
        sc.pp.filter_genes(adata, min_cells=int(adata.shape[0]*0.01))
        adata.layers["counts"] = adata.X.copy()
        adata.layers["counts"].data = np.ceil(adata.layers["counts"].data/2)
        adata.X = (adata.X > 0).astype(float)
    else:
        fn = [
            os.path.join(data_path, "GSE129785_scATAC-Hematopoiesis", "GSE129785_scATAC-Hematopoiesis-All.mtx"),
            os.path.join(data_path, "GSE129785_scATAC-Hematopoiesis", "GSE129785_scATAC-Hematopoiesis-All.cell_barcodes.txt.gz"),
            os.path.join(data_path, "GSE129785_scATAC-Hematopoiesis", "GSE129785_scATAC-Hematopoiesis-All.peaks.txt.gz")
        ]
        X = scipy.io.mmread(fn[0]).T.tocsr()
        obs = pd.read_csv(fn[1], sep="\t").set_index("Barcodes")
        obs.index.name = None
        var = pd.read_csv(fn[2], sep="\t")
        var.index = var['Feature'].values
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.obsm["X_umap"] = adata.obs[["UMAP1", "UMAP2"]].values
        keys = ("Cluster" + pd.Series(np.arange(1,32)).astype(str)).values
        cluster_names = dict(zip(keys, ["HSC",   "MEP",  "CMP-BMP",  "LMPP", "CLP",  "Pro-B",    "Pre-B",    "GMP",
                        "MDP",    "pDC",  "cDC",  "Monocyte-1",   "Monocyte-2",   "Naive-B",  "Memory-B",
                        "Plasma-cell",    "Basophil", "Immature-NK",  "Mature-NK1",   "Mature-NK2",   "Naive-CD4-T1",
                        "Naive-CD4-T2",   "Naive-Treg",   "Memory-CD4-T", "Treg", "Naive-CD8-T1", "Naive-CD8-T2",
                        "Naive-CD8-T3",   "Central-memory-CD8-T", "Effector-memory-CD8-T",    "Gamma delta T"]))
        adata.obs["cell_type"] = adata.obs["Clusters"].map(cluster_names)
        adata.write(cache_path)
    return adata