import anndata as ad
import os
import pandas as pd
import scanpy as sc
import scipy.io
import numpy as np
from ._utils import reads_to_fragments

def load_trapnell(convert_counts=True):
    data_path = '/lustre/groups/ml01/workspace/laura.martens/data/trapnell_sciATAC_fetal_tissue'
    cache_path = os.path.join(data_path, "all_tissues.h5ad")
    cached = os.path.exists(cache_path)
    if cached:
        adata = ad.read(cache_path)
        adata.obs_names_make_unique()
        sc.pp.filter_genes(adata, min_cells=int(adata.shape[0]*0.01))
        adata.layers["counts"] = adata.X.copy()
        if convert_counts:
            reads_to_fragments(adata, layer="counts")
        adata.X = (adata.X > 0).astype(float)
    else:
        files = pd.Series(os.listdir(data_path, ))
        files = files[files.str.contains('.h5ad')]
        adatas = [ad.read(os.path.join(data_path, file)) for file in files]
        adata = ad.concat(adatas)
        adata.write(cache_path)
    return adata

def load_aerts(convert_counts=True):
    data_path = '/lustre/groups/ml01/workspace/laura.martens/data/aerts_fly_brain'
    cache_path = os.path.join(data_path, "All_timepoints.h5ad")
    cached = os.path.exists(cache_path)
    if cached:
        adata = ad.read(cache_path)
        adata.obs_names_make_unique()
        #sc.pp.filter_genes(adata, min_cells=int(adata.shape[0]*0.01))
        adata.layers["counts"] = adata.X.copy()
        if convert_counts:
            reads_to_fragments(adata, layer="counts")
        adata.X = (adata.X > 0).astype(float)
    else:
        raise NotImplementedError
    return adata
    
def load_neurips(data_path='/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data', only_train=True, gex=False, batch=None, convert_counts=True, multiome=False):
    path = os.path.join(data_path, 'neurips', 'phase2-private-data/common/openproblems_bmmc_multiome_phase2', 'openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2.h5ad')
    adata = ad.read(path)
    
    if convert_counts:
        reads_to_fragments(adata, layer="counts")
        
    adata.layers["counts"] = scipy.sparse.csr_matrix(adata.layers["counts"])
    adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.obs["size_factor"] = adata.layers["counts"].sum(axis =1)
    if gex and not multiome:
        path = os.path.join(data_path, 'neurips', 'phase2-private-data/common/openproblems_bmmc_multiome_phase2', 'openproblems_bmmc_multiome_phase2.manual_formatting.output_rna.h5ad')
        adata_gex = ad.read(path)
        adata.obsm["X_gex"] = adata_gex.layers['counts'].A.copy()
    elif multiome and not gex:
        path = os.path.join(data_path, 'neurips', 'phase2-private-data/common/openproblems_bmmc_multiome_phase2', 'openproblems_bmmc_multiome_phase2.manual_formatting.output_rna.h5ad')
        adata_gex = ad.read(path)
        adata = ad.concat([adata_gex, adata], axis=1, label='modality', keys=['Gene Expression', 'Peaks'], merge='unique')
        print(adata.obs.columns)
    if only_train:   
        adata = adata[adata.obs.is_train]
    if batch is not None:
        batch = ([batch] if isinstance(batch, str) else batch)
        adata = adata[adata.obs["batch"].isin(batch)].copy()

    return adata

# Cell types from https://satijalab.org/signac/articles/monocle.html
def load_hematopoiesis(data_path='/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data', convert_counts=True):
    cache_path = os.path.join(data_path, "GSE129785_scATAC-Hematopoiesis", "GSE129785_scATAC-Hematopoiesis.h5ad")
    cached = os.path.exists(cache_path)
    if cached:
        adata = ad.read(cache_path)
        adata.obs_names_make_unique()
        sc.pp.filter_genes(adata, min_cells=int(adata.shape[0]*0.01))
        adata.layers["counts"] = adata.X.copy()
        if convert_counts:
            reads_to_fragments(adata, layer="counts")
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
        
        adata.obs["size_factor"] = adata.layers["counts"].sum(axis =1)
        adata.write(cache_path)
    return adata

def save_for_seurat(adata, outdir, sep=[('-', ':')]):
    import scipy.io 
    #barcodes
    barcodes = pd.DataFrame(adata.obs_names, columns = ['barcodes'])
    #peaks
    peaks = pd.DataFrame(adata.var_names, columns = ['peaks'])
    for replacement in sep:
        peaks.loc[:, 'peaks'] = peaks.loc[:, 'peaks'].str.replace(replacement[0], replacement[1], 1)
    #matrix
    matrix = adata.X
    counts = adata.layers['counts']
    
    #save data
    print("Writing barcodes")
    barcodes.to_csv(os.path.join(outdir, 'barcodes.csv'), index=False)
    print("Writing peaks")
    peaks.to_csv(os.path.join(outdir, 'peaks.csv'), index=False)
    
    print("Writing binary matrix")
    #scipy.io.mmwrite(os.path.join(outdir, 'matrix.mtx'), matrix)
    print("Writing count matrix")
    scipy.io.mmwrite(os.path.join(outdir, 'counts.mtx'), counts)
    print("Done!")