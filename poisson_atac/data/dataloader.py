import anndata as ad
import os
import pandas as pd
import gzip
import scipy.io

data_path = '/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data'

def load_neurips(data_path=data_path, only_train=True):
    path = os.path.join(data_path, 'neurips', 'phase2-private-data/common/openproblems_bmmc_multiome_phase2', 'openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2.h5ad')
    adata = ad.read(path)
    if only_train:   
        adata = adata[adata.obs.is_train].copy()
    return adata

def load_hematopoiesis(data_path=data_path):
    cache_path = os.path.join(data_path, "GSE129785_scATAC-Hematopoiesis", "GSE129785_scATAC-Hematopoiesis.h5ad")
    cached = os.path.exists(cache_path)
    if cached:
        adata = ad.read(cache_path)
    else:
        fn = [
            os.path.join(data_path, "GSE129785_scATAC-Hematopoiesis", "GSE129785_scATAC-Hematopoiesis-All.mtx"),
            os.path.join(data_path, "GSE129785_scATAC-Hematopoiesis", "GSE129785_scATAC-Hematopoiesis-All.cell_barcodes.txt.gz"),
            os.path.join(data_path, "GSE129785_scATAC-Hematopoiesis", "GSE129785_scATAC-Hematopoiesis-All.peaks.txt.gz")
        ]
        X = scipy.io.mmread(fn[0]).T.tocsr()
        obs = pd.read_csv(fn[1], header=None, sep="\t", index_col=0)
        obs.index.name = None
        var = pd.read_csv(fn[2], header=None, sep="\t", names=['names'])
        var.index = var['names'].values
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.write(cache_path)
    return adata