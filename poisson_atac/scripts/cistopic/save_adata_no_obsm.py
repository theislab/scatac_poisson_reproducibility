import poisson_atac as patac
import os

# path = "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/neurips/phase2-private-data/common/openproblems_bmmc_multiome_phase2/"
# adata = patac.data.load_neurips()
# adata.obsm = {}
# adata.uns = {}
# adata.layers = {}

# adata.write(os.path.join(path, "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2_lean.h5ad"))

# for batch in adata.obs.batch.unique():
#     print(batch)
#     a_batch = adata[adata.obs.batch == batch].copy()
#     a_batch.write(os.path.join(path, f"openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2_lean_{batch}.h5ad"))


# path = "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/GSE129785_scATAC-Hematopoiesis/"
# adata = patac.data.load_hematopoiesis()
# adata.obsm = {}
# adata.uns = {}
# adata.layers = {}

# adata.write(os.path.join(path, "GSE129785_scATAC-Hematopoiesis_lean.h5ad"))

path = "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/neurips/phase2-private-data/common/openproblems_bmmc_multiome_phase2/"
adata = patac.data.load_neurips()

adata.write(os.path.join(path, "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2_lean_counts.h5ad"))

for batch in adata.obs.batch.unique():
    print(batch)
    a_batch = adata[adata.obs.batch == batch].copy()
    a_batch.X = a_batch.layers["counts"]
    a_batch.obsm = {}
    a_batch.uns = {}
    a_batch.layers = {}
    a_batch.write(os.path.join(path, f"openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2_lean_{batch}_counts.h5ad"))


path = "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/GSE129785_scATAC-Hematopoiesis/"
adata = patac.data.load_hematopoiesis()
adata.X = adata.layers["counts"]
adata.obsm = {}
adata.uns = {}
adata.layers = {}

adata.write(os.path.join(path, "GSE129785_scATAC-Hematopoiesis_lean_counts.h5ad"))