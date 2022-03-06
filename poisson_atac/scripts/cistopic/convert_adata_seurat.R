suppressWarnings(library(cisTopic))
library(SeuratDisk)
library(Matrix)

# Converter always saves first obsm argument as embedding (can take very long if large)

path <- "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/neurips/phase2-private-data/common/openproblems_bmmc_multiome_phase2/"
# Convert(paste0(path, "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2_lean.h5ad"), 
# paste0(path, "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2.h5seurat"), assay="ATAC")

batches = c('s1d1','s1d2','s1d3','s2d1','s2d4','s2d5','s3d10','s3d3','s3d6','s3d7')
for (batch in batches) {
    print(batch)
   Convert(paste0(path, "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2_lean_", batch, ".h5ad"), 
paste0(path, "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2_", batch, ".h5seurat"), assay="ATAC")
}


# path <- "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/GSE129785_scATAC-Hematopoiesis/"
# Convert(paste0(path, "GSE129785_scATAC-Hematopoiesis_lean.h5ad"), 
# paste0(path, "GSE129785_scATAC-Hematopoiesis.h5seurat"), assay="ATAC")
