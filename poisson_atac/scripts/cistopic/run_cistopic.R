suppressWarnings(library(cisTopic))
library(SeuratDisk)
library(Matrix)

# Converter always saves first obsm argument as embedding (can take very long if large)

# Convert(paste0(path, "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2_lean.h5ad"), 
# paste0(path, "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2.h5seurat"), assay="ATAC")


#
# path <- "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/GSE129785_scATAC-Hematopoiesis/"
# Convert(paste0(path, "GSE129785_scATAC-Hematopoiesis_lean.h5ad"), 
# paste0(path, "GSE129785_scATAC-Hematopoiesis.h5seurat"), assay="ATAC")

# Read in Seurat
dataset <- "hematopoiesis"
if(dataset == "neurips"){
    path <- "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/neurips/phase2-private-data/common/openproblems_bmmc_multiome_phase2/"
    data <- LoadH5Seurat(paste0(path, "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2.h5seurat"))
} else if (dataset == "hematopoiesis") {
   path <- "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/GSE129785_scATAC-Hematopoiesis/"
   data <- LoadH5Seurat(paste0(path, "GSE129785_scATAC-Hematopoiesis.h5seurat"))
}




matrix <- Seurat::GetAssayData(data)
rownames(matrix) = sub('-', ':', rownames(matrix))


cisTopicObject <- cisTopic::createcisTopicObject(matrix)


cisTopicObject <- cisTopic::runWarpLDAModels(cisTopicObject, topic=c(20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130), seed=987, nCores=12, iterations = 500, addModels=FALSE)
cisTopicObject <- cisTopic::selectModel(cisTopicObject, type='derivative')

saveRDS(cisTopicObject, file=paste0(dataset, '.Rds'))
cellassign <- cisTopic::modelMatSelection(cisTopicObject, 'cell', 'Probability')
write.table(cellassign,file=paste0(dataset, ".txt"))