suppressWarnings(library(cisTopic))
library(SeuratDisk)
library(Matrix)
library(data.table)

args = commandArgs(trailingOnly=TRUE)
# # Read in Seurat
dataset <- args[1]
print(dataset)
if(dataset == "neurips"){
    path <- "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/neurips/phase2-private-data/common/openproblems_bmmc_multiome_phase2/"
    data <- LoadH5Seurat(paste0(path, "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2.h5seurat"))
} else if (dataset == "hematopoiesis") {
   path <- "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/GSE129785_scATAC-Hematopoiesis/"
   data <- LoadH5Seurat(paste0(path, "GSE129785_scATAC-Hematopoiesis.h5seurat"))
} else {
    path <- "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/neurips/phase2-private-data/common/openproblems_bmmc_multiome_phase2/"
    data <- LoadH5Seurat(paste0(path, "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2_", dataset, ".h5seurat"))
}

out_path <- "/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/models/cistopic/"
matrix <- Seurat::GetAssayData(data)
rownames(matrix) = sub('-', ':', rownames(matrix))


cisTopicObject <- cisTopic::createcisTopicObject(matrix, min.cells=0, min.regions=0)


cisTopicObject <- cisTopic::runWarpLDAModels(cisTopicObject, topic=c(20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130), seed=987, nCores=12, iterations = 500, addModels=FALSE)
cisTopicObject <- cisTopic::selectModel(cisTopicObject, type='derivative')

saveRDS(cisTopicObject, file=paste0(out_path, dataset, '.Rds'))
cellassign <- cisTopic::modelMatSelection(cisTopicObject, 'cell', 'Probability')
write.table(cellassign,file=paste0(out_path, dataset, ".txt"))
pred.matrix <- predictiveDistribution(cisTopicObject)
fwrite(pred.matrix,file=paste0(out_path, dataset, "_pred_matrix.txt"), sep='\t')


