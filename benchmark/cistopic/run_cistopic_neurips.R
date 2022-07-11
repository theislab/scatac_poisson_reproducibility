suppressWarnings(library(cisTopic))
library(Matrix)
library(data.table)

setwd('/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/benchmark/neurips/cistopic/')
matrix <- readMM("../matrix.mtx")
matrix <- t(matrix)

peaks <- fread('../peaks.csv', header=TRUE)
rownames(matrix) = peaks$peaks

barcodes <- fread('../barcodes.csv', header=TRUE)
colnames(matrix) = barcodes$barcodes

print("Creating Cistopic Object")
cisTopicObject <- cisTopic::createcisTopicObject(matrix, min.cells=0, min.regions=0)
print(cisTopicObject)

print("Run Cistopic")
cisTopicObject <- cisTopic:::runWarpLDAModelsCount(cisTopicObject, topic=c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100), seed=987, 
nCores=10, iterations = 500, addModels=FALSE)
cisTopicObject <- cisTopic::selectModel(cisTopicObject, type='derivative')

saveRDS(cisTopicObject, file='neurips.Rds')
cellassign <- cisTopic::modelMatSelection(cisTopicObject, 'cell', 'Z-score')
write.csv(t(cellassign),file='embedding.csv')
