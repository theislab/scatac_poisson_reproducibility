library(Signac)
library(Seurat)

library(Matrix)
library(data.table)
library(harmony)
library(anndata)

# Load input parameters
args <- commandArgs(trailingOnly = TRUE)
print(args)

# Set input parameters
input_file <- args[1]
output_file <- args[2]
harmony_file <- args[3]
replacement_1 <- args[4]
replacement_2 <- args[5]
batch_key <- args[6]

print(args)
# Load data
adata <- read_h5ad(input_file)
matrix <- adata$layers['counts']
matrix <- t(matrix)

peaks <- adata$var
rownames(matrix) = rownames(peaks)

barcodes <- adata$obs_names
colnames(matrix) = barcodes

metadata <- adata$obs
rownames(metadata) <- barcodes

max(matrix)

chrom_assay <- CreateChromatinAssay(
  counts = matrix,
  sep = c(replacement_1, replacement_2),
  min.cells = 0,
  min.features = 0
)

chrom_assay

atac <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "peaks",
  meta.data = metadata
)

atac <- RunTFIDF(atac)
atac <- FindTopFeatures(atac, min.cutoff = 'q0')
atac <- RunSVD(atac)

DepthCor(atac)

embedding = Embeddings(atac[['lsi']])[,2:30] #remove first component

write.csv(embedding, file = output_file)

## Run Harmony

atac <- RunHarmony(
  object = atac,
  group.by.vars = batch_key,
  reduction = 'lsi',
  assay.use = 'peaks',
  project.dim = FALSE
)

embedding = Embeddings(atac[['harmony']])[,2:30] #remove first component

write.csv(embedding, file = harmony_file)
