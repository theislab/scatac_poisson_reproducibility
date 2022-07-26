library(ArchR)
library(data.table)
library(parallel)

setwd('/lustre/groups/ml01/workspace/laura.martens/atac_poisson_data/benchmark/neurips/archr/')
addArchRGenome("hg38")

samples = c('s1d1','s1d2','s1d3','s2d1','s2d4','s2d5','s3d1','s3d10','s3d3','s3d6','s3d7','s4d1','s4d8','s4d9')
inputFiles <- sapply(samples, function(x){paste0('/lustre/groups/ml01/datasets/projects/20220323_neurips21_bmmc_christopher.lance/multiome/', x, '/cellranger_out/atac_fragments.tsv.gz')})
ArrowFiles <- createArrowFiles(
  inputFiles = inputFiles,
  sampleNames = samples,
  filterTSS = 0, #Dont set this too high because you can always increase later
  filterFrags = 0, 
  addTileMat = TRUE,
  addGeneScoreMat = FALSE, # we don't need them for embedding
)

# Combine into project
proj <- ArchRProject(
  ArrowFiles = ArrowFiles, 
  outputDirectory = "neurips",
  copyArrows = TRUE #This is recommened so that you maintain an unaltered copy for later usage.
)

#filter cells based on barcodes
barcodes <- fread('../barcodes.csv', header=TRUE)
proj <- proj[barcodes$barcodes] #filter based on barcodes

proj <- addIterativeLSI(ArchRProj = proj, useMatrix = "TileMatrix", name = "IterativeLSI")

embedding <- getReducedDims(
      ArchRProj = proj, 
      reducedDims = "IterativeLSI", 
      dimsToUse = NULL, 
      corCutOff = 0.75, 
      scaleDims = NULL
  )

write.csv(embedding, '/lustre/groups/ml01/workspace/laura.martens/atac_poisson_data/benchmark/neurips/archr/embedding.csv', row.names=TRUE)
proj <- saveArchRProject(ArchRProj = proj)

