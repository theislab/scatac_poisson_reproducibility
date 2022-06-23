library(ArchR)
library(data.table)
library(parallel)

setwd('/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/benchmark/neurips/archr/')
addArchRGenome("hg38")

inputFiles <- "/storage/groups/ml01/datasets/projects/20220323_neurips21_bmmc_christopher.lance/multiome/aggr_donors/atac_fragments.tsv.gz"

ArrowFiles <- createArrowFiles(
  inputFiles = inputFiles,
  sampleNames = 'neurips',
  filterTSS = 2, #Dont set this too high because you can always increase later
  filterFrags = 500, 
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

write.csv(embedding, '/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/benchmark/neurips/archr/embedding.csv', row.names=TRUE)
proj <- saveArchRProject(ArchRProj = proj)

