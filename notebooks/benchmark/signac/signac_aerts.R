.libPaths(c("~/miniconda3/envs/archr/lib/R/library", "~/miniconda3/envs/signac/lib/R/library"))
library(cisTopic)
library(stringr)
library(data.table)
library(Signac)
library(Seurat)
library(GenomeInfoDb)
library(anndata)

cistopic <- readRDS('/lustre/groups/ml01/workspace/laura.martens/data/aerts_fly_brain/AllTimepoints_cisTopic.Rds')

peaks <- cistopic@region.ranges

seqlevelsStyle(peaks) <- "NCBI"

files <- list.files('/lustre/groups/ml01/workspace/laura.martens/data/aerts_fly_brain/cellRanger')

# load metadata
ids <- list()
objects <- list()
for(sample in files){
    print(sample)
    id <- substr(sample, 9, 11)
    print(id)
    sample_path = paste0('/lustre/groups/ml01/workspace/laura.martens/data/aerts_fly_brain/cellRanger/', sample)
    metadata <- read.table(
      file = paste0(sample_path, '/singlecell.csv'),
      stringsAsFactors = FALSE,
      sep = ",",
      header = TRUE,
      row.names = 1
    )[-1, ] # remove the first row
    metadata$sample_barcode <- str_c(id,rownames(metadata), sep='.')
    metadata <- metadata[metadata$sample_barcode %in% cistopic@cell.names,]
    
    if(nrow(metadata) == 0){
        next
    }
    frags <- CreateFragmentObject(
      path = paste0(sample_path, "/fragments.tsv.gz"),
      cells = rownames(metadata)
    )
    counts <- FeatureMatrix(
  fragments = frags,
  features = peaks,
  cells = rownames(metadata)
        
)
    atac_assay <- CreateChromatinAssay(counts, fragments = frags)
    atac <- CreateSeuratObject(atac_assay, assay = "ATAC", meta.data=metadata)
 saveRDS(atac, paste0('/lustre/groups/ml01/workspace/laura.martens/data/aerts_fly_brain/', id, '.rds'))   
#    objects[id] <- atac
#    ids[id] <- id
    }

library(stringr)

path <- "/lustre/groups/ml01/workspace/laura.martens/data/aerts_fly_brain/"

count_matrices <- list.files(path, '.rds', full.names = FALSE)

ids <- str_split(count_matrices, '.rds', simplify=TRUE)[,1]

objects <- lapply(str_c(path, count_matrices), readRDS)

combined <- merge(
 x = objects[[1]],
 y = c(objects[[2]], objects[[3]], objects[[4]], objects[[5]], objects[[6]], objects[[7]], objects[[8]], objects[[9]], objects[[10]], objects[[11]],
      objects[[12]], objects[[13]], objects[[14]], objects[[15]], objects[[16]], objects[[17]], objects[[18]], objects[[19]], objects[[20]], objects[[21]], objects[[22]], objects[[23]]),
 add.cell.ids = ids
)

metadata <- merge(combined@meta.data, cistopic@cell.data[,15:ncol(cistopic@cell.data)], by.x='sample_barcode', by.y='row.names', sort = FALSE)

rownames(metadata) <- rownames(combined@meta.data)

combined@meta.data <- metadata

saveRDS(combined, paste0('/lustre/groups/ml01/workspace/laura.martens/data/aerts_fly_brain/combined.rds'))  

combined <- readRDS(paste0('/lustre/groups/ml01/workspace/laura.martens/data/aerts_fly_brain/combined.rds'))  

X = t(combined[['ATAC']]@data)

### Save as anndata

# Create adata - Gene activity in adata.X and peaks in adata.obsm
adata <- anndata::AnnData(X = X)

adata

# Add meta data
adata$obs = combined@meta.data


# Save to disk 
adata$write_h5ad("/lustre/groups/ml01/workspace/laura.martens/data/aerts_fly_brain/All_timepoints.h5ad",
                compression = "gzip")