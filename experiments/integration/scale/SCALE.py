#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Sat 28 Apr 2018 08:31:29 PM CST

# File Name: SCALE.py
# Description: Single-Cell ATAC-seq Analysis via Latent feature Extraction.
    Input:
        scATAC-seq data
    Output:
        1. latent feature
        2. cluster assignment
        3. imputation data
"""

import torch

import numpy as np
import os
import scanpy as sc
import argparse

from scale import SCALE
from scale.dataset import (
    SingleCellDataset,
    preprocessing_atac,
    CHUNK_SIZE,
)
from scale.utils import binarization

from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

import poisson_atac as patac
from poisson_atac.utils import scvi_random_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction"
    )
    parser.add_argument("--data_list", "-d", type=str, nargs="+", default=[])
    parser.add_argument(
        "--n_centroids", "-k", type=int, help="cluster number", default=30
    )
    parser.add_argument(
        "--outdir", "-o", type=str, default="output/", help="Output path"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print loss of training process"
    )
    parser.add_argument(
        "--pretrain", type=str, default=None, help="Load the trained model"
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--gpu",
        "-g",
        default=0,
        type=int,
        help="Select gpu device number when training",
    )
    parser.add_argument(
        "--seed", type=int, default=18, help="Random seed for repeat results"
    )
    parser.add_argument(
        "--encode_dim",
        type=int,
        nargs="*",
        default=[1024, 128],
        help="encoder structure",
    )
    parser.add_argument(
        "--decode_dim", type=int, nargs="*", default=[], help="encoder structure"
    )
    parser.add_argument("--latent", "-l", type=int, default=10, help="latent layer dim")
    parser.add_argument(
        "--min_peaks",
        type=float,
        default=100,
        help="Remove low quality cells with few peaks",
    )
    parser.add_argument(
        "--min_cells", type=float, default=0.01, help="Remove low quality peaks"
    )
    parser.add_argument(
        "--n_feature",
        type=int,
        default=30000,
        help="Keep the number of highly variable peaks",
    )
    parser.add_argument(
        "--log_transform", action="store_true", help="Perform log2(x+1) transform"
    )
    parser.add_argument(
        "--max_iter", "-i", type=int, default=30000, help="Max iteration"
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--impute", action="store_true", help="Save the imputed data in layer impute"
    )
    parser.add_argument(
        "--binary", action="store_true", help="Save binary imputed data in layer binary"
    )
    parser.add_argument("--embed", type=str, default="UMAP")
    parser.add_argument("--reference", type=str, default="celltype")
    parser.add_argument("--cluster_method", type=str, default="leiden")
    parser.add_argument(
        "--preprocessed", action="store_true", help="Data is already processed"
    )

    args = parser.parse_args()

    # Set random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():  # cuda device
        device = "cuda"
        torch.cuda.set_device(args.gpu)
        print("cuda is available")
    else:
        device = "cpu"
    batch_size = args.batch_size

    print("\n**********************************************************************")
    print("  SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction")
    print("**********************************************************************\n")

    # ===========================================================================
    # This part has been modified to load data directly
    # ===========================================================================
    def load_dataset(
        adata,
        batch_categories=None,
        join="inner",
        batch_key="batch",
        batch_name="batch",
        min_genes=600,
        min_cells=0.01,
        n_top_genes=30000,
        batch_size=64,
        chunk_size=CHUNK_SIZE,
        log=None,
        transpose=False,
    ):
        """
        Load dataset with preprocessing
        """
        if log:
            log.info("Raw dataset shape: {}".format(adata.shape))
        if batch_name != "batch":
            adata.obs["batch"] = adata.obs[batch_name]
        if "batch" not in adata.obs:
            adata.obs["batch"] = "batch"
        adata.obs["batch"] = adata.obs["batch"].astype("category")

        adata = preprocessing_atac(
            adata,
            min_genes=min_genes,
            min_cells=min_cells,
            n_top_genes=n_top_genes,
            chunk_size=chunk_size,
            log=log,
        )
        if log:
            log.info("Processed dataset shape: {}".format(adata.shape))
        scdata = SingleCellDataset(adata)  # Wrap AnnData into Pytorch Dataset
        trainloader = DataLoader(
            scdata, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4
        )
        #     batch_sampler = BatchSampler(batch_size, adata.obs['batch'], drop_last=False)
        testloader = DataLoader(
            scdata, batch_size=batch_size, drop_last=False, shuffle=False
        )
        #     testloader = DataLoader(scdata, batch_sampler=batch_sampler)

        return adata, trainloader, testloader

    print(args.data_list[0])
    if args.data_list[0] == "neurips":
        adata = patac.data.load_neurips(batch=None, only_train=False)
    elif args.data_list[0] == "satpathy":
        adata = patac.data.load_hematopoiesis()
    elif args.data_list[0] == "aerts":
        adata = patac.data.load_aerts()
    elif args.data_list[0] == "trapnell":
        adata = patac.data.load_trapnell()
    # Subset to training batch
    seed = args.seed
    adata = scvi_random_split(adata, seed=seed, train_size=0.8, validation_size=0.1)

    adata, trainloader, testloader = load_dataset(
        adata,
        batch_categories=None,
        join="inner",
        batch_key="batch",
        batch_name="batch",
        min_genes=args.min_peaks,
        min_cells=args.min_cells,
        batch_size=args.batch_size,
        n_top_genes=args.n_feature,
        log=None,
    )
    # ===========================================================================
    # End of modification
    # ===========================================================================

    cell_num = adata.shape[0]
    input_dim = adata.shape[1]

    #     if args.n_centroids is None:
    #         k = estimate_k(adata.X.T)
    #         print('Estimate k = {}'.format(k))
    #     else:
    #         k = args.n_centroids
    lr = args.lr
    k = args.n_centroids

    outdir = args.outdir + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # TODO: change back
    args.max_iter = 3
    print("\n======== Parameters ========")
    print(
        "Cell number: {}\nPeak number: {}\nn_centroids: {}\nmax_iter: {}\nbatch_size: {}\ncell filter by peaks: {}\npeak filter by cells: {}".format(
            cell_num,
            input_dim,
            k,
            args.max_iter,
            batch_size,
            args.min_peaks,
            args.min_cells,
        )
    )
    print("============================")

    dims = [input_dim, args.latent, args.encode_dim, args.decode_dim]
    model = SCALE(dims, n_centroids=k)
    print(model)

    if not args.pretrain:
        print("\n## Training Model ##")
        model.init_gmm_params(testloader)
        model.fit(
            trainloader,
            lr=lr,
            weight_decay=args.weight_decay,
            verbose=args.verbose,
            device=device,
            max_iter=args.max_iter,
            #                   name=name,
            outdir=outdir,
        )
        torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))  # save model
    else:
        print("\n## Loading Model: {}\n".format(args.pretrain))
        model.load_model(args.pretrain)
        model.to(device)

    ### output ###
    print("outdir: {}".format(outdir))
    # 1. latent feature
    adata.obsm["latent"] = model.encodeBatch(testloader, device=device, out="z")

    # 2. cluster
    # sc.pp.neighbors(adata, n_neighbors=30, use_rep="latent")
    # if args.cluster_method == "leiden":
    #     sc.tl.leiden(adata)
    # elif args.cluster_method == "kmeans":
    #     kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    #     adata.obs["kmeans"] = kmeans.fit_predict(adata.obsm["latent"]).astype(str)

    # #     if args.reference in adata.obs:
    # #         cluster_report(adata.obs[args.reference].cat.codes, adata.obs[args.cluster_method].astype(int))

    # sc.settings.figdir = outdir
    # sc.set_figure_params(dpi=80, figsize=(6, 6), fontsize=10)
    # if args.embed == "UMAP":
    #     sc.tl.umap(adata, min_dist=0.1)
    #     color = [c for c in ["celltype", args.cluster_method] if c in adata.obs]
    #     sc.pl.umap(adata, color=color, save=".pdf", wspace=0.4, ncols=4)
    # elif args.embed == "tSNE":
    #     sc.tl.tsne(adata, use_rep="latent")
    #     color = [c for c in ["celltype", args.cluster_method] if c in adata.obs]
    #     sc.pl.tsne(adata, color=color, save=".pdf", wspace=0.4, ncols=4)

    # if args.impute:
    #     adata.obsm["impute"] = model.encodeBatch(testloader, device=device, out="x")
    # if args.binary:
    #     adata.obsm["impute"] = model.encodeBatch(testloader, device=device, out="x")
    #     adata.obsm["binary"] = binarization(adata.obsm["impute"], adata.X)
    #     del adata.obsm["impute"]

    adata.write(outdir + "adata.h5ad", compression="gzip")