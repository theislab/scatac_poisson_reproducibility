from sacred import Experiment
import seml

import os
import poisson_atac as patac
from poisson_atac.utils import scvi_random_split

import pandas as pd
import numpy as np
import torch
import scanpy as sc

from scale import SCALE
from scale.dataset import (
    SingleCellDataset,
    preprocessing_atac,
    CHUNK_SIZE,
)

from torch.utils.data import DataLoader

ex = Experiment()
seml.setup_logger(ex)


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
    processed=False,
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

    if not processed:
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


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @ex.capture(prefix="data")
    def init_dataset(self, dataset, seed):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the
        respective sub-dictionary, enabling a modular experiment design.
        """
        self.dataset = dataset
        if dataset == "neurips":
            self.adata = patac.data.load_neurips(batch=None, only_train=False)
        elif dataset == "satpathy":
            self.adata = patac.data.load_hematopoiesis()
        elif dataset == "aerts":
            self.adata = patac.data.load_aerts()
        elif dataset == "trapnell":
            self.adata = sc.read(
                "/s/project/poisson_atac/anndata/adata_preprocessed.h5ad"
            )

        # Subset to training batch
        self.seed = seed
        self.adata = scvi_random_split(
            self.adata, seed=seed, train_size=0.8, validation_size=0.1
        )

    @ex.capture(prefix="setup")
    def setup_adata(self, batch_key):
        self.batch_key = batch_key

    @ex.capture(prefix="model")
    def setup_model(self, model_type):
        self.model_type = model_type

    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.setup_adata()

    @ex.capture(prefix="training")
    def run(self, save_path, model_path):
        dataset_celltype_mapping = {
            "neurips": 22,
            "satpathy": 31,
            "aerts": 74,
            "trapnell": 78,
        }

        # ================== Default code from SCALE.py ================== #
        # copy default from SCALE.py
        batch_size = 32
        gpu = 0
        n_feature = 30000
        k = dataset_celltype_mapping[self.dataset]
        lr = 0.0002
        max_iter = 30000
        min_peaks = 0
        min_cells = 0
        latent = 10
        encode_dim = [1024, 128]
        decode_dim = []
        pretrain = False
        weight_decay = 5e-4
        verbose = False

        model_path = os.path.join(model_path, f"seed{self.seed}")

        # create the directory if it does not exist
        os.makedirs(model_path, exist_ok=True)

        # start the SCALE.py script
        # Set random seed

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():  # cuda device
            device = "cuda"
            torch.cuda.set_device(gpu)
            print("cuda is available")
        else:
            device = "cpu"

        print(
            "\n**********************************************************************"
        )
        print("  SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction")
        print(
            "**********************************************************************\n"
        )

        adata, trainloader, testloader = load_dataset(
            self.adata,
            batch_categories=None,
            join="inner",
            batch_key="batch",
            batch_name="batch",
            min_genes=0,
            min_cells=0,
            batch_size=batch_size,
            n_top_genes=n_feature,
            log=None,
            processed=(self.dataset == "trapnell"),
        )

        cell_num = adata.shape[0]
        input_dim = adata.shape[1]

        #     if args.n_centroids is None:
        #         k = estimate_k(adata.X.T)
        #         print('Estimate k = {}'.format(k))
        #     else:
        #         k = args.n_centroids

        outdir = model_path + "/"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        print("\n======== Parameters ========")
        print(
            "Cell number: {}\nPeak number: {}\nn_centroids: {}\nmax_iter: {}\nbatch_size: {}\ncell filter by peaks: {}\npeak filter by cells: {}".format(
                cell_num,
                input_dim,
                k,
                max_iter,
                batch_size,
                min_peaks,
                min_cells,
            )
        )
        print("============================")

        dims = [input_dim, latent, encode_dim, decode_dim]
        model = SCALE(dims, n_centroids=k)
        print(model)

        if not pretrain:
            print("\n## Training Model ##")
            model.init_gmm_params(testloader)
            model.fit(
                trainloader,
                lr=lr,
                weight_decay=weight_decay,
                verbose=verbose,
                device=device,
                max_iter=max_iter,
                #                   name=name,
                outdir=outdir,
            )
            torch.save(
                model.state_dict(), os.path.join(outdir, "model.pt")
            )  # save model
        else:
            print("\n## Loading Model: {}\n".format(pretrain))
            model.load_model(pretrain)
            model.to(device)

        ### output ###
        print("outdir: {}".format(outdir))
        # 1. latent feature
        X_emb = model.encodeBatch(testloader, device=device, out="z")

        X_emb = pd.DataFrame(X_emb, index=adata.obs_names)

        X_emb.to_csv(os.path.join(save_path, f"embedding_seed_{self.seed}.csv"))

        results = {
            "save_path": save_path,
            "model_path": model_path,
        }
        return results


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print("get_experiment")
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.run()
