from sacred import Experiment
import seml

import os
import poisson_atac as patac
from poisson_atac.utils import scvi_random_split

import pandas as pd
import numpy as np
import subprocess

from torch.utils.data import DataLoader

ex = Experiment()
seml.setup_logger(ex)


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
    def init_dataset(self, dataset, seed, save_path, replacement):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the
        respective sub-dictionary, enabling a modular experiment design.
        """
        self.dataset = dataset
        self.replacement = replacement
        print(f"Replacement: {replacement}")
        if dataset == "neurips":
            self.adata = patac.data.load_neurips(batch=None, only_train=False)
        elif dataset == "satpathy":
            self.adata = patac.data.load_hematopoiesis()
        elif dataset == "aerts":
            self.adata = patac.data.load_aerts()
        elif dataset == "trapnell":
            self.adata = patac.data.load_trapnell()

        # Subset to training batch
        self.seed = seed
        self.adata = scvi_random_split(
            self.adata, seed=seed, train_size=0.8, validation_size=0.1
        )

        self.save_path = save_path
        self.adata.write(os.path.join(save_path, f"adata_{self.seed}.h5ad"))

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
    def run(self, model_path):
        result = subprocess.run(
            [
                "/opt/modules/i12g/anaconda/envs/signac/bin/Rscript",
                "/data/nasif12/home_if12/martensl/github_repos/scatac_poisson_private/notebooks/benchmark/signac/run_signac.R",
                os.path.join(self.save_path, f"adata_{self.seed}.h5ad"),
                os.path.join(self.save_path, f"embedding_seed_{self.seed}.csv"),
                os.path.join(self.save_path, f"embedding_harmony_seed_{self.seed}.csv"),
                self.replacement[0][0],
                self.replacement[0][1],
                self.batch_key,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"Output:{result.stderr}")
        print(f"Output:{result.stdout}")
        # Rscript run_signac.R /data/ceph/hdd/project/node_08/poisson_atac/anndata/openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2.h5ad test.csv test2.csv "("-",":")" batch
        # remove tmp files
        os.remove(os.path.join(self.save_path, f"adata_{self.seed}.h5ad"))

        results = {
            "save_path": self.save_path,
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
