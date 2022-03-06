import logging
from sacred import Experiment
import numpy as np
import seml

import os
import anndata as ad
import scanpy as sc
import pandas as pd
from scipy import sparse
import scvi
import poisson_atac as patac
from poisson_atac.seml import evaluation_table, evaluate_embedding
import sklearn.metrics

import wandb
from pytorch_lightning.loggers import WandbLogger
import uuid
from scvi.dataloaders._data_splitting import validate_data_split
import numpy as np
import scvi

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
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

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
    def init_dataset(self, dataset, batch):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
            
        self.dataset = dataset
        self.batch = (batch if batch is not None else 'NONE')
        if dataset == "neurips":
            self.adata = patac.data.load_neurips(batch=batch)
        elif dataset == "hematopoiesis":
            self.adata = patac.data.load_hematopoiesis()
            

    @ex.capture(prefix="setup")
    def setup_adata(self, batch_key, label_key):
        self.label_key = label_key
        self.batch_key = batch_key
    
    @ex.capture(prefix="model")
    def init_model(self, model_type: str):
        self.model_type = model_type
        
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.setup_adata()
    
    @ex.capture(prefix="training")
    def run(self, save_path):

        n_train, n_val = validate_data_split(
            self.adata.n_obs, 0.8, 0.1
        )

        random_state = np.random.RandomState(seed=scvi.settings.seed)
        permutation = random_state.permutation(self.adata.n_obs)
        test_idx = permutation[(n_val + n_train) :]
        
        #Peak and cell evaluations:
        if self.dataset == "neurips":
            if self.batch is None:
                model_path = os.path.join(save_path, f"neurips_pred_matrix.txt")
                embedding_path = os.path.join(save_path, f"neurips.txt")
            else:
                model_path = os.path.join(save_path, f"{self.batch}_pred_matrix.txt")
                embedding_path = os.path.join(save_path, f"{self.batch}.txt")
        elif self.dataset == "hematopoiesis":
            model_path = os.path.join(save_path, f"{self.dataset}_pred_matrix.txt")
            embedding_path = os.path.join(save_path, f"{self.dataset}.txt")
            
        y_pred =  pd.read_csv(model_path, sep = "\t", header = None, skiprows=1, index_col=0).values.T
        test_cells = evaluation_table(y_true=self.adata.X.A[test_idx], predictions={"Model": y_pred[test_idx]})
        
        # Latent space evaluation:
        X_emb = pd.read_csv(embedding_path, sep = " ", header = None, skiprows=1, index_col=0).values.T
        metrics = evaluate_embedding(self.adata, X_emb, self.label_key)

        
        results = {
            'test_cells': test_cells,
            'embedding': metrics,
            'average_precision': test_cells.loc['average_precision', 'Model'],
            'rmse': test_cells.loc['rmse', 'Model'],
            'bce': test_cells.loc['bce', 'Model'],
            'nmi': metrics.loc['NMI_cluster/label', 0],
            'ari': metrics.loc['ARI_cluster/label', 0],
            'model_path': model_path
        }
        return results

# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment

# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.run()
