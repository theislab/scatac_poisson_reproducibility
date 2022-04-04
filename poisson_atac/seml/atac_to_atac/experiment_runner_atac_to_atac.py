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
from poisson_atac.seml import evaluate_test_cells, evaluate_embedding
import sklearn.metrics

import wandb
from pytorch_lightning.loggers import WandbLogger
import uuid

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
        if isinstance(batch, str):
            batch = [batch]
            
        self.dataset = dataset
        self.batch = (batch if batch is not None else 'NONE')
        if dataset == "neurips":
            self.adata = patac.data.load_neurips(batch=batch)
        elif dataset == "hematopoiesis":
            self.adata = patac.data.load_hematopoiesis()
        elif dataset == "neurips_multiome":
            self.adata = patac.data.load_neurips(gex=True, batch=batch)
            

    @ex.capture(prefix="model")
    def init_model(self, model_type: str):
        self.model_type = model_type
        if model_type == "baseline":
            self.model = patac.model.Baseline
        elif model_type == "count":
            self.model = patac.model.CountPEAKVI
        elif model_type == "peakvi":
            self.model = scvi.model.PEAKVI
        elif model_type == "linear_count":
            self.model = patac.model.LinearCountPEAKVI
        elif model_type == "gex":
            self.model = patac.model.GEXtoATAC
        elif model_type == "gex_binary":
            self.model = patac.model.BinaryGEXtoATAC

    @ex.capture(prefix="optimization")
    def init_optimizer(self, regularization: dict):
        self.weight_decay = regularization['weight_decay']
        self.learning_rate = regularization['learning_rate']

    @ex.capture(prefix="setup")
    def setup_adata(self, layer, batch_key, label_key, model_params):
        self.label_key = label_key
        self.batch_key = batch_key
        if self.model_type == "gex" or self.model_type == "gex_binary":
            setup_params = {'adata_gex_obsm_key': "X_gex"}
        else:
            setup_params = {}
            
        self.model.setup_anndata(self.adata, layer=layer, batch_key=batch_key, **setup_params) # layer =None for peakvi!
        self.model = self.model(self.adata, **model_params)
        print(self.model.module)
        
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.setup_adata()
        self.init_optimizer()
    
    @ex.capture(prefix="training")
    def run(self, max_epochs, save_path, project_name):
        ## Setup logger
        logger = WandbLogger(project=project_name, name=f"{self.model_type}_{self.dataset}_{self.batch}")
        #train model
        self.model.train(logger=logger, 
                         lr=self.learning_rate,
                         weight_decay=self.weight_decay,
                         max_epochs=max_epochs,
                         train_size=0.8, 
                         validation_size=0.1
                         )
        wandb.finish()
        
        # save model
        ext = '{}_{}'.format(project_name, uuid.uuid1())
        model_path=os.path.join(save_path, ext)
        self.model.save(dir_path=model_path)   
        
        #Peak and cell evaluations:
        if self.model_type == "peakvi" or self.model_type == "gex_binary":
            kwargs = {"normalize_cells": True, "normalize_regions": True}
        else:
            kwargs = {}
        test_cells = evaluate_test_cells(self.model, self.adata, **kwargs)
        
        # Latent space evaluation:
        if self.model_type != "baseline":
            X_emb = self.model.get_latent_representation(self.adata)
            metrics = evaluate_embedding(self.adata, X_emb, self.label_key, batch_key=self.batch_key, mode="basic")
        else:
            metrics = pd.DataFrame(index=['NMI_cluster/label', 'ARI_cluster/label'], data={0: np.array([0,0])})
        
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
