from sacred import Experiment
import seml

import os
import poisson_atac as patac
from poisson_atac.utils import evaluate_embedding

import pandas as pd
import scanpy as sc

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
    def init_dataset(self, dataset, batch, data_path):
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
            self.adata = patac.data.load_neurips(data_path, batch=batch, only_train=False)
        elif dataset == "satpathy":
            self.adata = patac.data.load_hematopoiesis(data_path)
            

    @ex.capture(prefix="model")
    def init_model(self, model_type: str, data_path:str):
        self.model_type = model_type
        if self.model_type == 'scale':
            self.embedding_path = os.path.join(data_path, model_type, 'adata.h5ad')
        else:
            self.embedding_path = os.path.join(data_path, model_type, 'embedding.csv')

    @ex.capture(prefix="setup")
    def setup_adata(self, batch_key, label_key):
        self.label_key = label_key
        self.batch_key = batch_key
        
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.setup_adata()
    
    @ex.capture(prefix="")
    def run(self):
       #For the benchmarking methods we are only evaluating the latent embedding
        
        # Latent space evaluation:
        if self.model_type == 'scale':
            adata = sc.read(self.embedding_path)
            X_emb = adata.obsm['latent']
        else:
            X_emb = pd.read_csv(self.embedding_path, index_col=0).values
        
        # We evaluate integration metrics when we have more than one batch
        if ("pseudotime_order_ATAC" in self.adata.obs.columns):
            mode = "extended"
        elif ("pseudotime_order_ATAC" not in self.adata.obs.columns):
            mode = "fast"
        else:
            mode="basic"
            
        metrics = evaluate_embedding(self.adata, X_emb, self.label_key, batch_key=self.batch_key, mode=mode)
               
        results = {
            'embedding': metrics,
            'nmi': metrics.loc['NMI_cluster/label', 0],
            'ari': metrics.loc['ARI_cluster/label', 0],
            'model_path': self.embedding_path
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
