
from sacred import Experiment
import seml

import os
import anndata as ad
import poisson_atac as patac
from poisson_atac.utils import evaluation_table

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
    def init_dataset(self, dataset, batch, data_split, data_path):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
        self.dataset=dataset
        if isinstance(batch, str):
            batch = [batch]
        self.batch=batch
        adata = patac.data.load_neurips(gex=True, batch=batch, only_train=False)
        
        #subset on 10000 peaks from Neurips
        train_adata_mod2 = ad.read(os.path.join(data_path, f"openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod2_split_{data_split}.h5ad")) 
        adata = adata[:, adata.var_names.isin(train_adata_mod2.var_names)]
        
        self.adata = adata[adata.obs.is_train].copy()
        self.adata_test = adata[~adata.obs.is_train].copy()
        del train_adata_mod2
        del adata

    @ex.capture(prefix="model")
    def init_model(self, model_type: str):
        self.model_type = model_type
        if model_type == "gex":
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
            
        self.model.setup_anndata(self.adata, layer=layer, batch_key=batch_key, **setup_params) 
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
    def run(self, max_epochs, save_path, project_name, final):
        ## Setup logger
        logger = WandbLogger(project=project_name, name=f"{self.model_type}_{self.dataset}_{self.batch}")
        #train model
        self.model.train(logger=logger, 
                         lr=self.learning_rate,
                         weight_decay=self.weight_decay,
                         max_epochs=max_epochs,
                         train_size=0.9, 
                         validation_size=0.1
                         )
        wandb.finish()
        
        # save model
        ext = '{}_{}'.format(project_name, uuid.uuid1())
        model_path=os.path.join(save_path, ext)
        self.model.save(dir_path=model_path)   
        
        #Peak and cell evaluations with true library size and mean library size:
        if self.model_type == "gex":
            size_factor = self.adata.layers["counts"].sum(axis =1)
            kwargs_true = {"library_size": "latent", "binarize": True}
            kwargs_mean = {"library_size": size_factor.mean(), "binarize": True}
        elif self.model_type == "gex_binary":
            size_factor = self.adata.X.A.sum(axis =1)
            kwargs_true = {"library_size": "latent"}
            kwargs_mean = {"library_size": size_factor.mean()}

        if final:
            y_mean = self.model.get_accessibility_estimates(self.adata_test, return_numpy=True, **kwargs_mean) # with mean training size factor
            y_pred = self.model.get_accessibility_estimates(self.adata_test, return_numpy= True, **kwargs_true)
            predictions = {'Model': y_pred, 'Mean': y_mean}
            test_cells = evaluation_table(y_true=self.adata_test.X.A, predictions=predictions, bce=False)
        else:
            val_indices = self.model.validation_indices
            y_mean = self.model.get_accessibility_estimates(self.adata, return_numpy=True, indices=val_indices, **kwargs_mean) # with mean training size factor
            y_pred = self.model.get_accessibility_estimates(self.adata, return_numpy= True, indices=val_indices, **kwargs_true)
            predictions = {'Model': y_pred, 'Mean': y_mean}
            test_cells = evaluation_table(y_true=self.adata[val_indices].X.A, predictions=predictions, bce=False)
            
        results = {
            'test_cells': test_cells,
            'average_precision': test_cells.loc['average_precision', 'Model'],
            'rmse': test_cells.loc['rmse', 'Model'],
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
