from sacred import Experiment
import seml
import os

import subprocess
import anndata as ad
from poisson_atac.seml import evaluation_table

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
    def init_dataset(self, data_path, data_split):
          self.data_split=data_split  
          self.train_adata_mod1 = os.path.join(data_path, f"openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod1.h5ad")
          self.test_adata_mod1 = os.path.join(data_path, f"openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_test_mod1.all.h5ad")
          self.train_adata_mod2 = os.path.join(data_path, f"openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod2_split_{data_split}.h5ad")
          self.test_adata_mod2 = os.path.join(data_path, f"openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_test_mod2_split_{data_split}.h5ad")

    @ex.capture(prefix="model")
    def init_model(self, method: str):
        self.method = method
        if method == "LS_lab":
            from LS_lab.script import run
            self.fnc = run

        
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()

    
    @ex.capture(prefix="training")
    def run(self, save_path):
        #define correct save path
        output_dir = os.path.join(save_path, self.method)
        os.makedirs(output_dir, exist_ok=True)
        output_file = '.'.join([os.path.basename(self.test_adata_mod2).split('.')[0], self.method, f'split_{self.data_split}', 'output.h5ad'])
        output = os.path.join(output_dir, output_file)
        

        self.fnc(self.train_adata_mod1,
            self.train_adata_mod2,
            self.test_adata_mod1,
            self.test_adata_mod2,
            output
            )
        
        # Read in true data
        adata = ad.read(self.test_adata_mod2)
        
        # Read in predicted adata
        adata_pred = ad.read(output)
        
        test_cells = evaluation_table(y_true=adata.X.A, predictions={"Model": adata_pred.X.A}, bce=False) # They have negative values in y_pred, bce throws error
   
        results = {
            'test_cells': test_cells,
            'average_precision': test_cells.loc['average_precision', 'Model'],
            'rmse': test_cells.loc['rmse', 'Model'],
            'bce': test_cells.loc['bce', 'Model'],
            'output_path': output
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
