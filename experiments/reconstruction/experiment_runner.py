from sacred import Experiment
import seml

import os
import scvi
import poisson_atac as patac
from poisson_atac.utils import evaluate_test_cells

import pandas as pd
import numpy as np

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
    def init_dataset(self, dataset, batch):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
        if isinstance(batch, str):
            batch = [batch]

        self.dataset = dataset
        self.batch = batch if batch is not None else "NONE"
        if dataset == "neurips":
            self.adata = patac.data.load_neurips(batch=batch, only_train=False)
        elif dataset == "satpathy":
            self.adata = patac.data.load_hematopoiesis()
        elif dataset == "aerts":
            self.adata = patac.data.load_aerts()
        elif dataset == "trapnell":
            self.adata = patac.data.load_trapnell()

    @ex.capture(prefix="model")
    def init_model(self, model_type: str):
        self.model_type = model_type
        if model_type == "peakvi":
            self.model = scvi.model.PEAKVI
        elif model_type == "poissonvi":
            self.model = patac.model.PoissonVI
        elif model_type == "binaryvi":
            self.model = patac.model.BinaryVI

    @ex.capture(prefix="optimization")
    def init_optimizer(self, regularization: dict):
        self.weight_decay = regularization["weight_decay"]
        self.learning_rate = regularization["learning_rate"]

    @ex.capture(prefix="scvi")
    def init_seed(self, seed):
        self.seed = seed
        scvi.settings.seed = seed

    @ex.capture(prefix="setup")
    def setup_adata(self, layer, batch_key, label_key, model_params):
        self.label_key = label_key
        self.batch_key = batch_key
        setup_params = {}

        self.model.setup_anndata(
            self.adata, layer=layer, batch_key=batch_key, **setup_params
        )
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
        self.init_seed()

    @ex.capture(prefix="training")
    def run(self, max_epochs, save_path, model_path, project_name):
        # train model
        self.model.train(
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            max_epochs=max_epochs,
            train_size=0.8,
            validation_size=0.1,
        )

        # save model
        print("Save model")
        ext = "{}_{}_{}".format(self.dataset, self.model_type, self.seed)
        model_path = os.path.join(model_path, self.dataset, self.model_type)
        model_path = os.path.join(model_path, ext)
        self.model.save(dir_path=model_path, overwrite=True)

        # Peak and cell evaluations:
        print("Evaluate test cells")
        if self.model_type == "peakvi":
            kwargs = {"normalize_cells": True, "normalize_regions": True}
        elif self.model_type == "poissonvi":
            kwargs = {"library_size": "latent", "binarize": True}
        elif self.model_type == "binaryvi":
            kwargs = {"library_size": "latent"}

        if self.dataset != "trapnell":
            test_cells = evaluate_test_cells(self.model, self.adata, **kwargs)
        else:
            cell_idx = self.model.test_indices
            test_cells_all = [
                evaluate_test_cells(self.model, self.adata, cell_idx=idx, **kwargs)
                for idx in np.array_split(cell_idx, 10)
            ]
            test_cells = pd.concat(test_cells_all, axis=1)

        # save embedding
        print("Save embeddings")
        X_emb = pd.DataFrame(
            self.model.get_latent_representation(self.adata), index=self.adata.obs_names
        )
        save_path = os.path.join(
            save_path, self.dataset, self.model_type, f"embedding_seed_{self.seed}.csv"
        )
        X_emb.to_csv(save_path)

        results = {
            "test_cells": test_cells,
            "average_precision": (
                test_cells.loc["average_precision", "Model"]
                if self.dataset != "trapnell"
                else test_cells.loc["average_precision"].mean()
            ),
            "rmse": (
                test_cells.loc["rmse", "Model"]
                if self.dataset != "trapnell"
                else test_cells.loc["rmse"].mean()
            ),
            "bce": (
                test_cells.loc["bce", "Model"]
                if self.dataset != "trapnell"
                else test_cells.loc["bce"].mean()
            ),
            "model_path": model_path,
            "save_path": save_path,
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
