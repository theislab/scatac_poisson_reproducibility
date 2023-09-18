from sacred import Experiment
import seml

import scvi
import poisson_atac as patac

import numpy as np
import sklearn.metrics

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
    def init_dataset(self, dataset):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """

        self.dataset = dataset
        if dataset == "neurips":
            self.adata = patac.data.load_neurips(batch=None, only_train=False)
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

    @ex.capture(prefix="scvi")
    def init_seed(self, seed):
        self.seed = seed
        scvi.settings.seed = seed
        np.random.seed(seed)

    @ex.capture(prefix="setup")
    def setup_adata(self, layer, batch_key, model_path):
        self.batch_key = batch_key
        setup_params = {}

        self.model.setup_anndata(
            self.adata, layer=layer, batch_key=batch_key, **setup_params
        )

        model_path = f"{model_path}/{self.dataset}/{self.model_type}/{self.dataset}_{self.model_type}_{self.seed}"
        self.model = self.model.load(model_path, adata=self.adata)
        print(self.model.module)

    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_seed()
        self.init_model()
        self.setup_adata()

    @ex.capture(prefix="training")
    def run(self):
        # Peak and cell evaluations:
        print("Evaluate test cells")
        if self.model_type == "peakvi":
            kwargs = {"normalize_cells": True, "normalize_regions": True}
        elif self.model_type == "poissonvi":
            kwargs = {"library_size": "latent", "binarize": True}
        elif self.model_type == "binaryvi":
            kwargs = {"library_size": "latent"}

        cell_idx = (
            self.model.test_indices
        )  # np.random.choice(self.model.train_indices, size=10000, replace=False)

        y_pred = self.model.get_accessibility_estimates(
            self.adata,
            indices=cell_idx,
            return_numpy=True,
            **kwargs,
        )
        y_true = self.adata[cell_idx].X.A
        y_true_counts = self.adata[cell_idx].layers["counts"].A

        y_pred_low_counts = y_pred[y_true_counts < 10]
        y_true_low_counts = y_true[y_true_counts < 10]

        ap = sklearn.metrics.average_precision_score(
            y_true.ravel(), y_pred.ravel(), pos_label=1
        )

        ap_low_counts = sklearn.metrics.average_precision_score(
            y_true_low_counts.ravel(), y_pred_low_counts.ravel(), pos_label=1
        )

        results = {
            "average_precision": ap,
            "average_precision_low_counts": ap_low_counts,
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
