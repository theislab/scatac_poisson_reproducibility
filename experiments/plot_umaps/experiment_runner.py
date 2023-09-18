from sacred import Experiment
import seml

import os
import poisson_atac as patac
from poisson_atac.utils import compute_embedding, scvi_random_split

import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt


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
    def init_dataset(self, dataset, seed):
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

        # Subset to training batch
        self.seed = seed
        self.adata = scvi_random_split(
            self.adata, seed=seed, train_size=0.8, validation_size=0.1
        )

    @ex.capture(prefix="model")
    def init_model(self, model_type: str, save_path: str):
        self.model_type = model_type

        # check if modeltype contains "harmony"
        if "harmony" in model_type:
            self.embedding_path = os.path.join(
                save_path,
                self.dataset,
                self.model_type.split("_")[0],
                f"embedding_harmony_seed_{self.seed}.csv",
            )
        else:
            self.embedding_path = os.path.join(
                save_path,
                self.dataset,
                self.model_type,
                f"embedding_seed_{self.seed}.csv",
            )

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

    @ex.capture(prefix="training")
    def run(self, fig_path: str):
        # For the benchmarking methods we are only evaluating the latent embedding

        # Load embedding
        X_emb = pd.read_csv(self.embedding_path, index_col=0)

        # make sure that ordering is the same
        X_emb = X_emb.loc[self.adata.obs.index]
        # get numpy array
        X_emb = X_emb.values

        # compute umap
        sc.settings.figdir = fig_path
        sc.set_figure_params(
            dpi_save=300,
            vector_friendly=False,
            figsize=(5, 5),
            frameon=False,
        )
        compute_embedding(self.adata, X_emb)

        # save umap
        umap = pd.DataFrame(
            self.adata.obsm["X_umap"],
            columns=["UMAP_1", "UMAP_2"],
            index=self.adata.obs_names,
        )

        umap_path = f"/s/project/poisson_atac/panels/revision2/Figure2/data/embeddings/{self.dataset}_{self.model_type}_umap.csv"
        umap.to_csv(umap_path)

        # plot umap for batch and label keys
        for color in [self.batch_key, self.label_key]:
            ax = sc.pl.umap(
                self.adata,
                color=color,
                save=f"{self.dataset}_{self.model_type}_{color}.pdf",
                legend_loc=None,
                show=False,
            )
            sc.pl.umap(
                self.adata,
                color=color,
                save=f"{self.dataset}_{self.model_type}_{color}.png",
                legend_loc=None,
            )
            # Plot the legend separately
            ax = sc.pl.umap(self.adata, color=color, show=False)
            handles, labels = ax.get_legend_handles_labels()
            plt.figure()
            plt.axis("off")  # Turn off the axis
            plt.legend(handles, labels, ncol=5, loc="center", frameon=False)

            # Show the plots
            plt.tight_layout()
            plt.savefig(
                os.path.join(fig_path, f"{self.dataset}_{color}_legend.pdf"),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(fig_path, f"{self.dataset}_{color}_legend.png"),
                bbox_inches="tight",
            )
            plt.show()

        results = {
            "fig_path": os.path.join(
                fig_path, f"{self.dataset}_{self.model_type}_{color}"
            ),
            "umap_path": umap_path,
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
