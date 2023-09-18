import seml
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata
import torch
import scib
import sklearn
import pandas as pd
import numpy as np


sns.set_style("whitegrid")
matplotlib.rcParams["font.size"] = 16


model_type_map = {
    "poissonvi": "Poisson VAE",
    "binaryvi": "Binary VAE",
    "peakvi": "PeakVI",
    "cistopic": "cisTopic",
    "cistopic_harmony": "cisTopic (Harmony)",
    "signac": "Signac",
    "signac_harmony": "Signac (Harmony)",
    "scale": "SCALE",
}

dataset_map_simple = {
    "neurips": "10x Human NeurIPS",
    "satpathy": "10x Human Satpathy",
    "aerts": "10x Fly",
    "trapnell": "sci-ATAC-seq3 Human",
}


def compute_embedding(adata, X_emb):
    adata.obsm["X_emb"] = X_emb

    if "X_umap" in adata.obsm.keys():
        adata.obsm.pop("X_umap")

    if "umap" in adata.obsm.keys():
        adata.obsm.pop("umap")

    if "neighbors" in adata.uns.keys():
        adata.uns.pop("neighbors")

    sc.pp.neighbors(adata, use_rep="X_emb")
    sc.tl.umap(adata)


def evaluate_test_cells(model, adata, cell_idx=None, **kwargs):
    """
    Evaluate performance on test peaks
    ----------
    model_dir
        Path to saved trained model
    model_dir2
        Path to peakvi model.
    adata
        adata file with chrom_idx for test_chrom
    """
    if not isinstance(cell_idx, np.ndarray):
        print("Using test set of model")
        cell_idx = model.test_indices
    y_true = adata[cell_idx].X.A
    predictions = setup_prediction_dict(adata, model, cell_idx, **kwargs)

    results = evaluation_table(y_true, predictions)
    return results


def setup_prediction_dict(adata, model, cell_idx, **kwargs):
    y = model.get_accessibility_estimates(
        adata, indices=cell_idx, return_numpy=True, **kwargs
    )

    predictions = {"Model": y}

    return predictions


def bce_loss(y_true, y_pred):
    try:
        bce = torch.nn.BCELoss(reduction="none")(
            torch.from_numpy(y_pred).float(), torch.from_numpy(y_true).float()
        ).sum(dim=-1)
    except:
        bce = torch.nn.BCELoss(reduction="none")(
            torch.from_numpy(y_pred), torch.from_numpy(y_true)
        ).sum(dim=-1)
    return bce.sum().numpy() / y_true.shape[0]


def poisson_loss(y_true, y_pred):
    try:
        pl = torch.nn.PoissonNLLLoss(reduction="none", log_input=False, full=True)(
            torch.from_numpy(y_pred), torch.from_numpy(y_true).float()
        ).sum(dim=-1)
    except:
        pl = torch.nn.PoissonNLLLoss(reduction="none", log_input=False, full=True)(
            torch.from_numpy(y_pred), torch.from_numpy(y_true).float()
        ).sum(dim=-1)
    return pl.sum().numpy() / y_true.shape[0]


def evaluation_table(y_true, predictions, bce=True):
    results = {}
    for key, y_pred in predictions.items():
        ap = sklearn.metrics.average_precision_score(
            y_true.ravel(), y_pred.ravel(), pos_label=1
        )
        auroc = sklearn.metrics.roc_auc_score(y_true.ravel(), y_pred.ravel())
        rmse = sklearn.metrics.mean_squared_error(
            y_true.ravel(), y_pred.ravel(), squared=False
        )
        if bce:
            bce = bce_loss(y_true, y_pred)
        else:
            bce = None
        results.update(
            {key: {"average_precision": ap, "auroc": auroc, "rmse": rmse, "bce": bce}}
        )
    results = pd.DataFrame(results)
    return results


def evaluate_embedding(adata, X_emb, labels_key, batch_key="batch", mode="basic"):
    adata.obsm["X_emb"] = X_emb
    if "neighbors" in adata.uns.keys():
        adata.uns.pop("neighbors")

    if mode == "basic":
        metrics = scib.metrics.metrics(
            adata,
            adata,
            batch_key=labels_key,
            label_key=labels_key,
            nmi_=True,
            ari_=True,
            embed="X_emb",
        )
    elif mode == "extended":
        adata_int = anndata.AnnData(X_emb, obs=adata.obs)
        adata_int.obsm["X_emb"] = X_emb
        metrics = scib.metrics.metrics(
            adata,
            adata_int,
            batch_key=batch_key,
            label_key=labels_key,
            trajectory_=True,
            nmi_=True,
            embed="X_emb",
            ari_=True,
            silhouette_=True,
            isolated_labels_=True,
            isolated_labels_f1_=True,
            isolated_labels_asw_=True,
            graph_conn_=True,
            pcr_=True,
            lisi_graph_=True,
            ilisi_=True,
            clisi_=True,
            verbose=True,
        )
    elif mode == "fast":
        adata_int = anndata.AnnData(X_emb, obs=adata.obs)
        adata_int.obsm["X_emb"] = X_emb
        metrics = scib.metrics.metrics(
            adata,
            adata_int,
            batch_key=batch_key,
            label_key=labels_key,
            trajectory_=False,
            nmi_=True,
            embed="X_emb",
            ari_=True,
            silhouette_=True,
            isolated_labels_=True,
            isolated_labels_f1_=True,
            isolated_labels_asw_=True,
            graph_conn_=True,
            pcr_=True,
            lisi_graph_=True,
            ilisi_=True,
            clisi_=True,
            verbose=True,
        )

    return metrics


def evaluate_counts(model, adata, cell_idx=None, **kwargs):
    if cell_idx is None:
        print("Using test set of model")
        cell_idx = model.test_indices

    y_true = adata[cell_idx].layers["counts"].A

    y_pred = model.get_accessibility_estimates(
        adata, indices=cell_idx, return_numpy=True, **kwargs
    )
    r2 = sklearn.metrics.r2_score(y_true.ravel(), y_pred.ravel())
    pl = poisson_loss(y_true, y_pred)

    results = {"Model": {"r2_score": r2, "poisson_loss": pl}}
    results = pd.DataFrame(results)
    return results


# SEML utils
def load_config(seml_collection, model_hash):
    results_df = seml.get_results(
        seml_collection,
        to_data_frame=True,
        fields=["config", "config_hash"],
        states=["COMPLETED"],
        filter_dict={"config_hash": model_hash},
    )
    experiment = results_df.apply(
        lambda exp: {
            "hash": exp["config_hash"],
            "seed": exp["config.seed"],
            "_id": exp["_id"],
        },
        axis=1,
    )
    assert len(experiment) == 1
    experiment = experiment[0]
    collection = seml.database.get_collection(seml_collection)
    config = collection.find_one({"_id": experiment["_id"]})["config"]
    config["config_hash"] = model_hash
    return config


def get_model_path(seml_collection, model_hash):
    results = seml.get_results(
        seml_collection,
        to_data_frame=True,
        fields=["result"],
        states=["COMPLETED"],
        filter_dict={"config_hash": model_hash},
    )
    path = results["result.model_path"].values[0]
    path = path.replace("storage", "lustre")  # adapt to cluster changes
    return path


def load_experiment(seml_collection, model_hash, get_experiment_fn, load_model=True):
    config = load_config(seml_collection, model_hash)
    ex = get_experiment_fn()
    ex.init_dataset(**config["data"])
    ex.init_model(**config["model"])
    ex.setup_adata(**config["setup"])

    if load_model:
        model = ex.model.load(
            get_model_path(seml_collection, model_hash), adata=ex.adata
        )
    else:
        model = None
    return ex, model, config


def load_data(dataset, model, best, get_experiment, seml_database):
    model_hash = best.loc[
        (dataset_map_simple[dataset], model_type_map[model]), "config_hash"
    ]
    ex, model, config = load_experiment(seml_database, model_hash, get_experiment)

    X_emb = model.get_latent_representation(ex.adata)
    compute_embedding(ex.adata, X_emb)


def plot_umap(adata, model, dataset, color_list=["cell_type", "batch"]):
    for color in color_list:
        sc.pl.umap(adata, color=color, save=f"{dataset}_{model}_{color}.png")


def validate_data_split(n_samples: int, train_size: float, validation_size: float):
    # From https://github.com/scverse/scvi-tools/blob/2cf00ecef4a04dfa2d35fb0fd3cb3aa0eb101330/scvi/model/base/_training_mixin.py#L10
    """Check data splitting parameters and return n_train and n_val.
    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    from math import ceil, floor

    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, train_size={} and validation_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, train_size, validation_size)
        )

    return n_train, n_val


def scvi_random_split(adata, seed, train_size, validation_size):
    # Adapted from https://github.com/scverse/scvi-tools/blob/2cf00ecef4a04dfa2d35fb0fd3cb3aa0eb101330/scvi/model/base/_training_mixin.py#L10
    import scvi

    n_train, n_val = validate_data_split(adata.n_obs, train_size, validation_size)

    """Split indices in train/test/val sets."""
    random_state = np.random.RandomState(seed=seed)
    permutation = random_state.permutation(adata.n_obs)
    train_idx = permutation[n_val : (n_val + n_train)]

    # return adata of train index like scvi
    return adata[train_idx, :]
