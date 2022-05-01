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



matplotlib.style.use("seaborn-colorblind")
sns.set_style("whitegrid")
matplotlib.style.use("seaborn-poster")
matplotlib.rcParams["font.size"] = 16

model_type_map = {
    'gexTrue':"Poisson encoder-decoder\n(observed seq. coverage)" , 
    'gexFalse':"Poisson encoder-decoder\n(mean observed seq. coverage)" , 
    'gex_binaryTrue':"Binary encoder-decoder\n(observed seq. coverage)" , 
    'gex_binaryFalse':"Binary encoder-decoder\n(mean observed seq. coverage)" , 
    'binaryviTrue': "Binary VAE\n(observed seq. coverage)",
    'binaryviFalse': "Binary VAE\n(encoded seq. coverage)",
    'poissonviTrue': "Poisson VAE\n(observed seq. coverage)",
    'poissonviFalse': "Poisson VAE\n(encoded seq. coverage)",
    'nbviTrue': "Neg. Binomial VAE\n(observed seq. coverage)",
    'nbviFalse': "Neg. Binomial VAE\n(encoded seq. coverage)",
    'peakvi': "PeakVI\n(encoded seq. coverage)",
    'LS_lab': 'Neurips winner: LS_lab\n(binary, no seq. coverage)'
}

def compute_embedding(adata, X_emb):
            
    adata.obsm['X_emb'] = X_emb
    
    if 'X_umap' in adata.obsm.keys():
        adata.obsm.pop('X_umap')
    
    if 'umap' in adata.obsm.keys():
        adata.obsm.pop('umap')
        
    if 'neighbors' in adata.uns.keys():
        adata.uns.pop('neighbors')

    sc.pp.neighbors(adata, use_rep='X_emb')
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
    if not cell_idx:
        print("Using test set of model")
        cell_idx = model.test_indices
    y_true = adata[cell_idx].X.A
    predictions = setup_prediction_dict(adata, model, cell_idx, **kwargs)

    results = evaluation_table(y_true, predictions)
    return results

def setup_prediction_dict(adata, model, cell_idx, **kwargs):
    y = model.get_accessibility_estimates(adata, 
                                                 indices=cell_idx, 
                                                 return_numpy=True,
                                                 **kwargs
                                                )
    
    predictions = {'Model': y}
    
    return predictions

#TODO: sometimes fails with floats and sometimes not
def bce_loss(y_true, y_pred): 
    try:
        bce = torch.nn.BCELoss(reduction="none")(torch.from_numpy(y_pred).float(), torch.from_numpy(y_true).float()).sum(dim=-1)
    except:
        bce = torch.nn.BCELoss(reduction="none")(torch.from_numpy(y_pred), torch.from_numpy(y_true)).sum(dim=-1)
    return bce.sum().numpy()/y_true.shape[0]

def poisson_loss(y_true, y_pred): 
    try:
        pl = torch.nn.PoissonNLLLoss(reduction='none', log_input=False, full=True)(torch.from_numpy(y_pred), torch.from_numpy(y_true).float()).sum(dim=-1)
    except:
        pl = torch.nn.PoissonNLLLoss(reduction='none', log_input=False, full=True)(torch.from_numpy(y_pred), torch.from_numpy(y_true).float()).sum(dim=-1)
    return pl.sum().numpy()/y_true.shape[0]

def evaluation_table(y_true, predictions, bce=True):
    results = {}
    for key, y_pred in predictions.items():
        ap = sklearn.metrics.average_precision_score(y_true.ravel(), y_pred.ravel(), pos_label=1)
        auroc = sklearn.metrics.roc_auc_score(y_true.ravel(), y_pred.ravel())
        rmse = sklearn.metrics.mean_squared_error(y_true.ravel(), y_pred.ravel(), squared=False)
        if bce:
            bce = bce_loss(y_true, y_pred)
        else:
            bce = None
        results.update({key: {'average_precision': ap, 'auroc': auroc, 'rmse': rmse, 'bce': bce}})
    results = pd.DataFrame(results)
    return results

def evaluate_embedding(adata, X_emb, labels_key, batch_key="batch", mode='basic'):
    adata.obsm['X_emb'] = X_emb
    if 'neighbors' in adata.uns.keys():
        adata.uns.pop('neighbors')
        
    if mode == 'basic':
        metrics = scib.metrics.metrics(adata, 
                                       adata, 
                                       batch_key=labels_key, 
                                       label_key=labels_key, 
                                       nmi_=True, 
                                       ari_=True, 
                                       embed="X_emb")
    elif mode == 'extended':
        adata_int = anndata.AnnData(X_emb, obs= adata.obs)    
        adata_int.obsm["X_emb"]=X_emb
        metrics = scib.metrics.metrics(adata, 
                     adata_int, 
                     batch_key=batch_key, 
                     label_key=labels_key, 
                     trajectory_=True, 
                     pseudotime_key="pseudotime_order_ATAC", 
                     nmi_=True, embed="X_emb", 
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
                     verbose=True)
    elif mode=="fast":
        adata_int = anndata.AnnData(X_emb, obs= adata.obs)    
        adata_int.obsm["X_emb"]=X_emb
        metrics = scib.metrics.metrics(adata, 
                     adata_int, 
                     batch_key=batch_key, 
                     label_key=labels_key, 
                     trajectory_=False, 
                     nmi_=True, embed="X_emb", 
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
                     verbose=True)
         
    return metrics

def evaluate_counts(model, adata, cell_idx=None, **kwargs):
    if cell_idx is None:
        print("Using test set of model")
        cell_idx = model.test_indices
        
    y_true = adata[cell_idx].layers["counts"].A
    
    y_pred = model.get_accessibility_estimates(adata, 
                                                indices=cell_idx, 
                                                return_numpy=True,
                                                **kwargs
                                            )
    r2 = sklearn.metrics.r2_score(y_true.ravel(), y_pred.ravel())
    pl = poisson_loss(y_true, y_pred)

    results= {'Model': {'r2_score': r2, 'poisson_loss': pl}}
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
    return results["result.model_path"].values[0]
    
def load_experiment(seml_collection, model_hash, get_experiment_fn):
    config = load_config(seml_collection, model_hash)
    ex = get_experiment_fn()
    ex.init_dataset(**config['data'])
    ex.init_model(**config['model'])
    ex.setup_adata(**config['setup'])
    
    model = ex.model.load(get_model_path(seml_collection, model_hash), adata=ex.adata)
    
    return ex, model, config

    