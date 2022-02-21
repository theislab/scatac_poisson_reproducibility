import pandas as pd
import sklearn.metrics
import scib
import torch


def evaluate_test_cells(model, adata, **kwargs):
    """
    Evaluate performance on test peaks compared to scbasset and baseline
    ----------
    model_dir
        Path to saved trained model
    model_dir2
        Path to peakvi model.
    adata
        adata file with chrom_idx for test_chrom
    """


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

    
def bce_loss(y_true, y_pred): 
    bce = torch.nn.BCELoss(reduction="none")(torch.from_numpy(y_pred), torch.from_numpy(y_true).float()).sum(dim=-1)
    return bce.sum().numpy()/y_true.shape[0]

def evaluation_table(y_true, predictions):
    results = {}
    for key, y_pred in predictions.items():
        ap = sklearn.metrics.average_precision_score(y_true.ravel(), y_pred.ravel(), pos_label=1)
        auroc = sklearn.metrics.roc_auc_score(y_true.ravel(), y_pred.ravel())
        rmse = sklearn.metrics.mean_squared_error(y_true.ravel(), y_pred.ravel(), squared=False)
        bce = bce_loss(y_true, y_pred)
        results.update({key: {'average_precision': ap, 'auroc': auroc, 'rmse': rmse, 'bce': bce}})
    results = pd.DataFrame(results)
    return results

def evaluate_embedding(adata, X_emb, labels_key):
    adata.obsm['X_emb'] = X_emb
    if 'neighbors' in adata.uns.keys():
        adata.uns.pop('neighbors')
    metrics = scib.metrics.metrics(adata, adata, batch_key=labels_key, label_key=labels_key, nmi_=True, ari_=True, embed="X_emb")
    return metrics