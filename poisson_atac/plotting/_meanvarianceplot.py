
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from matplotlib import pyplot as plt
import seaborn as sns
import os


def mean_var_scatter(df: pd.DataFrame, 
                     label:Optional[Union[str, list]] = None, 
                     endpoint: Optional[Tuple] =(1,1), 
                     linestyle: Optional[str] = "solid", 
                     xlabel: Optional[str] = r"mean $\mu$", 
                     ylabel: Optional[str] = r"variance $\sigma^2$",
                     figsize: Optional[Tuple] = (6,6),
                     prefix: Optional[str] = "", 
                     save_path: Optional[str] = None
                     ):
    
    fig, ax = plt.subplots(figsize=figsize)
    g=sns.scatterplot(data=df, x="mean", y ="variance", ax=ax)
    g.set_yscale('log')
    g.set_xscale('log')
    
    if isinstance(label, list):
        for i, l in enumerate(label):
            g.axline((0,0), endpoint[i], color='darkgrey', linestyle=linestyle[i], label=l)
    else:
        g.axline((0,0), endpoint, color='darkgrey', linestyle=linestyle, label=label)
        
    g.grid(":", color="lightgrey")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, f"{prefix}_mean_var_scatter.pdf"))
        plt.savefig(os.path.join(save_path, f"{prefix}_mean_var_scatter.png"))

def compute_mean_variance(X):
    mean_counts = X.mean(axis = 0)
    var_counts = X.var(axis = 0)
    df = pd.DataFrame({'mean': mean_counts, 'variance': var_counts})
    return df

def mean_variance(
    adata: AnnData, 
    layer: Optional[str] = None, 
    figsize: Optional[Tuple] = None,
    save_prefix: Optional[str] = "", 
    save_path: Optional[str] = None,
    ):
    
    if layer:
        X = adata.layers[layer].A
    else:
        X = adata.X.A
    df = compute_mean_variance(X)
    
    mean_var_scatter(df, 
                     label=["Poisson limit", r"$\sigma^2=2\mu$"],
                     endpoint=[(1,1),(1,2)], 
                     linestyle=["solid", ":"],
                     xlabel=r"Read mean $\mu$", 
                     ylabel=r"Read variance $\sigma^2$",
                     prefix=f"{save_prefix}_reads",
                     save_path=save_path,
                     figsize=figsize
                     )

    X = np.ceil(X/2)
    df = compute_mean_variance(X)
    
    mean_var_scatter(df, 
                    label="Poisson limit", 
                    endpoint=(1,1), 
                    xlabel=r"Fragment mean $\mu$", 
                    ylabel=r"Fragment variance $\sigma^2$",
                    prefix=f"{save_prefix}_fragments",
                    save_path=save_path,
                    figsize=figsize
                    )
