
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

from matplotlib import pyplot as plt
import seaborn as sns
import os


def make_plot(all_counts, xlabel, figsize, prefix="", save_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    all_counts["color"] = np.concatenate([np.array(['0', '1', '2']), np.repeat('>2', all_counts.shape[0]-3)])
    g = sns.barplot(data=all_counts, x = 'bin', y = 'count', hue="color", palette="Blues", dodge=False, ax=ax)
    g.set_yscale('log') # log scale so we can see the change
    

    labels = g.get_xticklabels() # only plot every second label
    labels[1:len(all_counts):2] = list(np.repeat('', len(labels[1:len(all_counts):2])))
    if np.any(~all_counts["in_limit"]):
        limit = int(all_counts.loc[~all_counts["in_limit"], "bin"].values)
        labels = labels[:-1]
        labels += [f">{limit-1}"]
    g.set_xticklabels(labels, rotation=90, va="top", fontsize = 12)
    g.legend().set_visible(False)
    g.grid(False)
    plt.xlabel(xlabel)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{prefix}_{xlabel}_count_distribution.pdf"))
        plt.savefig(os.path.join(save_path, f"{prefix}_{xlabel}_count_distribution.pdf"))

def counts_per_bin(data, adata, limit):
    #count non-zero occurences   
    non_zero_counts = pd.Series(data).value_counts().to_frame().reset_index().rename({0: 'count', 'index':'bin'}, axis = 1)
    
    #Add zeros
    all_counts = pd.concat([non_zero_counts, pd.DataFrame({'bin':0 , 'count':adata.shape[0]*adata.shape[1]-len(data) }, index=[0])])
    
    # convert reads to category
    all_counts['bin'] = pd.Categorical(all_counts['bin'].astype(int), categories=np.arange(all_counts['bin'].max() + 1).astype(int), ordered=True)
    
    if limit:
        all_counts = pd.concat([all_counts[all_counts['bin'] <= limit], pd.DataFrame({'bin': int(limit) + 1, "count": all_counts["count"][all_counts["bin"]>limit].sum()}, index =[0])])
        all_counts['in_limit'] = all_counts['bin'] <= limit
    else:
        all_counts['in_limit'] = True
    return all_counts.sort_values('bin')

def count_distribution(
    adata: AnnData, 
    layer: Optional[str] = None, 
    limit: Optional[int] = None,
    figsize: Optional[Tuple] = None,
    save_prefix: Optional[str] = "", 
    save_path: Optional[str] = None,
    ):
    
    if layer:
        data = adata.layers[layer].data
    else:
        data = adata.X.data
    
    all_counts = counts_per_bin(data, adata, limit)
    make_plot(all_counts, xlabel="Reads", figsize=figsize, prefix=save_prefix, save_path=save_path)
        
    ## Convert counts
    data = np.ceil(data/2)
    
    all_counts = counts_per_bin(data, adata, (limit//2 if limit is not None else None))
    make_plot(all_counts, xlabel="Fragments", figsize=figsize, prefix=save_prefix, save_path=save_path)
    