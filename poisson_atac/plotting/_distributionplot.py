from abc import ABC
from typing import Optional, Union, Mapping  # Special
from typing import Sequence  # ABCs
from typing import Tuple  # Classes

import numpy as np
import pandas as pd
from anndata import AnnData

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib import rcParams, ticker, gridspec, axes
import seaborn as sns


class _AxesSubplot(Axes, axes.SubplotBase, ABC):
    """Intersection between Axes and SubplotBase: Has methods of both"""
    
    
def plot_df(df, figsize=(15, 8), xlabel="Number of reads"):
    fig, ax = plt.subplots(figsize=figsize)
    
    #numeric labels
    num_labels=df["reads"][~df["reads"].str.contains(">")].astype(int).max()
    order = list(np.arange(num_labels + 1).astype(str))
    order = order + list(df["reads"][df["reads"].str.contains(">")])
    
    df.reads = df.reads.astype("category")
    df.reads.cat.set_categories(order, inplace=True)
    df = df.sort_values("reads")
    df["color"] = np.concatenate([np.array(['0', '1', '2']), np.repeat('3', df.shape[0]-3)])
    print(df.head())
    g = sns.barplot(data=df, x = 'reads', y = 'count', hue="color", palette="Blues", ax = ax, dodge=False, order=order)
    g.set_yscale('log')
    labels = g.get_xticklabels()
    labels[1:-1:2] = np.repeat('', len(labels[1:-1:2]))
    g.set_xticklabels(labels, rotation=90, va="top", fontsize = 12);
    g.legend().set_visible(False)
    g.grid(False)
    plt.xlabel(xlabel)
    plt.tight_layout()




def count_distribution(
    adata: AnnData, 
    layer: Optional[str] = None, 
    limit: Optional[int] = None,
    ax: Optional[_AxesSubplot] = None,
    save_prefix: Optional[str] = "count_distribution", 
    save_path: Optional[str] = '.',
    ):
    
    if layer:
        data = adata.layers[layer].data
    else:
        data = adata.X.data
    
    #count non-zero occurences   
    counts = pd.Series(data).value_counts().to_frame().reset_index().rename({0: 'count', 'index':'reads'}, axis = 1)
    
    #Add zeros
    sns_df = pd.concat([counts, pd.DataFrame({'reads':0 , 'count':adata.shape[0]*adata.shape[1]-len(data) }, index=[0])])
    
    # convert reads to category
    sns_df['reads'] = pd.Categorical(sns_df['reads'].astype(int), categories=np.arange(sns_df['reads'].max() + 1).astype(int), ordered=True)

    if limit:
        sns_df = pd.concat([sns_df[(sns_df["reads"] <= limit)], pd.DataFrame({'reads': f">{int(limit)}", "count": sns_df["count"][sns_df["reads"]>limit].sum()}, index =[0])])
        
    sns_df["reads"] = sns_df["reads"].astype(str)
    plot_df(sns_df, figsize = (12,4.8), xlabel="Reads")
    if save_path:
        plt.gcf().savefig(os.path.join(save_path, f"{prefix}_raw_data_dist_zoom.pdf"))
        plt.gcf().savefig(os.path.join(save_path, f"{prefix}_raw_data_dist_zoom.png"))
        
    ## Convert counts
    data = np.ceil(adata.layers["counts"].data/2)

    counts = pd.Series(data).value_counts().to_frame().reset_index().rename({0: 'count', 'index':'reads'}, axis = 1)
    sns_df = pd.concat([counts, pd.DataFrame({'reads':0.0 , 'count':adata.shape[0]*adata.shape[1]-len(adata.layers["counts"].data) }, index=[0])])
    sns_df['reads'] = pd.Categorical(sns_df['reads'].astype(int), categories=np.arange(sns_df['reads'].max() + 1).astype(int), ordered=True)

    if limit:
        sns_df = pd.concat([sns_df[(sns_df["reads"] <= int(limit/2))], pd.DataFrame({'reads': f">{int(limit/2)}", "count": sns_df["count"][sns_df["reads"]>int(limit/2)].sum()}, index =[0])])
        
    sns_df["reads"] = sns_df["reads"].astype(str)
    plot_df(sns_df, figsize = (12,4.8), xlabel="Fragments")
    if save_path:
        plt.gcf().savefig(os.path.join(save_path, f"{prefix}_transformed_data_dist_zoom.pdf"))
        plt.gcf().savefig(os.path.join(save_path, f"{prefix}_transformed_data_dist_zoom.png"))