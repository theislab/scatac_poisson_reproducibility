
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

from matplotlib import pyplot as plt
import seaborn as sns
import os

def proportions(
    adata: AnnData, 
    upper_limit: int,
    lower_limit: int,
    layer: Optional[str] = None, 
    figsize: Optional[Tuple] = None,
    fontsize: Optional[int] = 14,
    label='read(s)',
    save_prefix: Optional[str] = "", 
    save_path: Optional[str] = None,
    ):
    
    if layer:
        data = adata.layers[layer].data
    else:
        data = adata.X.data
    
    # Adapted from scvelo https://github.com/theislab/scvelo/blob/master/scvelo/plotting/proportions.py
    non_zero_counts = pd.Series(data).value_counts().to_frame().reset_index().rename({0: 'count', 'index':'bin'}, axis = 1)
    all_counts = pd.concat([non_zero_counts, pd.DataFrame({'bin': 0.0 , 'count':adata.shape[0]*adata.shape[1]-len(data) }, index=[0])])
    all_counts["bin"] = all_counts["bin"].astype(int)
    
    df = pd.concat([all_counts[(all_counts["bin"] <= upper_limit) & (all_counts["bin"] >= lower_limit)], pd.DataFrame({'bin': f">{upper_limit}", "count": all_counts["count"][all_counts["bin"]>upper_limit].sum()}, index =[0])])
    df["bin"] = df["bin"].astype(str)
    df = df.sort_values("bin")
    
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    colors = sns.color_palette("Blues", df.shape[0]+1, as_cmap=False)[1:]
    autopct = "%1.1f%%" 
    explode=None
    
    pie = ax.pie(
        df["count"],
        colors=colors,
        explode=explode,
        autopct=autopct,
        shadow=False,
        startangle=45,
        textprops={'size': 'medium'}, 
        labels=list(df["bin"].astype(str) + f" {label}")
    )
    if autopct is not None:
        for pct, color in zip(pie[-1], colors):
            pct.set_color("black")
            pct.set_fontsize(fontsize)

    plt.tight_layout()
    
    if save_path:
        fig.savefig(os.path.join(save_path, f"{save_prefix}_proportions.png"), bbox_inches='tight')
        fig.savefig(os.path.join(save_path, f"{save_prefix}_proportions.pdf"), bbox_inches='tight')