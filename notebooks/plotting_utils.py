
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def create_annotated_boxplot(data, pairs, x, y, hue, hue_order, order, test, ax, x_label=None, y_label=None, legend=True, y_lim=None):
    sns.boxplot(data = data, x=x, y=y, hue=hue, hue_order = hue_order, order=order, orient="v", ax=ax)
    annot = Annotator(ax, pairs, data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annot.configure(test=test, verbose=0, text_format='star', loc='outside')
    annot.apply_test()
    ax, test_results = annot.annotate()
    
        
    if x_label:
        plt.xlabel(x_label)
    else:
        plt.xlabel("")
        #plt.xlabel(x.split(".")[-1])
        
    if y_label:
        plt.ylabel(y_label)
    else:
        plt.ylabel("")
        ax.axes.get_yaxis().set_ticklabels([])
        #ax.axes.get_yaxis().set_visible(False)
        #plt.ylabel(y.split(".")[-1])
        
    if y_lim:
        plt.ylim(y_lim)
        
    if legend:
        plt.legend()
    else:
        ax.axes.get_legend().set_visible(False)
        
    plt.tight_layout()
    return ax


def create_annot_boxplot_neurips(data, x, y, hue, hue_order, order_all, order_batch, test='t-test_paired', x_label=None, y_label=None, save=None):
    pairs_all = [tuple(itertools.product([order_i], model_order)) for order_i in order_all]
    pairs_batch = [tuple(itertools.product([order_i], model_order)) for order_i in order_batch]
    
    diff = data[y].max() - data[y].min()
    y_lim = (data[y].min()-0.01*diff, data[y].max()+0.15*diff)
    print(y_lim)
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1 = create_annotated_boxplot(data, pairs_batch, x, y, hue, hue_order, order_batch, test, ax1, x_label="Neurips batch", y_label=None, y_lim=y_lim)
    fig2, ax2 = plt.subplots(figsize=(4,5))
    ax2 = create_annotated_boxplot(data, pairs_all, x, y, hue, hue_order, order_all, test, ax2, x_label="Dataset", y_label=y_label, legend=False, y_lim=y_lim)
    
    
    if save:
        fig1.savefig(os.path.join(save, f'{y.split(".")[-1]}_{x.split(".")[-1]}_{"_".join(hue_order)}_{test}_1.pdf'))
        fig1.savefig(os.path.join(save, f'{y.split(".")[-1]}_{x.split(".")[-1]}_{"_".join(hue_order)}_{test}_1.png'))
        fig2.savefig(os.path.join(save, f'{y.split(".")[-1]}_{x.split(".")[-1]}_{"_".join(hue_order)}_{test}_2.pdf'))
        fig2.savefig(os.path.join(save, f'{y.split(".")[-1]}_{x.split(".")[-1]}_{"_".join(hue_order)}_{test}_2.png'))
    return 


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

def mean_var_scatter(df, label=r"$\sigma^2=\mu$", endpoint=(1,1), linestyle="solid", xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(6,6))
    g=sns.scatterplot(data=df, x="mean", y ="variance")
    g.set_yscale('log')
    g.set_xscale('log')
    
    if isinstance(label, list):
        for i, l in enumerate(label):
            g.axline((0,0), endpoint[i], color='darkgrey', linestyle=linestyle[i], label=l)
    else:
        g.axline((0,0), endpoint, color='darkgrey', linestyle=linestyle, label=label)
    g.grid(":", color="lightgrey")
    plt.xlabel(xlabel)#r"mean $\mu$"
    plt.ylabel(ylabel)#r"variance $\sigma^2$"
    plt.legend()
    plt.tight_layout()

def plot_mean_variance(adata, save_path=None, prefix=""):

    # number of counts
    X = np.ceil(adata.layers["counts"].A/2)
    mean_counts = X.mean(axis = 0)
    var_counts = X.var(axis = 0)
    df = pd.DataFrame({'mean': mean_counts, 'variance': var_counts})
    
    mean_var_scatter(df, label="Poisson limit", endpoint=(1,1), xlabel=r"Fragment mean $\mu$", ylabel=r"Fragment variance $\sigma^2$")
    if save_path:
        plt.gcf().savefig(os.path.join(save_path, f"{prefix}_raw_data_scatter.png"))
        plt.gcf().savefig(os.path.join(save_path, f"{prefix}_raw_data_scatter.pdf"))
    
    X = adata.layers["counts"].A
    mean_counts = X.mean(axis = 0)
    var_counts = X.var(axis = 0)
    df = pd.DataFrame({'mean': mean_counts, 'variance': var_counts})
    
    mean_var_scatter(df, label=["Poisson limit", r"$\sigma^2=2\mu$"], endpoint=[(1,1),(1,2)], linestyle=["solid", ":"], xlabel=r"Read mean $\mu$", ylabel=r"Read variance $\sigma^2$")
    if save_path:
        plt.gcf().savefig(os.path.join(save_path, f"{prefix}_transformed_data_scatter.png"))
        plt.gcf().savefig(os.path.join(save_path, f"{prefix}_transformed_data_scatter.pdf"))

def plot_dataset(adata, save_path=None, prefix=None, limit=None):
    counts = pd.Series(adata.layers["counts"].data).value_counts().to_frame().reset_index().rename({0: 'count', 'index':'reads'}, axis = 1)
    sns_df = pd.concat([counts, pd.DataFrame({'reads':0.0 , 'count':adata.shape[0]*adata.shape[1]-len(adata.layers["counts"].data) }, index=[0])])
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

def plot_proportions(adata, figsize, fontsize, save_path=None, prefix="", lower_limit=0, upper_limit=2):
    
    # Adapted from scvelo https://github.com/theislab/scvelo/blob/master/scvelo/plotting/proportions.py
    counts = pd.Series(adata.layers["counts"].data).value_counts().to_frame().reset_index().rename({0: 'count', 'index':'reads'}, axis = 1)
    counts = pd.concat([counts, pd.DataFrame({'reads': 0.0 , 'count':adata.shape[0]*adata.shape[1]-len(adata.layers["counts"].data) }, index=[0])])
    counts["reads"] = counts["reads"].astype(int)
    
    sns_df = pd.concat([counts[(counts["reads"] <= upper_limit) & (counts["reads"] >= lower_limit)], pd.DataFrame({'reads': f">{upper_limit}", "count": counts["count"][counts["reads"]>upper_limit].sum()}, index =[0])])
    sns_df["reads"] = sns_df["reads"].astype(str)
    sns_df = sns_df.sort_values("reads")
    
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    colors = sns.color_palette("Blues", sns_df.shape[0]+1, as_cmap=False)[1:]
    autopct = "%1.1f%%" 
    explode=None
    pie = ax.pie(
        sns_df["count"],
        colors=colors,
        explode=explode,
        autopct=autopct,
        shadow=False,
        startangle=45,
        textprops={'size': 'medium'}, 
        labels=list(sns_df["reads"].astype(str) + " read(s)")
    )
    if autopct is not None:
        for pct, color in zip(pie[-1], colors):
            r, g, b = color
            pct.set_color("black")
            pct.set_fontsize(fontsize)
    # ax.legend(
    #     sns_df["reads"].astype(str) + " read(s)",
    #     ncol=len(sns_df["reads"]),
    #     bbox_to_anchor=(0, 1),
    #     loc="lower left",
    #     fontsize=fontsize,
    # )
    plt.tight_layout()
    if save_path:
        fig.savefig(os.path.join(save_path, f"{prefix}_proportions.png"))
        fig.savefig(os.path.join(save_path, f"{prefix}_proportions.pdf"))