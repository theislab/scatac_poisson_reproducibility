import seml
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from statannotations.Annotator import Annotator
from poisson_atac.utils import model_type_map
from poisson_atac.utils import dataset_map_simple as dataset_map
import scipy
import matplotlib as mpl
import sklearn
import scvelo as scv
import pyranges as pr

cell_types = [
    "HSC",
    "MK/E prog",
    "Proerythroblast",
    "Erythroblast",
    "Normoblast",
    "G/M prog",
    "CD14+ Mono",
    "CD16+ Mono",
    "ID2-hi myeloid prog",
    "cDC2",
    "pDC",
    "Lymph prog",
    "Transitional B",
    "Naive CD20+ B",
    "B1 B",
    "Plasma cell",
    "CD8+ T naive",
    "CD4+ T naive",
    "CD4+ T activated",
    "CD8+ T",
    "NK",
    "ILC",
]

cell_type_palette = {
    "B1 B": "#023fa5",
    "CD14+ Mono": "#7d87b9",
    "CD16+ Mono": "#bec1d4",
    "CD4+ T activated": "#d6bcc0",
    "CD4+ T naive": "#bb7784",
    "CD8+ T": "#8e063b",
    "CD8+ T naive": "#4a6fe3",
    "cDC2": "#8595e1",
    "Erythroblast": "#b5bbe3",
    "G/M prog": "#e6afb9",
    "HSC": "#e07b91",
    "ID2-hi myeloid prog": "#d33f6a",
    "ILC": "#11c638",
    "Lymph prog": "#8dd593",
    "MK/E prog": "#c6dec7",
    "Naive CD20+ B": "#ead3c6",
    "NK": "#f0b98d",
    "Normoblast": "#ef9708",
    "pDC": "#0fcfc0",
    "Plasma cell": "#9cded6",
    "Proerythroblast": "#d5eae7",
    "Transitional B": "#f3e1eb",
}


model_palette = {
"Poisson VAE": '#1f77b4',
"Binary VAE": '#ff7f0e',
"PeakVI": '#2ca02c',
"cisTopic": '#d62728',
"cisTopic (Harmony)": '#9467bd',
"Signac": '#8c564b',
"Signac (Harmony)": '#e377c2',
"SCALE": '#7f7f7f',
}

def load_seml(seml_database, convert_dataset_name=True):
    results = seml.get_results(
        seml_database,
        to_data_frame=True,
        fields=["config", "config_hash", "result", "batch_id"],
    )

    if convert_dataset_name:
        results["config.data.dataset"] = results["config.data.dataset"].map(dataset_map)
    results["config.model.model_type"] = results["config.model.model_type"].map(
        model_type_map
    )
    results["config.model.model_type"] = pd.Categorical(
        results["config.model.model_type"], categories=model_type_map.values()
    )

    return results


def plot_metrics_per_dataset(
    data,
    datasets,
    x,
    y,
    hue,
    hue_order,
    y_label=None,
    x_label=None,
    save_path=None,
    figsize=(10, 4),
    test="t-test_paired",
    order=None,
    ax=None,
    **kwargs,
):
    pairs_prelim = [(model, model) for model in hue_order]
    pairs = list(itertools.combinations(pairs_prelim, 2))
    print(pairs)
    pairs=[((model_type_map['poissonvi'], model_type_map['poissonvi']), 
        (model_type_map['binaryvi'], model_type_map['binaryvi'])),
       #
        ((model_type_map['binaryvi'], model_type_map['binaryvi']), 
        (model_type_map['peakvi'], model_type_map['peakvi'])),
      ]
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax = create_annotated_stripplot(
        data=data,
        y=y,
        x=x,
        hue=hue,
        order=order,
        hue_order=hue_order,
        ax=ax,
        y_label=y_label,
        x_label=x_label,
        test=test,
        pairs=pairs,
        **kwargs,
    )

    sns.despine(left=False)
    if save_path:
        fig.savefig(
            os.path.join(save_path, f'{y.split(".")[-1]}_{"_".join(hue_order)}.pdf')
        )
        fig.savefig(
            os.path.join(save_path, f'{y.split(".")[-1]}_{"_".join(hue_order)}.png')
        )
    return ax


def create_annotated_stripplot(
    data,
    x,
    y,
    hue,
    hue_order,
    order,
    ax,
    x_label=None,
    y_label=None,
    legend=True,
    y_lim=None,
    pvalue_format="star",
    pairs=None,
    test=None,
    **kwargs,
):
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.lines import Line2D

    markers = dict(
        zip(
            data["config.scvi.seed"].unique(),
            Line2D.filled_markers[: len(data["config.scvi.seed"].unique())],
        )
    )

    for run, group in data.groupby("config.scvi.seed"):
        ax = sns.stripplot(
            data=group,
            x=x,
            y=y,
            hue=hue,
            hue_order=hue_order,
            order=order,
            ax=ax,
            marker=markers[run],
            **kwargs,
        )
    if test:
        annot = Annotator(
            ax,
            pairs,
            data=data,
            x=x,
            y=y,
            order=order,
            hue=hue,
            hue_order=hue_order,
            **kwargs,
        )
        annot.configure(
            test=test,
            text_format=pvalue_format,
            loc="outside",
            comparisons_correction="Benjamini-Hochberg",
            verbose=2,
        )
        annot.apply_test()
        ax, test_results = annot.annotate()
    median_width = 0.4

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        name = text.get_text()  # "X" or "Y"
        # calculate the median value for all replicates of either X or Y
        median_val = data.loc[data[x] == name, y].median()

        # plot horizontal lines across the column, centered on the tick
        ax.plot(
            [tick - median_width / 2, tick + median_width / 2],
            [median_val, median_val],
            lw=2,
            color="lightgray",
        )

    if x_label:
        ax.set_xlabel(x_label)
        ax.axes.get_xaxis().set_ticklabels([])
    else:
        ax.set_xlabel("")

    if y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")
        ax.axes.get_yaxis().set_ticklabels([])

    if y_lim:
        ylim_bot = (data[y].min() * 1000 // 5) * 0.005
        ylim_up = ylim_bot + 0.03
        ax.set_ylim((ylim_bot, ylim_up))

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = [
            Line2D([], [], color="k", linestyle="", marker=item)
            for key, item in markers.items()
        ]
        labels, handles = zip(*sorted(zip(markers.keys(), handles)))
        ax.legend(
            handles,
            labels,
            title="CV Fold",
            bbox_to_anchor=(1.01, 1.01),
            loc="upper left",
            fontsize=10,
            title_fontsize=10,
            frameon=False,
            markerscale=0.8,
        )
    else:
        ax.axes.get_legend().set_visible(False)
    ax.axes.get_yaxis().set_major_formatter(FormatStrFormatter("%.3f"))
    ax.axes.tick_params(axis="y", which="major", left=True)
    plt.tight_layout()
    return ax


def plot_density(
    adata_subset, key, expression, gene, xlabel="", ylabel="", title="", save=None
):
    from utils import cell_type_palette

    with mpl.rc_context(
        {
            "xtick.major.pad": 0,
            "xtick.minor.pad": 0,
            "ytick.major.pad": 0,
            "ytick.minor.pad": 0,
        }
    ):
        sns.set_style("whitegrid")
        scv.set_figure_params()

        # define plotting df
        sns_df = pd.DataFrame(
            {
                "Cell type": adata_subset.obs.cell_type,
                "Normalized accessibility": adata_subset.obs[key],
                "Expression": expression,
            }
        )

        # plot kde for only positive genes
        ax = sns.JointGrid(
            data=sns_df.loc[(sns_df.Expression > 0)],
            x="Normalized accessibility",
            y="Expression",
            hue="Cell type",
            height=5,
            palette=cell_type_palette,
            ratio=3,
        )

        # add zeros
        def custom_joint(x, y, data_new, hue, palette):
            sns.scatterplot(
                x=data_new["Normalized accessibility"],
                y=data_new["Expression"],
                hue=data_new["Cell type"],
                palette=palette,
                alpha=0.5,
            )

        # compute roc functions
        def compute_roc(sns_df):
            roc = sklearn.metrics.roc_auc_score(
                y_score=sns_df["Normalized accessibility"].values,
                y_true=sns_df["Cell type"].values,
                labels=sns_df["Cell type"].unique(),
            )

            if roc < 0.5:
                roc = 1 - roc
            return roc

        def compute_mean_roc(sns_df):
            from itertools import combinations

            combos = list(combinations(sns_df["Cell type"].values.unique(), 2))
            roc = np.mean(
                [
                    compute_roc(sns_df.loc[sns_df["Cell type"].isin(combo)])
                    for combo in combos
                ]
            )
            return roc

        ax.plot_joint(sns.kdeplot, common_norm=False, thresh=0.1)
        ax.plot_joint(
            custom_joint,
            data_new=sns_df.loc[(sns_df.Expression == 0)],
            palette=cell_type_palette,
        )

        # plot marginal for all data
        sns.kdeplot(
            data=sns_df,
            x="Normalized accessibility",
            hue="Cell type",
            palette=cell_type_palette,
            ax=ax.ax_marg_x,
            fill=True,
            common_norm=False,
        )

        # compute metrics
        silh = sklearn.metrics.silhouette_score(
            X=sns_df["Normalized accessibility"].values.reshape(-1, 1),
            labels=sns_df["Cell type"].values,
        )
        roc = compute_mean_roc(sns_df)

        ax.ax_joint.text(
            0.4,
            0.95,
            f"Silhouette: " + "{:.3f}".format(silh),
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.ax_joint.transAxes,
            fontsize=9,
        )
        ax.ax_joint.text(
            0.4,
            0.90,
            f"ROC AUC: " + "{:.3f}".format(roc),
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.ax_joint.transAxes,
            fontsize=9,
        )

        # set labels
        ax.ax_joint.set(xlabel=xlabel, ylabel=ylabel)

        # remove legend
        ax.ax_joint.legend().remove()
        ax.ax_marg_x.legend().remove()

        # remove y marginal
        ax.ax_marg_y.remove()

        # Add title
        ax.ax_marg_x.set(title=title)

        if save:
            # plt.tight_layout()
            plt.savefig(
                os.path.join(save, f"jointplot_{title}_{gene}.pdf"), bbox_inches="tight"
            )
            plt.savefig(
                os.path.join(save, f"jointplot_{title}_{gene}.png"), bbox_inches="tight"
            )
    return


def plot_regions(
    region, correlation, adata, adata_gex, acc, label, cell_types, save_path=None
):
    gene = correlation.loc[region, "gene"]
    print(gene)
    adata.obs["normalized_acc"] = acc[region]

    expr = adata_gex[adata_gex.obs.cell_type.isin(cell_types), gene].X.A.squeeze()

    plot_density(
        adata[adata.obs.cell_type.isin(cell_types)],
        "normalized_acc",
        expression=expr,
        gene=gene,
        xlabel=f"Promoter normalized accessibility",
        ylabel=r"Expression $\log(\mathrm{norm}(x) + 1)$",
        title=f"{gene}, {label}",
        save=save_path,
    )

    return


def contigency_table(grs, peak_set, M, n, anno="Super enhancers"):
    N = peak_set.shape[0]
    x = (
        (pr.count_overlaps(grs, pr.PyRanges(peak_set)).df.loc[:, list(grs.keys())] > 0)
        * 1
    )[
        anno
    ].sum()  # overlap.loc[peak_set.index, anno].sum()
    return np.array([[x, n - x], [N - x, M - (n + N) + x]])


def plot_correlation(test_regions, adata, adata_gex, gene, fig_path=None):
    quant = adata.obs.size_factor.quantile([0.25, 0.75])
    region = test_regions.loc[test_regions.gene.isin([gene])].index[0]
    cell_restriction = adata.obs.size_factor.between(quant[0.25], quant[0.75])

    sns_df = pd.DataFrame(
        {
            "acc": adata[cell_restriction, region].layers["counts"].A.squeeze(),
            "expr": adata_gex[cell_restriction, gene].layers["counts"].A.squeeze(),
            "cell_type": adata[cell_restriction].obs.cell_type.values,
        }
    )
    sns_df = sns_df[sns_df["expr"] <= 50]  # filter one outlier
    sns_df["acc"] = sns_df["acc"].astype(int)
    sns_df["acc"] = pd.Categorical(sns_df["acc"], ordered=True)
    
    n = sns_df[sns_df['acc'] > 0].shape[0]
    print(f"Gene: {gene}, n={n}")

    with sns.color_palette("Blues"):
        # plot data
        fig, ax = plt.subplots(figsize=(3, 6))
        ax = sns.boxplot(data=sns_df, x="acc", y="expr", fliersize=0, palette="Blues")
        ax = sns.stripplot(
            data=sns_df, x="acc", y="expr", jitter=True, color="black", size=2
        )

        # annotate
        corr = np.round(test_regions.loc[region, "correlation"], 3)
        pval = test_regions.loc[region, "pvalue"]
        ax.text(
            0.96,
            0.95,
            f"Spearman: R={corr}, " + "p={:0.1e}".format(pval),
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        plt.xlabel(f"Fragment count in promoter")
        plt.ylabel(f"Expression count")
        plt.title(gene)
        sns.despine(left=False)

        # save plot
        if fig_path:
            fig.savefig(
                os.path.join(fig_path, f"{gene}_correlation.pdf"), bbox_inches="tight"
            )
            fig.savefig(
                os.path.join(fig_path, f"{gene}_correlation.png"), bbox_inches="tight"
            )


def plot_integration_results(df, show=True, save_path=None):
    """Plot the benchmarking results.
    
    Adapted from https://github.com/yoseflab/scib-metrics

    Parameters
    ----------
    df
        Metric to plot.
    show
        Whether to show the plot.
    save_path
        The path to save the plot to. If `None`, the plot is not saved.
    """

    import numpy as np
    from plottable import ColumnDefinition, Table
    from plottable.cmap import normed_cmap
    from plottable.plots import bar
    import matplotlib

    _METRIC_TYPE = "Metric Type"
    _AGGREGATE_SCORE = "Aggregate score"

    metric_type_map = {
        "NMI_cluster/label": "Bio conservation",
        "ARI_cluster/label": "Bio conservation",
        "ASW_label": "Bio conservation",
        "ASW_label/batch": "Batch correction",
        "PCR_batch": "Batch correction",
        "isolated_label_F1": "Bio conservation",
        "isolated_label_silhouette": "Bio conservation",
        "graph_conn": "Batch correction",
        "iLISI": "Batch correction",
        "cLISI": "Bio conservation",
        "trajectory":"Bio conservation",
        "Bio conservation": _AGGREGATE_SCORE,
        "Batch correction": _AGGREGATE_SCORE,
        "Total": _AGGREGATE_SCORE,
    }

    metric_clean_map = {
        "NMI_cluster/label": "NMI",
        "ARI_cluster/label": "ARI",
        "ASW_label": "Silhouette\nlabel",
        "ASW_label/batch": "Silhouette\nbatch",
        "PCR_batch": "PCR\ncomparison",
        "isolated_label_F1": "Isolated label\nF1",
        "isolated_label_silhouette": "Isolated label\nsilhouette",
        "graph_conn": "Graph\nconnectivity",
        "iLISI": "iLISI",
        "cLISI": "cLISI",
        'trajectory': 'Trajectory conservation',
        "Bio conservation": "Bio conservation",
        "Batch correction": "Batch correction",
        "Total": "Total",
    }

    df.loc[_METRIC_TYPE, :] = df.columns.map(metric_type_map)
    df = df.sort_values("Metric Type", axis=1, ascending=False)

    df.columns = df.columns.map(metric_clean_map)

    num_embeds = len(df) - 1
    cmap_fn = lambda col_data: normed_cmap(
        col_data, cmap=matplotlib.cm.PRGn, num_stds=2.5
    )

    # Do not want to plot what kind of metric it is
    plot_df = df.drop(_METRIC_TYPE, axis=0)
    # Sort by total score
    plot_df = plot_df.sort_values(by="Total", ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    # Split columns by metric type, using df as it doesn't have the new method col
    score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
    other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]

    column_definitions = [
        ColumnDefinition(
            "Method", width=1.5, textprops={"ha": "left", "weight": "bold"}
        ),
    ]
    # Circles for the metric values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=cmap_fn(plot_df[col]),
            group=df.loc[_METRIC_TYPE, col],
            formatter="{:.3f}",
        )
        for i, col in enumerate(other_cols)
    ]

    # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": matplotlib.cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.3f}",
            },
            group=df.loc[_METRIC_TYPE, col],
            border="left" if i == 0 else None,
        )
        for i, col in enumerate(score_cols)
    ]

    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.5, 3 + 0.5 * num_embeds))
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)

    if show:
        plt.show()
    if save_path is not None:
        fig.savefig(
            save_path,
            facecolor=ax.get_facecolor(),
            dpi=300,
        )

    return tab
