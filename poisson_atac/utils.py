
import seml

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

matplotlib.style.use("seaborn-colorblind")
sns.set_style("whitegrid")
matplotlib.style.use("seaborn-poster")
matplotlib.rcParams["font.size"] = 16
#matplotlib.rcParams["figure.dpi"] = 60
#matplotlib.pyplot.rcParams["savefig.facecolor"] = "white"
#sns.set_context("paper")

model_type_map = {
    'gexTrue':"Poisson encoder-decoder\n(observed library size)" , 
    'gexFalse':"Poisson encoder-decoder\n(mean observed library size)" , 
    'gex_binaryTrue':"Binary encoder-decoder\n(observed library size)" , 
    'gex_binaryFalse':"Binary encoder-decoder\n(mean observed library size)" , 
    'binaryviTrue': "Binary VAE (obs. ls)",
    'binaryviFalse': "Binary VAE (enc. ls)",
    'poissonviTrue': "Poisson VAE (obs. ls)",
    'poissonviFalse': "Poisson VAE (enc. ls)",
    'peakvi': "PeakVI",
    'LS_lab': 'Neurips winner: LS_lab\n(binary)'
}

    
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

def get_test_indices(seml_collection, model_hash):
    _, model, _ = load_experiment(seml_collection, model_hash)
    return model.test_indices
    
def load_experiment(seml_collection, model_hash, get_experiment_fn):
    config = load_config(seml_collection, model_hash)
    ex = get_experiment_fn()
    ex.init_dataset(**config['data'])
    ex.init_model(**config['model'])
    ex.setup_adata(**config['setup'])
    
    model = ex.model.load(get_model_path(seml_collection, model_hash), adata=ex.adata)
    
    return ex, model, config

import scanpy as sc
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
    
def plot_embedding(seml_collection, model_hash, fig_path, fig_name):
    
    sc.settings.figdir = fig_path
    ex, model, config = load_experiment(seml_collection, model_hash)
    X_emb = model.get_latent_representation(ex.adata)
    compute_embedding(ex.adata, X_emb)
    sc.pl.umap(ex.adata, color = config['setup']['label_key'], save=f"{fig_name}.pdf")
    sc.pl.umap(ex.adata, color = config['setup']['label_key'], save=f"{fig_name}.png")
    return ex.adata