from sacred import Experiment
import seml

import os
import poisson_atac as patac
import pickle
from itertools import chain


from pycisTopic.cistopic_class import run_cgs_models_mallet, create_cistopic_object, evaluate_models
from pycisTopic.clust_vis import harmony

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @ex.capture(prefix="data")
    def init_dataset(self, dataset, replacement):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """      
        self.dataset = dataset
        if dataset == "neurips":
            adata = patac.data.load_neurips(batch=None, only_train=False)
        elif dataset == "satpathy":
            adata = patac.data.load_hematopoiesis()
        elif dataset == "aerts":
            adata = patac.data.load_aerts()
        elif dataset == "trapnell":
            adata = patac.data.load_trapnell()
        
        print(replacement)
        region_names=adata.var_names 
        for repl in replacement:
            region_names = region_names.str.replace(repl[0], repl[1], n=1)

        self.cistopic_obj = create_cistopic_object(adata.X.tocsc().T, 
                                            cell_names=adata.obs_names, 
                                            region_names=region_names, 
                                            min_cell=0, 
                                            min_frag=0)

        metadata = adata.obs
        metadata.index = metadata.index + '___cisTopic'

        self.cistopic_obj.cell_data = self.cistopic_obj.cell_data.join(metadata)

    @ex.capture(prefix="setup")
    def setup_adata(self, batch_key):
        self.batch_key = batch_key

        
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.setup_adata()
    
    @ex.capture(prefix="training")
    def run(self, path_to_mallet_binary, n_topics, save_path):

        #path_to_mallet_binary='/lustre/groups/ml01/code/laura.martens/github_repos/pycisTopic/Mallet-202108/bin/mallet'
        # Run models
        n_topics_new = [ntopic for ntopic in n_topics if not os.path.exists(os.path.join(save_path, 'Topic' + str(ntopic) + '.pkl'))]
        os.environ['MALLET_MEMORY'] = '250G'

        if len(n_topics_new) > 0:
            models=run_cgs_models_mallet(path_to_mallet_binary,
                                self.cistopic_obj,
                                n_topics=n_topics_new,
                                n_cpu=10,
                                n_iter=500,
                                random_state=555,
                                alpha=50,
                                alpha_by_topic=True,
                                eta=0.1,
                                eta_by_topic=False,
                                tmp_path=save_path, #Use SCRATCH if many models or big data set
                                save_path=save_path)
        else:
            models = []

        models_prerun = [pickle.load(open(os.path.join(save_path, 'Topic' + str(ntopic) + '.pkl'), 'rb')) for ntopic in n_topics if os.path.exists(os.path.join(save_path, 'Topic' + str(ntopic) + '.pkl'))]
        
        models.append(models_prerun)
        models = list(chain(*models))
        model=evaluate_models(models,
                            select_model=None,
                            return_model=True,
                            metrics=['Minmo_2011', 'loglikelihood'],
                            plot_metrics=False)

        # Add model to cisTopicObject
        self.cistopic_obj.add_LDA_model(model)

        # Run harmony
        harmony(self.cistopic_obj, vars_use=[self.batch_key])

        model = self.cistopic_obj.selected_model

        X_emb_harmony = model.cell_topic_harmony.T
        X_emb_harmony.index = X_emb_harmony.index.str.split('__').str[0]
        X_emb = model.cell_topic.T
        X_emb.index = X_emb.index.str.split('__').str[0]
        
        X_emb_harmony.to_csv(os.path.join(save_path, 'embedding_harmony.csv'))
        X_emb.to_csv(os.path.join(save_path, 'embedding.csv'))
          
        results = {
            'embedding_path': os.path.join(save_path, 'embedding.csv'),
            'embedding_harmony_path': os.path.join(save_path, 'embedding_harmony.csv'),
            'model_path': save_path
        }
        return results

# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment

# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.run()
