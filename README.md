# Modeling ATAC data quantitatively
Study to compare the modeling of the ATAC count data instead of the binarized matrix. All models have been implemented based on the [scvi-tools suite](https://github.com/scverse/scvi-tools).

To setup the environment, run:
```python
python setup.py install -e .
```

- `poisson_atac/`: contains the code for the models, the dataloading, and plotting functions.
- `poisson_atac/experiments`: Experiment files to reproduce the results of the paper. Each folder contains a `.yaml` file with the seml configuration and a `.py` file with the code to run the experiment.
- `notebooks`: Includes an example analysis notebook for using the Poisson VAE model and notebooks to reproduce the Figures of the paper.
 
All experiments where run through [seml](https://github.com/TUM-DAML/seml).
The entry function is `ExperimentWrapper.__init__` in the respective experiment runners `experiment_runner.py`.

As you will not be able to connect to the mongoDB via SEML, you have to use the provided part of the database in the respective experiments folder. To align with the notebooks, simply define your own `load_config` function similar to this: 

```python
import json 

def load_config(seml_collection, model_hash):
    file_path = f'{seml_collection}.json' # Provide path to json

    with open(file_path) as f:
        file_data = json.load(f)
    
    for config in file_data:
        if config['config_hash']==model_hash:
            config = config['config']
    return config
```
The trained models will also soon be provided.

