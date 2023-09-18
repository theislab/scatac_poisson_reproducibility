# Modeling ATAC data quantitatively
Study to compare the modeling of the ATAC count data instead of the binarized matrix. All models have been implemented based on the [scvi-tools suite](https://github.com/scverse/scvi-tools). IMPORTANT: The model is now implemented as an extension of `scvi-tools`. See [here](https://github.com/lauradmartens/scvi-tools) for installation.

To setup the environment to reproduce the results, run:
```python
pip install -e .
```

- `poisson_atac/`: contains the code for the models, the dataloading, and plotting functions.
- `poisson_atac/experiments`: Experiment files to reproduce the results of the paper. Each folder contains a `.yaml` file with the seml configuration and a `.py` file with the code to run the experiment.
- `notebooks`: Includes an example analysis notebook for using the Poisson VAE model `scvi_tools_example_notebook.ipynb` and notebooks to reproduce the Figures of the paper. Additionally, we showcase the use together with Signac in `scvi_tools_example_notebook_in_R.ipynb`. Please follow the [tutorial](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/python_in_R.html) of `scvi-tools` to setup the environment. 
 
All experiments where run through [seml](https://github.com/TUM-DAML/seml).
The entry function is `ExperimentWrapper.__init__` in the respective experiment runners `experiment_runner.py`.

As you will not be able to connect to the mongoDB via SEML, you have to use the provided part of the database in the respective experiments folder. To align with the notebooks, simply define your own `load_config` function similar to this: 

```python
import json 
import pandas as pd

def load_config(seml_collection, model_hash):
    file_path = f'{seml_collection}.json' # Provide path to json

    with open(file_path) as f:
        file_data = json.load(f)
    
    config = pd.json_normalize(file_data, sep='.')
    return config
```

