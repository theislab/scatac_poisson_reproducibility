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

