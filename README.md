# Modeling ATAC data quantitatively
Study to compare the modeling of the ATAC count data instead of the binarized matrix.

To setup the environment, run:
```python
python setup.py install -e .
```

- `poisson_atac/`: contains the code for the models, the dataloading, and plotting functions.
- `experiments`: Each folder contains a `README.md` with the experiment description and a `.yaml` file with the seml configuration.
- `notebooks`: Example analysis notebooks using the Poisson VAE model and notebooks to reproduce the Figures.
 
All experiments where run through [seml](https://github.com/TUM-DAML/seml).
The entry function is `ExperimentWrapper.__init__` in the respective experiment runners `experiment_runner.py`.

