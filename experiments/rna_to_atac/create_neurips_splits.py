import anndata as ad
import os
import numpy as np

data_path = '/storage/groups/ml01/workspace/laura.martens/atac_poisson_data/data/neurips/phase2-private-data/'
save_path = os.path.join(data_path, 'predict_modality', 'openproblems_bmmc_multiome_phase2_rna')

path = os.path.join(data_path, 'common/openproblems_bmmc_multiome_phase2', 'openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2.h5ad')

adata = ad.read(path)

neurips_test = ad.read(os.path.join(data_path, 'predict_modality', 'openproblems_bmmc_multiome_phase2_rna', 'openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_test_mod2.h5ad'))

neurips_peaks = neurips_test.var_names.values

# Create sets of 10000 peaks
peaks = adata.var_names[~adata.var_names.isin(neurips_peaks)].values #exclude Neurips peaks

np.random.seed(0)
peaks = np.random.permutation(peaks)

splits=np.array_split(peaks, np.arange(start=10000, stop=len(peaks), step=10000))[:-1]

splits.append(neurips_peaks)

print(len(splits))

for i, split in enumerate(splits):
    print(i)
    a_split = adata[:, split]
    a_train = a_split[a_split.obs.is_train].copy()
    a_train.write(os.path.join(save_path, 
                               f'openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod2_split_{i}.h5ad'))
    a_test=a_split[~a_split.obs.is_train].copy()
    a_test.write(os.path.join(save_path, 
                f'openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_test_mod2_split_{i}.h5ad'))
    