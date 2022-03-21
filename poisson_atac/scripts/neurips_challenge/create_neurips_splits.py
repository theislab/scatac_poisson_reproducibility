import anndata as ad
import os
import numpy as np

data_path = '/mnt/data/output/datasets/'
save_path = os.path.join(data_path, 'predict_modality', 'openproblems_bmmc_multiome_phase2_rna')

path = os.path.join(data_path, 'common/openproblems_bmmc_multiome_phase2', 'openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2.h5ad')

adata = ad.read(path)

neurips_test = ad.read(os.path.join(data_path, 'predict_modality', 'openproblems_bmmc_multiome_phase2_rna', 'openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_test_mod2.h5ad'))

neurips_peaks = neurips_test.var_names.values

# Create sets of 10000 peaks
peaks = adata.var_names[~adata.var_names.isin(neurips_peaks)].values

np.random.seed(0)
peaks = np.random.permutation(peaks)

splits=np.array_split(peaks, np.arange(start=10000, stop=len(peaks), step=10000))[:-1]

splits.append(neurips_peaks)

print(len(splits))

for i, split in enumerate(splits):
    print(i)
    a_split = adata[:, split].copy()
    a_split[a_split.obs.is_train].write(os.path.join(save_path, 
                                                     f'openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod2_split_{i}.h5ad'))
    a_split[~a_split.obs.is_train].write(os.path.join(save_path, 
                                                 f'openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_test_mod2_split_{i}.h5ad'))
    