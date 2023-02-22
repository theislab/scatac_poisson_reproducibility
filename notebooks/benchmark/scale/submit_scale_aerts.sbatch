#!/bin/bash
#SBATCH -o scale_%A.out
#SBATCH -e scale_%A.out
#SBATCH -J scale
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4 #CPU cores required
#SBATCH --mem 90Gb
#SBATCH --time 48:00:00
#SBATCH --nice=10000
##SBATCH --exclude=supergpu03pxe,supergpu02pxe
source $HOME/.bashrc
env_name=poisson_atac_new
path_to_conda_sh=$(conda info --base)/etc/profile.d/conda.sh
source $path_to_conda_sh

conda activate $env_name

input='/lustre/groups/ml01/workspace/laura.martens/data/aerts_fly_brain/All_timepoints_filtered.h5ad'
output="/lustre/groups/ml01/workspace/laura.martens/atac_poisson_data/benchmark/aerts/scale/"

#change number of ks according to number of expected cell types
python SCALE.py -d $input -o $output -k 74 --min_cells 0 --min_peaks 0
