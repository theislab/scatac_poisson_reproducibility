#!/bin/bash
#SBATCH -o signac_%A.out
#SBATCH -e signac_%A.out
#SBATCH -J signac
#SBATCH -p cpu_p
#SBATCH -c 32
#SBATCH --mem 80Gb
#SBATCH --time 72:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
env_name=archr
path_to_conda_sh=$(conda info --base)/etc/profile.d/conda.sh
source $path_to_conda_sh

conda activate $env_name

/home/icb/laura.martens/miniconda3/envs/archr/bin/Rscript /lustre/groups/ml01/code/laura.martens/scatac_poisson_private/benchmark/signac/signac_aerts.R