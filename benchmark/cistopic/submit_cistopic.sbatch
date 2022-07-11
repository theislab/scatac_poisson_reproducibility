#!/bin/bash
#SBATCH -o cistopic_%A.out
#SBATCH -e cistopic_%A.out
#SBATCH -J cistopic
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

/home/icb/laura.martens/miniconda3/envs/archr/bin/Rscript $1 #/storage/groups/ml01/code/laura.martens/scatac_poisson_private/benchmark/cistopic/run_cistopic_neurips.R