#!/bin/bash
#SBATCH -o slurm_outputs/cistopic_%A.out
#SBATCH -e slurm_outputs/cistopic_%A.out
#SBATCH -J cisTopic
#SBATCH -p cpu_p
#SBATCH -c 20
#SBATCH --mem 80Gb
#SBATCH --time 24:00:00
#SBATCH --nice=10000


source $HOME/.bashrc


source activate signac

~/miniconda3/envs/signac/bin/Rscript run_cistopic.R