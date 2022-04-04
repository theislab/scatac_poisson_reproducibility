#!/bin/bash
#SBATCH -o slurm_outputs/cistopic_%A_%a.out
#SBATCH -e slurm_outputs/cistopic_%A_%a.out
#SBATCH -J cisTopic
#SBATCH -p cpu_p
#SBATCH -c 12
#SBATCH --mem 80Gb
#SBATCH --time 24:00:00
#SBATCH --nice=10000
#SBATCH --array=0


source $HOME/.bashrc


source activate signac

batches=(s1d1 s1d2 s1d3 s2d1 s2d4 s2d5 s3d10 s3d3 s3d6 s3d7) #Change accordingly(hematopoiesis neurips) #

echo ${batches[@]}

dataset=${batches[$SLURM_ARRAY_TASK_ID]}
~/miniconda3/envs/signac/bin/Rscript run_cistopic.R $dataset