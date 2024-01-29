#! /bin/bash

#SBATCH --job-name yAware_UKB_100  #job name을 다르게 하기 위해서
#SBATCH --nodes=1
#SBATCH --nodelist=node3 #used node2
#SBATCH -t 200:00:00 # Time for running job #길게 10일넘게 잡음
#SBATCH -o ./slurm_output/output_%J.out #%j : job id 가 들어가는 것
#SBATCH -e ./slurm_output/error_%J.error
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1024MB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2


python ../main_jan.py --mode pretraining
