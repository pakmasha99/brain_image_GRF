#!/bin/bash
#SBATCH -t 240:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --gpus-per-node=2
#SBATCH -J b32_yaware
#SBATCH --chdir=../
#SBATCH -o logs/%j-%x.out
#SBATCH -e logs/%j-%x.err

set +x

#will use 16 cpus!! (num of workers)
# -C : constraints 
#-n : ntasks
#-c : --cpus-per-task
#-G : --gpus-per-task

#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
#conda activate 3DCNN

#env | grep SLURM


#try later with 64!! ans 128!!

srun python main.py --mode pretraining --framework yaware --ckpt_dir ./ckpt_intelligence_0.01temp --batch_size 32 --label_name intelligence --sigma 0.006176507443170335


