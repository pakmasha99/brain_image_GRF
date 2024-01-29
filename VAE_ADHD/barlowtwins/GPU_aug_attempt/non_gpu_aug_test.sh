#! /bin/bash

#SBATCH --job-name ABCD_T1_8000_SSL  #job name을 다르게 하기 위해서
#SBATCH --nodes=1
#SBATCH --nodelist=node3 #used node2
#SBATCH -t 96:00:00 # Time for running job #길게 10일넘게 잡음
#SBATCH -o ./0_gpu_output_%J.out #%j : job id 가 들어가는 것
#SBATCH -e ./0_gpu_error_%J.error
#SBATCH --ntasks=1
#SBATCH --mail-user=dyhan0316@gmail.com
#SBATCH --mem-per-cpu=5GB #최대한 GPU VRAM 하나+ extra정도는 있어야 하지 않을까
#SBATCH --cpus-per-task=8 #16
#SBATCH --gpus-per-task=1

python BACKUP_main_3D.py /scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked --checkpoint-dir ./ckpt_polaris_gpu_0_REAL --batch-size 32 --print-freq 5 --epochs 10 --workers 8
