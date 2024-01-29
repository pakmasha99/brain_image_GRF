#! /bin/bash

#SBATCH --job-name ABCD_T1_8000_SSL  #job name을 다르게 하기 위해서
#SBATCH --nodes=1
#SBATCH --nodelist=node1 #used node2
#SBATCH -t 96:00:00 # Time for running job #길게 10일넘게 잡음
#SBATCH -o ./ckpt_gpu_aug_test/2_gpu_pin_false_output_%J.out #%j : job id 가 들어가는 것
#SBATCH -e ./ckpt_gpu_aug_test/2_gpu_pin_false_error_%J.error
#SBATCH --ntasks=1
#SBATCH --mail-user=dyhan0316@gmail.com
#SBATCH --mem-per-cpu=5GB #최대한 GPU VRAM 하나+ extra정도는 있어야 하지 않을까
#SBATCH --cpus-per-task=16 #16
#SBATCH --gpus-per-task=2

#python main_3D.py /scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked --checkpoint-dir ./ckpt_polaris_gpu_1_REAL_pin_TRUE --batch-size 32 --print-freq 5 --epochs 10 --workers 1

w_per_gpu=2
batch_per_gpu=16


python main_3D.py /scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked \
--checkpoint-dir ./ckpt_gpu_aug_test/REAK_ckpt_polaris_gpu_2_w_per_gpu${w_per_gpu}_batch_per_gpu_${batch_per_gpu} --batch-size $((2*${batch_per_gpu})) --print-freq 5 --epochs 5 --workers $w_per_gpu


python BACKUP_main_3D.py /scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked \
--checkpoint-dir ./ckpt_gpu_aug_test/REAL_ckpt_polaris_gpu_0_w_per_gpu${w_per_gpu}_batch_per_gpu_${batch_per_gpu} --batch-size $((2*${batch_per_gpu})) --print-freq 5 --epochs 5 --workers $w_per_gpu




w_per_gpu=1
batch_per_gpu=16


python main_3D.py /scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked \
--checkpoint-dir ./ckpt_gpu_aug_test/REAL_ckpt_polaris_gpu_2_w_per_gpu${w_per_gpu}_batch_per_gpu_${batch_per_gpu} --batch-size $((2*${batch_per_gpu})) --print-freq 5 --epochs 5 --workers $w_per_gpu

python BACKUP_main_3D.py /scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked \
--checkpoint-dir ./ckpt_gpu_aug_test/REAL_ckpt_polaris_gpu_0_w_per_gpu${w_per_gpu}_batch_per_gpu_${batch_per_gpu} --batch-size $((2*${batch_per_gpu})) --print-freq 5 --epochs 5 --workers $w_per_gpu
