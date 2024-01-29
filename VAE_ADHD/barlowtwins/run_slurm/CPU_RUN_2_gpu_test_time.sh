#! /bin/bash

#SBATCH --job-name time_test_TMPlambda_20%_64_ABCD_T1_8000_SSL  #job name을 다르게 하기 위해서
#SBATCH --nodes=1
#SBATCH --nodelist=node3 #used node2
#SBATCH -t 0:30:00 # Time for running job #길게 10일넘게 잡음
#SBATCH -o ./shell_output/CPU_ABCD_SSL_output_%J.out #%j : job id 가 들어가는 것
#SBATCH -e ./shell_output/CPU_ABCD_SSL_error_%J.error
#SBATCH --ntasks=2
#SBATCH --mail-user=dyhan0316@gmail.com
#SBATCH --mem-per-cpu=5GB #6#최대한 GPU VRAM 하나+ extra정도는 있어야 하지 않을까
#SBATCH --cpus-per-task=16#16
#SBATCH --gpus-per-task=1

batch_size=64 #36 per gpu max
lr_w=0.2 #default 0.2
lr_b=0.0048 #default
w_d=1e-6 #default : 1e-6
lambd=0.00102 #default:0.0051
epochs=500 #epoch을 무엇을 두느냐에 따라 완전히 달라짐! 
base_dir=/scratch/connectome/dyhan316/VAE_ADHD/barlowtwins/

#/scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked #ABCD data 
#/scratch/connectome/dyhan316/trash_ABCD_sample_data    #sample data

##CHANGE DATA DIR AND SAVE_EVERY to 25 and print-freq to 10
#/scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked
#/tmp/VAE_ADHD/ABCD_preprocessec_masked 
python $base_dir/BACKUP_main_3D_test_time.py /tmp/VAE_ADHD/ABCD_preprocessec_masked   --checkpoint-dir $base_dir/test_time/TEST_persistent_True_LESS_MEMORY_MORE_MORE_CPU\
 --batch-size $batch_size --print-freq 1 --save-every 50 --epochs $epochs  --lambd $lambd --workers 16 #--weight-decay $w_d #원래는 save-every 10, epochs 100

#below : additional changes if wanted
#--lambd $lambd --lr-w $lr_w

#$base_dir/main_3D.py : default, main_3D_different_port.py if wanting to do another job (so that port doens't collide)
#_wd_${w_d} if wanna vary the wd
#_lr_w_${lr_w} if wanna vary the lr_w
#_lambd_${lambd} if wanna vary the lambd
#python /scratch/connectome/dyhan316/VAE_ADHD/barlowtwins/main_3D.py /scratch/connectome/mieuxmin/UKB_t1_MNI/unzip_mni_1000 --batch-size 8 --print-freq 25 --checkpoint-dir /scratch/connectome/dyhan316/VAE_ADHD/barlowtwins/checkpoint_try_3D/batch_long_size_8 
