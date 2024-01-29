#!/bin/bash

#SBATCH --job-name Nonefr20_hyper_tune_pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH -p voltadebug
#SBATCH -t 4:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --cpus-per-task 5
#SBATCH --mem-per-cpu 6GB

echo "HI"
module load python
module load sqlite #없어도 되는 듯?
source activate VAE_3DCNN_older_MONAI
#SBATCH --array=0-1 #upto (number of tasks) -1  there are 
##강제로 기다리기

cd ..

python main_optuna_fix_see_if_same.py --pretrained_path /sdcc/u/dyhan316/misc_VAE/junbeom_weights/UKByAa64a.pth --mode finetuning --train_num 100 --layer_control tune_all --stratify strat --random_seed 0 --task ABCD_sex --input_option yAware --batch_size 64 --save_path finetune_test_see_if_same --binary_class True --run_where sdcc