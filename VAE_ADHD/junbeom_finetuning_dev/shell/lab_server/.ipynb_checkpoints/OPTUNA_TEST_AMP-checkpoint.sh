#!/bin/bash

#SBATCH --job-name TEST_AMP  #job name을 다르게 하기 위해서
#SBATCH -t 04:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --cpus-per-task=32 #같이 하는게 훨씬 빠름(?)(test해보기.. .전에 넣은것이랑 비교해서)
#SBATCH --mem-per-cpu=4GB

cd ../..


python3 main_optuna_fix_5_dev_amp.py --pretrained_path ./weights/y-Aware_Contrastive_MRI_epoch_99.pth --mode finetuning --train_num 1000 --layer_control tune_all --stratify iter_strat --random_seed 0 --task ABCD_sex --input_option yAware --binary_class True --save_path finetune_RESULTS/TEST_3/test_AMP/with_AMP_16_per_task --AMP True --run_where lab --batch_size 32 --lr_schedule cosine_annealing_decay &\
python3 main_optuna_fix_5_dev_amp.py --pretrained_path ./weights/y-Aware_Contrastive_MRI_epoch_99.pth --mode finetuning --train_num 1000 --layer_control tune_all --stratify iter_strat --random_seed 0 --task ABCD_sex --input_option yAware --binary_class True --save_path finetune_RESULTS/TEST_3/test_AMP/wo_AMP_16_per_task --AMP False --run_where lab --batch_size 32 --lr_schedule cosine_annealing_decay



###python3 main_optuna_fix_5_dev_amp.py --pretrained_path ./weights/y-Aware_Contrastive_MRI_epoch_99.pth --mode finetuning --train_num 50 --layer_control tune_all --stratify iter_strat --random_seed 0 --task test --input_option yAware --binary_class True --save_path finetune_trash/wo_AMP --AMP False --run_where sdcc