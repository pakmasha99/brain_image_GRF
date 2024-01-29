#!/bin/bash

#SBATCH --job-name CHA_SCRATCH  #job name을 다르게 하기 위해서
#SBATCH -t 96:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --array=0-2 #upto (number of tasks) -1  there are 
#SBATCH --cpus-per-task=16 #같이 하는게 훨씬 빠름(?)(test해보기.. .전에 넣은것이랑 비교해서)
#SBATCH --mem-per-cpu=4GB


##강제로 기다리기
##SBATCH --nodelist=node3
sleep_list=( 1 30 60 90 120 ) #( 1e-3 1e-4 1e-5 ) #default : 1e-4, but found out that that is too small for freeze
sleep_time="${sleep_list[${SLURM_ARRAY_TASK_ID}]}"

sleep $sleep_time #sleep for this much time 

###만약 BT하려면 input_option을 바꿔야함!

#직접 정해주기
base_dir=/scratch/connectome/study_group/VAE_ADHD/junbeom_weights
task=CHA_ASDGDD #( ADNI_sex ADNI_age ) #task_list=( ADNI_ALZ_ADCN ADNI_ALZ_ADMCI ADNI_ALZ_CNMCI ADNI_sex ADNI_age )

weight=None #${base_dir}/UKByAa64a.pth  #( None ${base_dir}/UKByAa64a.pth ) #weight_list=( None ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 

num=237 #maximum for iter_strat of CHA 


cd ../.. #move to directory above and execute stuff 

#only do iter_strat, without balan iter strat because not that imbalanced, and not that much data to begin with, so no loss in data is a must 
#maybe could try upweight if found to be useful

#for loop등등 하던지 하려고했는데 and 문이 잘안되서 이렇게함!
batch_1=32 #if doing 20  
prune=True
#batch_2=16

sleep 1 ; python3 main_optuna_fix_7.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify iter_strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_RESULTS/TEST_5/CHA_ASDGDD --binary_class True --run_where lab --early_criteria loss --lr_schedule cosine_annealing_decay --lr_range 2e-6/2e0 --wd_range 1e-2/1e1 --workers 16 --wandb_name TEST_2 --AMP True --prune $prune --nb_epochs 100 --patience 20 --lr_range 1e-4/1e-3 --wd_range 1e-2/1e0 &\
sleep 5 ; python3 main_optuna_fix_7.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify iter_strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_RESULTS/TEST_5/CHA_ASDGDD --binary_class True --run_where lab --early_criteria loss --lr_schedule cosine_annealing_decay --lr_range 2e-6/2e0 --wd_range 1e-2/1e1 --workers 16 --wandb_name TEST_2 --AMP True --prune $prune --nb_epochs 100 --patience 20 --lr_range 1e-4/1e-3 --wd_range 1e-2/1e0 &\
sleep 10 ; python3 main_optuna_fix_7.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify iter_strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_RESULTS/TEST_5/CHA_ASDGDD --binary_class True --run_where lab --early_criteria loss --lr_schedule cosine_annealing_decay --lr_range 2e-6/2e0 --wd_range 1e-2/1e1 --workers 16 --wandb_name TEST_2 --AMP True --prune $prune --nb_epochs 100 --patience 20 --lr_range 1e-4/1e-3 --wd_range 1e-2/1e0 ; wait

#python3 main_optuna_fix_7.py --pretrained_path ./weights/y-Aware_Contrastive_MRI_epoch_99.pth --mode finetuning --train_num 1013 --layer_control tune_all --stratify iter_strat --random_seed 0 --task ABCD_ADHD_strict --input_option yAware --binary_class True --save_path finetune_trash/ADHD_with_upweight --AMP True --run_where lab --nb_epochs 100 --upweight True --lr_schedule cosine_annealing_decay --workers 16 --

#####MODIFY ABOVE, WITH SOME PRUNING AND STUFF####(See if it works)(with full data)

#not sure if it'll work because it's strat, but whatever


