#!/bin/bash

#SBATCH --job-name nopr80_hyper_tune_pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH -t 24:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --array=0-2 #upto (number of tasks) -1  there are 
#SBATCH --cpus-per-task=6 #not that much needed, sicne only doing 100 samples
#SBATCH --mem-per-cpu=4GB


##강제로 기다리기
##SBATCH --nodelist=node3
sleep_list=( 1 30 60 90 120 ) #( 1e-3 1e-4 1e-5 ) #default : 1e-4, but found out that that is too small for freeze
sleep_time="${sleep_list[${SLURM_ARRAY_TASK_ID}]}"

sleep $sleep_time #sleep for this much time 

###만약 BT하려면 input_option을 바꿔야함!

#직접 정해주기
base_dir=/scratch/connectome/study_group/VAE_ADHD/junbeom_weights
task=ADNI_ALZ_ADCN #( ADNI_sex ADNI_age ) #task_list=( ADNI_ALZ_ADCN ADNI_ALZ_ADMCI ADNI_ALZ_CNMCI ADNI_sex ADNI_age )

weight=${base_dir}/UKByAa64a.pth  #( None ${base_dir}/UKByAa64a.pth ) #weight_list=( None ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 

num=80


cd ../.. #move to directory above and execute stuff 


#for loop등등 하던지 하려고했는데 and 문이 잘안되서 이렇게함!
batch_1=64 #if doing 20, 64 is twice possible  
num_workers=6
#batch_2=16

#if batch size : 64, can only do one! 
#batch size :64, can do ONE TUNEALL AND ONE FREEZE (in lab)
sleep 1 ; python3 main_optuna_fix_7.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan_iter_strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_RESULTS/TEST_4/ADCN_100_samples_100_epoch_balan_no_prune --binary_class True --run_where lab --early_criteria loss --lr_schedule cosine_annealing_decay  --AMP True  --wandb_name TEST_2 --workers $num_workers --patience 20 --nb_epochs 100 --prune False  &\
sleep 5 ; python3 main_optuna_fix_7.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan_iter_strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_RESULTS/TEST_4/ADCN_100_samples_100_epoch_balan_no_prune --binary_class True --run_where lab --early_criteria loss --lr_schedule cosine_annealing_decay  --AMP True  --wandb_name TEST_2 --workers $num_workers --patience 20 --nb_epochs 100 --prune False ; wait


#balan iter strat?

#--prune True #prune removed because not doing it when small dataset


#DO TWO!