#!/bin/bash

#SBATCH --job-name 550_UKB_early  #job name을 다르게 하기 위해서
#SBATCH -t 8:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --cpus-per-task=8 #같이 하는게 훨씬 빠름(?)(test해보기.. .전에 넣은것이랑 비교해서)
#SBATCH --mem-per-cpu=4GB


##강제로 기다리기
##SBATCH --nodelist=node3
#sleep_list=( 1 30 60 90 120 ) #( 1e-3 1e-4 1e-5 ) #default : 1e-4, but found out that that is too small for freeze
#sleep_time="${sleep_list[${SLURM_ARRAY_TASK_ID}]}"

#sleep $sleep_time #sleep for this much time 

###만약 BT하려면 input_option을 바꿔야함!

#직접 정해주기
base_dir=/scratch/connectome/study_group/VAE_ADHD/junbeom_weights
task=ABCD_sex #( ADNI_sex ADNI_age ) #task_list=( ADNI_ALZ_ADCN ADNI_ALZ_ADMCI ADNI_ALZ_CNMCI ADNI_sex ADNI_age )

weight=${base_dir}/UKByAa64a.pth  #( None ${base_dir}/UKByAa64a.pth ) #weight_list=( None ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 

num=550


cd ../.. #move to directory above and execute stuff 

#use ; wait!! https://unix.stackexchange.com/questions/541311/bash-wait-for-all-subprocesses-of-script
##trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/) (and https://www.cyberciti.biz/faq/bash-iterate-array/ )

##freeze

#for loop등등 하던지 하려고했는데 and 문이 잘안되서 이렇게함!
batch_1=32 #if doing 20  
#batch_2=16

#if batch size : 64, can only do one! (maybe increase batch size?)
#batch size :64, can do ONE TUNEALL AND ONE FREEZE (in lab)
sleep 1 ; python3 main_optuna_fix_5.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify iter_strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_RESULTS/TEST_2/550_cosine_annealing_decay --binary_class True --run_where lab --lr_schedule cosine_annealing_decay --lr_range 2e-6/2e-4 --wd_range 3e-4/3e-2  &\
sleep 5 ; python3 main_optuna_fix_5.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify iter_strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_RESULTS/TEST_2/550_cosine_annealing_decay --binary_class True --run_where lab --early_criteria loss --lr_schedule cosine_annealing_decay --lr_range 2e-6/2e-4 --wd_range 3e-4/3e-2
#doing two possible because batch size of 32!

#doing freezing (check RAM usage, may be able to add even more )
#sleep 1 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True --run_where lab &\
#sleep 5 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True --run_where lab &\
#sleep 10 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True --run_where lab ; wait

#12cpus, 2 workers1:52 8 => 1:56 32 => 2:01 40
#38 17 => 40 29 => 41 35 => 43 43 

#2cpus, 1 worker : 2:08 0 => 2:14 24 
#2cpus 6 worker : 2:24 12(~15) => 2:26 24 => 32c 54 =>  


