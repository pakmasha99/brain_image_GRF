#!/bin/bash

#SBATCH --job-name Nonefr20_hyper_tune_pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH -p voltadebug
#SBATCH -t 4:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --array=0-1 #upto (number of tasks) -1  there are 


echo "HI"
module load python
module load sqlite #없어도 되는 듯?
source activate VAE_3DCNN_older_MONAI

##강제로 기다리기

sleep_list=( 1 60 120 180 240 ) #( 1e-3 1e-4 1e-5 ) #default : 1e-4, but found out that that is too small for freeze
sleep_time="${sleep_list[${SLURM_ARRAY_TASK_ID}]}"

sleep $sleep_time #sleep for this much time 


##밑에 cpu mem 제한두면 오히려 느려지기는 함
#SBATCH --cpus-per-task=5 #같이 하는게 훨씬 빠름
#SBATCH --mem-per-cpu=5GB

#for i in `ls`; do echo $i ; ls -R $i | grep "eval_stats.txt" ; done
#위에걸로 찾고 하기 
###만약 BT하려면 input_option을 바꿔야함!

#직접 정해주기
base_dir=/sdcc/u/dyhan316/misc_VAE/junbeom_weights
task=ADNI_ALZ_ADCN #( ADNI_sex ADNI_age ) #task_list=( ADNI_ALZ_ADCN ADNI_ALZ_ADMCI ADNI_ALZ_CNMCI ADNI_sex ADNI_age )

weight=None #${base_dir}/UKByAa64a.pth #( None ${base_dir}/UKByAa64a.pth ) #weight_list=( None ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 

num=20


cd .. #move to directory above and execute stuff 

#use ; wait!! https://unix.stackexchange.com/questions/541311/bash-wait-for-all-subprocesses-of-script
##trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/) (and https://www.cyberciti.biz/faq/bash-iterate-array/ )

##freeze

#for loop등등 하던지 하려고했는데 and 문이 잘안되서 이렇게함!
batch_1=16 #if doing 20  
#batch_2=16

####deciding number of things to run in parallel
#freeze : 
# 8 => 14 ; 16 => 11; 32 => 10
#tuneall : 
# 8 => 7 ; 16 => 4 ; 32 => 2 


#FREEZE batch size 하나로 고정해놓고, 8개 해서 평균내기
sleep 1 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\
sleep 2 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 1 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\
sleep 3 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 2 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\
sleep 4 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 3 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\
sleep 5 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 4 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\
sleep 6 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 5 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\
sleep 7 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 6 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\
sleep 8 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 7 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc ; wait


#TUNEALL batch size 하나로 고정해놓고, 8개 해서 평균내기 (not 6) (2개씩 한다)
#sleep 1 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\
#sleep 2 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 1 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc ; wait
#sleep 3 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 2 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\ 
#sleep 4 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 3 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc ; wait
#sleep 5 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 4 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\ 
#sleep 6 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 5 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc ; wait
#sleep 7 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 6 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\
#sleep 8 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 7 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc ; wait


#python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 8 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\
#python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 9 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test --binary_class True --run_where sdcc &\
#python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 10 --task $task --input_option yAware --batch_size $batch_1  --save_path finetune_test --binary_class True --run_where sdcc &\
#python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 11 --task $task --input_option yAware --batch_size $batch_1  --save_path finetune_test --binary_class True --run_where sdcc &\
#python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 12 --task $task --input_option yAware --batch_size $batch_1  --save_path finetune_test --binary_class True --run_where sdcc &\
#python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 13 --task $task --input_option yAware --batch_size $batch_1  --save_path finetune_test --binary_class True --run_where sdcc &\
#python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 14 --task $task --input_option yAware --batch_size $batch_1  --save_path finetune_test --binary_class True --run_where sdcc &\
#python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 15 --task $task --input_option yAware --batch_size $batch_1  --save_path finetune_test --binary_class True --run_where sdcc ; wait


##$#!/bin/b