#!/bin/bash

#SBATCH --job-name seedUF100_hyper_tune_pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH -t 24:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --nodelist=node3
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --array=0-3 #upto (number of tasks) -1  there are 
#SBATCH --cpus-per-task=12 #같이 하는게 훨씬 빠름(?)(test해보기.. .전에 넣은것이랑 비교해서)
#SBATCH --mem-per-cpu=4GB


##강제로 기다리기
sleep_list=( 1 30 60 90 120 ) #( 1e-3 1e-4 1e-5 ) #default : 1e-4, but found out that that is too small for freeze
sleep_time="${sleep_list[${SLURM_ARRAY_TASK_ID}]}"

sleep $sleep_time #sleep for this much time 

###만약 BT하려면 input_option을 바꿔야함!

#직접 정해주기
base_dir=/scratch/connectome/dyhan316/VAE_ADHD/barlowtwins/pretrain_results
task=ABCD_sex #( ADNI_sex ADNI_age ) #task_list=( ADNI_ALZ_ADCN ADNI_ALZ_ADMCI ADNI_ALZ_CNMCI ADNI_sex ADNI_age )


cd ../.. #move to directory above and execute stuff 

#use ; wait!! https://unix.stackexchange.com/questions/541311/bash-wait-for-all-subprocesses-of-script
##trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/) (and https://www.cyberciti.biz/faq/bash-iterate-array/ )

##freeze

#for loop등등 하던지 하려고했는데 and 문이 잘안되서 이렇게함!
batch_1=32 #if doing 20  
weight=${base_dir}/ABCDbt128a102.pth
num=100



#if batch size : 64, can only do one! 
#batch size :64, can do ONE TUNEALL AND ONE FREEZE (in lab)
#sleep 1 ; python3 main_optuna.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 0 --task $task --input_option BT_org --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True --run_where lab --eval_mode False


sleep 1 ; python3 main_optuna.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option BT_org --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True --run_where lab --eval_mode False &\
sleep 10 ; python3 main_optuna.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option BT_org --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True -- run_where lab --eval_mode False &\
sleep 20 ; python3 main_optuna.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option BT_org --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True -- run_where lab --eval_mode False &\
sleep 30 ; python3 main_optuna.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option BT_org --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True -- run_where lab --eval_mode False &\
sleep 40 ; python3 main_optuna.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option BT_org --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True --run_where lab --eval_mode False ; wait










#doing freezing (check RAM usage, may be able to add even more )
#sleep 1 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True --run_where lab &\
#sleep 5 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True --run_where lab &\
#sleep 10 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test_same_seed_lab --binary_class True --run_where lab ; wait

#12cpus, 2 workers1:52 8 => 1:56 32 => 2:01 40
#38 17 => 40 29 => 41 35 => 43 43 

#2cpus, 1 worker : 2:08 0 => 2:14 24 
#2cpus 6 worker : 2:24 12(~15) => 2:26 24 => 32c 54 =>  

