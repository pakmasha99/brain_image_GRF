#!/bin/bash

#SBATCH --job-name seedNFTall300_hyper_tune_pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH -p volta
#SBATCH -t 24:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --array=0-5 #upto (number of tasks) -1  there are 
#SBATCH --cpus-per-task=5 #같이 하는게 훨씬 빠름(?)(test해보기.. .전에 넣은것이랑 비교해서)
#SBATCH --mem-per-cpu=8GB

echo "HI"
module load python
module load sqlite #없어도 되는 듯?
source activate VAE_3DCNN_older_MONAI

##강제로 기다리기
sleep_list=( 1 20 30 40 50 ) #( 1e-3 1e-4 1e-5 ) #default : 1e-4, but found out that that is too small for freeze
sleep_time="${sleep_list[${SLURM_ARRAY_TASK_ID}]}"

sleep $sleep_time #sleep for this much time 


##밑에 cpu mem 제한두면 오히려 느려지기는 함
#SBATCH --cpus-per-task=5 #같이 하는게 훨씬 빠름
#SBATCH --mem-per-cpu=5GB

#for i in `ls`; do echo $i ; ls -R $i | grep "eval_stats.txt" ; done
#위에걸로 찾고 하기 
###만약 BT하려면 input_option을 바꿔야함!

#junbeom weights
base_dir=/sdcc/u/dyhan316/misc_VAE/junbeom_weights
task=ADNI_ALZ_ADCN #( ADNI_sex ADNI_age ) #task_list=( ADNI_ALZ_ADCN ADNI_ALZ_ADMCI ADNI_ALZ_CNMCI ADNI_sex ADNI_age )



weight=None #${base_dir}/UKByAa64a.pth #( None ${base_dir}/UKByAa64a.pth ) #weight_list=( None ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 


#ABCD
#base_dir=/sdcc/u/dyhan316/misc_VAE/BT_weights/BT_weights_no_norm5/
#task=ABCD_sex

#weight=${base_dir}/ABCDbt128a102.pth
#CHANGE TO BT_org too!!!!


num=300


cd .. #move to directory above and execute stuff 

#use ; wait!! https://unix.stackexchange.com/questions/541311/bash-wait-for-all-subprocesses-of-script
##trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/) (and https://www.cyberciti.biz/faq/bash-iterate-array/ )

##freeze

#for loop등등 하던지 하려고했는데 and 문이 잘안되서 이렇게함!
batch_1=64 #if doing 20  
#batch_2=16



#if batch size : 64, can only do one! 
#batch size :64, can do ONE TUNEALL AND FOUR FREEZE (reduced to TWO FREEZE just in case)
sleep 1 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test_same_seed --binary_class True --run_where sdcc &\
sleep 2 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test_same_seed --binary_class True --run_where sdcc  &\
sleep 3 ; python3 main_optuna_fix.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path finetune_test_same_seed --binary_class True --run_where sdcc  ; wait


