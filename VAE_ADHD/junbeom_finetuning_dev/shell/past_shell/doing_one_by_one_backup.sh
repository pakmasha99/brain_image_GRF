#!/bin/bash

#SBATCH --job-name hyper_tune_pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH -p voltadebug
#SBATCH -t 4:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --array=0-2 #upto (number of tasks) -1  there are 

echo "HI"
module load python
source activate VAE_3DCNN_older_MONAI


#for i in `ls`; do echo $i ; ls -R $i | grep "eval_stats.txt" ; done
#위에걸로 찾고 하기 
###만약 BT하려면 input_option을 바꿔야함!

#직접 정해주기
base_dir=/sdcc/u/dyhan316/misc_VAE/junbeom_weights
task=ADNI_sex #( ADNI_sex ADNI_age ) #task_list=( ADNI_ALZ_ADCN ADNI_ALZ_ADMCI ADNI_ALZ_CNMCI ADNI_sex ADNI_age )
num=20
weight=None #${base_dir}/UKByAa64a.pth #( None ${base_dir}/UKByAa64a.pth ) #weight_list=( None ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 


##trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/) (and https://www.cyberciti.biz/faq/bash-iterate-array/ )

#things that are changed (lr : slurm task array id로 control)
lr_list=( 3e0 1e0 3e-1 ) #default : 1e-4, but found out that that is too small for freeze
lr="${lr_list[${SLURM_ARRAY_TASK_ID}]}"

#batch_list=( 8 16 ) #32 )
#wd_list=(5e-5 5e-6 )#( 5e-4 5e-5 5e-6 ) #default : 5e-5
batch=16 #try 16 too later
wd=5e-8 #do ( 5e-3 5e-5 5e-7 )


cd .. #move to directory above and execute stuff 

#use ; wait!! https://unix.stackexchange.com/questions/541311/bash-wait-for-all-subprocesses-of-script

#for batch in "${batch_list[@]}"
#do
#for wd in "${wd_list[@]}"
#do 

##freeze
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none ; wait

#don't run tune_all because we only get 4 hrs of run time for voltadebug, so only this can be run during that time (I think)(maybe even less)

###tune_all (16: 세개씩, 32: 1개싹 (or 2개?) 묶어서 하기)
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none ; wait
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none ; wait

#done
#done
#


##$#!/bin/bash
##$#SBATCH -t 7:00:00
##$#SBATCH -n 1
##$#SBATCH --ntasks-per-node=1
##$#SBATCH -c 16
##$#SBATCH --gpus-per-node=4
##$#SBATCH --partition=volta
##$#SBATCH --output=slurm_logs/R-optuna-%j-%x.out
##$#SBATCH --chdir=../../
##$#SBATCH --mail-user=kjb961013@snu.ac.kr