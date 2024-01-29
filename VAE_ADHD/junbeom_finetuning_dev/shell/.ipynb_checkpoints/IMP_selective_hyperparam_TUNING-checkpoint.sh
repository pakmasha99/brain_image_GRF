#!/bin/bash

#SBATCH --job-name hyper_tune_pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH --nodes=1
#SBATCH --nodelist=node1 #used node4
#SBATCH -t 120:00:00 # Time for running job #길게 10일넘게 잡음
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --array=0-3 #upto (number of tasks) -1  there are 


##trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/) (and https://www.cyberciti.biz/faq/bash-iterate-array/ )

###REMOVE test below later!!
#task_list=( ADNI_sex ADNI_age ABCD_sex )
#task="${task_list[${SLURM_ARRAY_TASK_ID}]}" #select the corresponding jobarray part from the list of tasks to do 
task=ADNI_age
num=20
lr_list=( 5e-4 1e-3 1e-4 1e-5 ) #default : 1e-4, but found out that that is too small for freeze
lr="${lr_list[${SLURM_ARRAY_TASK_ID}]}"


base_dir=/scratch/connectome/study_group/VAE_ADHD/junbeom_weights
weight_list=( None ${base_dir}/UKByAa64a.pth ) 
wd_list=( 5e-4 5e-5 5e-6 ) #default : 5e-5
batch_list=( 8 16 32 )

cd .. #move to directory above and execute stuff 

#use ; wait!! https://unix.stackexchange.com/questions/541311/bash-wait-for-all-subprocesses-of-script

for weight in "${weight_list[@]}"
do 
for batch in "${batch_list[@]}"
do
for wd in "${wd_list[@]}"
do 
#echo $num ; echo $weight ; echo $task
#do 5 times, so that we can average the results

##freeze
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none ; wait

####don't do tune_all rn because time limits
#tune_all이면 batch 32 : 한번에 하나씩 만하기가능, batch 16 : 한번에 두개씩만 하기가능
###tune_all (두개씩 묶어서 하기)
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none ; wait
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none ; wait
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd --BN none ; wait

done
done
done


##batch 8이면 위에서처럼 freeze, tune_alld을 6개, 3개씩 번갈아면서 하도록 하는게 좋은듯하다!!! (램의 50%를 넘어가게 되면  python script가 properly exit을 하기전에 다음 python script가 들어와서 GPU OOM 뜸)