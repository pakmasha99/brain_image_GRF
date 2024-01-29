#!/bin/bash

#SBATCH --job-name hyper_tune_pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH --nodes=1
#SBATCH --nodelist=node3 #used node4
#SBATCH -t 120:00:00 # Time for running job #길게 10일넘게 잡음
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --array=0-3 #upto (number of tasks) -1  there are 

#sometimes not doing stratify =balan  could be important, as 그렇게하면 majority case가 minority case의 갯수만큼 되도록 강제로 sample을 버리니

##trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/) (and https://www.cyberciti.biz/faq/bash-iterate-array/ )

###REMOVE test below later!!
task_list=( ADNI_ALZ_ADCN ADNI_sex ADNI_age ABCD_sex )
task="${task_list[${SLURM_ARRAY_TASK_ID}]}" #select the corresponding jobarray part from the list of tasks to do 


base_dir=/scratch/connectome/study_group/VAE_ADHD/junbeom_weights
#weight_list=( None ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 
weight_list=( None ${base_dir}/UKBsim64a.pth ${base_dir}/UKByAa64a.pth ) 
#${base_dir}/UKBsim32a.pth ${base_dir}/UKByAa32a.pth  #only select the few


#hyperparameter tuning list
#batch_list=( 4 8 16 32 )
lr_list=( 1e-2 1e-3 1e-4 1e-5 ) #default : 1e-4, but found out that that is too small for freeze
wd_list=( 5e-4 5e-5 5e-6 ) #default : 5e-5


cd .. #move to directory above and execute stuff 

#use ; wait!! https://unix.stackexchange.com/questions/541311/bash-wait-for-all-subprocesses-of-script


#case1
batch=8
#set train_num
#train_num_list=( 10 20 40 80 100 200 )
train_num_list=( 10 80 200 ) #train_num_list=( 10 80 200 ) #only test a few

for weight in "${weight_list[@]}"
do 
for num in "${train_num_list[@]}"
do
for lr in "${lr_list[@]}"
do 
for wd in "${wd_list[@]}"
do 
#echo $num ; echo $weight ; echo $task
#do 5 times, so that we can average the results

##freeze
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait

##tune_all (두개씩 묶어서 하기)
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait

done 
done
done
done

batch=16
train_num_list=( 80 200 ) #10은 없앰, as overlapping

for weight in "${weight_list[@]}"
do 
for num in "${train_num_list[@]}"
do
for lr in "${lr_list[@]}"
do 
for wd in "${wd_list[@]}"
do 
#echo $num ; echo $weight ; echo $task
#do 5 times, so that we can average the results

##freeze
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait

##tune_all (두개씩 묶어서 하기)
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait

done 
done
done
done


batch=32
train_num_list=( 80 200 ) #10은 없앰, as overlapping

for weight in "${weight_list[@]}"
do 
for num in "${train_num_list[@]}"
do
for lr in "${lr_list[@]}"
do 
for wd in "${wd_list[@]}"
do 
#echo $num ; echo $weight ; echo $task
#do 5 times, so that we can average the results

##freeze
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd &\
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify balan --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait

##tune_all (두개씩 묶어서 하기)
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait
python3 NEW_main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch --weight_decay $wd ; wait

done 
done
done
done



##batch 8이면 위에서처럼 freeze, tune_alld을 6개, 3개씩 번갈아면서 하도록 하는게 좋은듯하다!!! (램의 50%를 넘어가게 되면  python script가 properly exit을 하기전에 다음 python script가 들어와서 GPU OOM 뜸)