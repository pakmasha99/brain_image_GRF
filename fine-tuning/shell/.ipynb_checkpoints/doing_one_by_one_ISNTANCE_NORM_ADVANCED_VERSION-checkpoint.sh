#!/bin/bash

#SBATCH --job-name relUKB_20_inst_BN_wo_cpu_None_hyper_tune_pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH -p voltadebug
#SBATCH -t 04:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --array=0-2 #upto (number of tasks) -1  there are 


echo "HI"
module load python
source activate VAE_3DCNN_older_MONAI

##밑에 cpu mem 제한두면 오히려 느려지기는 함
#SBATCH --cpus-per-task=5 #같이 하는게 훨씬 빠름
#SBATCH --mem-per-cpu=5GB

#for i in `ls`; do echo $i ; ls -R $i | grep "eval_stats.txt" ; done
#위에걸로 찾고 하기 
###만약 BT하려면 input_option을 바꿔야함!

#직접 정해주기
base_dir=/sdcc/u/dyhan316/misc_VAE/junbeom_weights
task=ADNI_sex #( ADNI_sex ADNI_age ) #task_list=( ADNI_ALZ_ADCN ADNI_ALZ_ADMCI ADNI_ALZ_CNMCI ADNI_sex ADNI_age )

weight=None #${base_dir}/UKByAa64a.pth #( None ${base_dir}/UKByAa64a.pth ) #weight_list=( None ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 


# rm -r batch_8-lr_1.0-wd_1e-06-tf_cutout  batch_8-lr_0.1-wd_1e-08-tf_cutout batch_8-lr_10.0-wd_1e-05-tf_cutout batch_8-lr_0.1-wd_0.001-tf_cutout batch_16-lr_1e-06-wd_5e-07-tf_cutout
#rm -r batch_16-lr_1.0-wd_1e-06-tf_cutout  batch_16-lr_10.0-wd_1e-05-tf_cutout batch_16-lr_0.1-wd_1e-08-tf_cutout  batch_16-lr_0.1-wd_0.001-tf_cutout


#1e-8 3e0
#8e-7 1e0
#1e-3 1e1
#3e-9 1e-1




if [ ${SLURM_ARRAY_TASK_ID} == 0 ] 
then 
    num=20
    wd=4e-4
    lr=4e-2
elif [ ${SLURM_ARRAY_TASK_ID} == 1 ]
then 
    num=20
    wd=4e-9
    lr=1e0
elif [ ${SLURM_ARRAY_TASK_ID} == 2 ]
then
    num=20
    wd=4e-9
    lr=3e-1
#elif [ ${SLURM_ARRAY_TASK_ID} == 3 ]
#then
#    num=20
#    wd=8e-4
#    lr=2e-1
fi


cd .. #move to directory above and execute stuff 

#use ; wait!! https://unix.stackexchange.com/questions/541311/bash-wait-for-all-subprocesses-of-script
##trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/) (and https://www.cyberciti.biz/faq/bash-iterate-array/ )

##freeze

#for loop등등 하던지 하려고했는데 and 문이 잘안되서 이렇게함!
batch_1=8
batch_2=16

echo $batch_1
echo $lr
echo $wd



#wd(y) lr(x) 
#5e-3 5e0
#5e-4 5e-2
#1e-9 1e0
#5e-9 1e1


#wd_list=( 5e-3 5e-4 1e-9 5e-9 )
#
#for wd in "${wd_list[@]}"
#do
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN ; wait
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_2 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 1 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_2 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 2 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_2 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 3 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_2 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 4 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_2 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 5 --task $task --input_option yAware --learning_rate $lr --batch_size $batch_2 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN ; wait 
#done




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