#!/bin/bash

#SBATCH --job-name 100tu_all_inst_BN_wo_cpu_None_hyper_tune_pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH -t 48:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --array=0-2 #upto (number of tasks) -1  there are 
#SBATCH --cpus-per-task=12 #같이 하는게 훨씬 빠름
#SBATCH --mem-per-cpu=5GB

##100 : tuneall in lab, freeze in SDCC

#for i in `ls`; do echo $i ; ls -R $i | grep "eval_stats.txt" ; done
#위에걸로 찾고 하기 
###만약 BT하려면 input_option을 바꿔야함!

#직접 정해주기
base_dir=/scratch/connectome/dyhan316/VAE_ADHD/barlowtwins/pretrain_results
task=ABCD_sex #ABCD_sex #ADNI_sex


#task가 ABCD일때는 weight None으로도 해보기!
weight=None #${base_dir}/ABCDbt128a102.pth  #이거는 None돌리지 말기!! (when doing (80 x 80 x 80으로돌린 것만드로 일단 하기.. 만약 BT가 그것보다 좋게 나오면 BT사이즈에 맞춰서 다시 돌려보고)


# rm -r batch_8-lr_1.0-wd_1e-06-tf_cutout  batch_8-lr_0.1-wd_1e-08-tf_cutout batch_8-lr_10.0-wd_1e-05-tf_cutout batch_8-lr_0.1-wd_0.001-tf_cutout batch_16-lr_1e-06-wd_5e-07-tf_cutout


#1e-8 3e0
#8e-7 1e0
#1e-3 1e1
#3e-9 1e-1



num=100
#if [ ${SLURM_ARRAY_TASK_ID} == 0 ]
#then 
#    #wd=1e-5
#    lr=5e-1 #don't do 1e0, it diverges
if [ ${SLURM_ARRAY_TASK_ID} == 0 ]
then
    #wd=1e-5
    lr=5e-4 #if lr 1e0 diverges
elif [ ${SLURM_ARRAY_TASK_ID} == 1 ]
then
    #wd=1e-2
    lr=5e-5
elif [ ${SLURM_ARRAY_TASK_ID} == 2 ]
then
    #wd=1e-7
    lr=5e-6
fi


cd ../.. #move to directory above and execute stuff 

#use ; wait!! https://unix.stackexchange.com/questions/541311/bash-wait-for-all-subprocesses-of-script
##trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/) (and https://www.cyberciti.biz/faq/bash-iterate-array/ )

##freeze

#for loop등등 하던지 하려고했는데 and 문이 잘안되서 이렇게함!
batch_1=16
#batch_2=16

#echo $batch_1
#echo $lr
#echo $wd



#wd(y) lr(x) 
#5e-3 5e0
#5e-4 5e-2
#1e-9 1e0
#5e-9 1e1


wd_list=( 5e-2 5e-3 ) #( 1e-2 1e-3 1e-4 1e-5 1e-6 )

for wd in "${wd_list[@]}"
do

##BT여서 input_option을 바꿔야함! 
#batch size 하나로 고정해놓고, 8개 해서 평균내기 (not 6)
#since BT tuneall needs lots of RAM, only two per run can be used
#so only do 3 instead of 9 (only one per thing) 
#do only one tuneall and one freeze per thing (not that much memory is avail)

#ONLY RUN TWO!! (not even 3)
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 0 --task $task --input_option BT_org --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN --run_where lab &\
python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 0 --task $task --input_option BT_org --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN --run_where lab ; wait

#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 1 --task $task --input_option BT_org --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN --run_where lab &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 1 --task $task --input_option BT_org --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN --run_where lab ; wait

#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 2 --task $task --input_option BT_org --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN --run_where lab &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control freeze --stratify strat --random_seed 2 --task $task --input_option BT_org --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN --run_where lab ; wait

#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 6 --task $task --input_option BT_org --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN --run_where lab &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 7 --task $task --input_option BT_org --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN --run_where lab &\
#python3 main.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 8 --task $task --input_option BT_org --learning_rate $lr --batch_size $batch_1 --weight_decay $wd --BN inst --save_path finetune_results_InstanceBN --run_where lab ; wait

done 
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
