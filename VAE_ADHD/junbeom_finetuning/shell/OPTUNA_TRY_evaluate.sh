#!/bin/bash

#SBATCH --job-name eval_freeze  #job name을 다르게 하기 위해서
#SBATCH -t 04:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH -p voltadebug
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --cpus-per-task=5 #같이 하는게 훨씬 빠름(?)(test해보기.. .전에 넣은것이랑 비교해서)
#SBATCH --mem-per-cpu=4GB


cd .. #move to directory above and execute stuff 

module load python
source activate VAE_3DCNN_older_MONAI




#12cpus, 2 workers1:52 8 => 1:56 32 => 2:01 40
#38 17 => 40 29 => 41 35 => 43 43 

#2cpus, 1 worker : 2:08 0 => 2:14 24 
#2cpus 6 worker : 2:24 12(~15) => 2:26 24 => 32c 54 =>  


#######
base_dir=/sdcc/u/dyhan316/misc_VAE/junbeom_weights
#/scratch/connectome/study_group/VAE_ADHD/junbeom_weights/UKByAa64a.pth

#python3 main_optuna_fix_2.py --pretrained_path None --mode finetuning --train_num 100 --layer_control freeze --stratify strat --random_seed 0 --task ABCD_sex --input_option yAware  --save_path finetune_test_same_seed_EVAL --binary_class True --batch_size 64 --eval_mode True --learning_rate 5.504698217767698e-05 --weight_decay 0.000599740134672065 --BN inst 
#
#
#python3 main_optuna_fix_2.py --pretrained_path ${base_dir}/UKByAa64a.pth --mode finetuning --train_num 100 --layer_control freeze --stratify strat --random_seed 0 --task ABCD_sex --input_option yAware  --save_path finetune_test_same_seed_EVAL --binary_class True --batch_size 64 --eval_mode True --learning_rate 0.32903134078397495 --weight_decay 2.671232775564691e-09 --BN inst

python3 main_optuna_fix_2.py --pretrained_path None --mode finetuning --train_num 100 --layer_control tune_all --stratify strat --random_seed 0 --task ABCD_sex --input_option yAware  --save_path finetune_test_same_seed_EVAL --binary_class True --batch_size  --eval_mode True --learning_rate 0.017484892311713827 --weight_decay 4.8567180166065447e-08 --BN none 


python3 main_optuna_fix_2.py --pretrained_path ${base_dir}/UKByAa64a.pth --mode finetuning --train_num 100 --layer_control tune_all --stratify strat --random_seed 0 --task ABCD_sex --input_option yAware  --save_path finetune_test_same_seed_EVAL --binary_class True --batch_size 64 --eval_mode True --learning_rate 1.050531802303288e-05 --weight_decay 1.6467213137570964e-09 --BN inst





#ADNI, all 64 
#print("run the things below tmr!! wanna see the results :)")
#Na : 0.0001901345087671811 6.958007026697329e-05 False inst
#Ua : 6.426425125067863e-06 2.2611770778719344e-09 False inst
#Nf : 0.25463017943043054 0.23019943315376581 False inst
#Uf : 0.7959121369946751 2.095952457066269e-05 True none