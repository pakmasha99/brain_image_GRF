#!/bin/bash

#SBATCH --job-name evalNT80_hyper_tune_pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH -t 24:00:00 #volta can only do four hours for voltadebug...
#SBATCH -N 1
#SBATCH --nodelist=node3
#SBATCH --gres=gpu:1 #how many gpus each job array should have 
#SBATCH --ntasks=1 #여기서부터는 내가 추가
#SBATCH -o ./shell_output/output_%A.output
#SBATCH -e ./shell_output/error_%A.error
#SBATCH --cpus-per-task=12 #같이 하는게 훨씬 빠름(?)(test해보기.. .전에 넣은것이랑 비교해서)
#SBATCH --mem-per-cpu=4GB


##강제로 기다리기
sleep_list=( 1 30 60 90 120 ) #( 1e-3 1e-4 1e-5 ) #default : 1e-4, but found out that that is too small for freeze
sleep_time="${sleep_list[${SLURM_ARRAY_TASK_ID}]}"

sleep $sleep_time #sleep for this much time 

###만약 BT하려면 input_option을 바꿔야함!

#직접 정해주기
base_dir=/scratch/connectome/study_group/VAE_ADHD/junbeom_weights
task=ABCD_sex #( ADNI_sex ADNI_age ) #task_list=( ADNI_ALZ_ADCN ADNI_ALZ_ADMCI ADNI_ALZ_CNMCI ADNI_sex ADNI_age )

weight=${base_dir}/UKByAa64a.pth #( None ${base_dir}/UKByAa64a.pth ) #weight_list=( None ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 

num=7265 #80


cd ../.. #move to directory above and execute stuff 

##freeze

#for loop등등 하던지 하려고했는데 and 문이 잘안되서 이렇게함!
batch_1=64 #if doing 20  
#batch_2=16

#if batch size : 64, can only do one! 
#batch size :64, can do ONE TUNEALL AND ONE FREEZE (in lab)
sleep 1 ; python3 main_optuna_fix_6.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify balan_iter_strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path XXXX --binary_class True --run_where lab --eval True

#python3 main_optuna_fix_4.py --pretrained_path $weight --mode finetuning --train_num $num --layer_control tune_all --stratify strat --random_seed 0 --task $task --input_option yAware --batch_size $batch_1 --save_path ./finetune_TEST_remove_rn/null/ --binary_class True --run_where lab --eval True


#data도 fp16으로?
#AMP는 해볼만한 가치가 있을 것 

#sex 를 쓴다면, 함수식을 보고 그전에 다른 논문하고 비교해서 얼마나다른지 보기 (XOR algorithm기반으로 했는데, 
#hyperparamter tuning => 

#intelligence를 쓴다면 
#input 
#hyperpameter tuign만 

#move to hplfcs
#yhan316@node3:/scratch/connectome/3DCNN/data/2.UKB$ ls
#.sMRI_fs_cropped  2.demo_qc  subject_list.txt  UKB_data.py  UKB_data.sh


#ABCD age prediction 해보기
#ADNI 에서 age prediction


#ABCD ADHD => slightly above random보다 더 좋다면 => 좋다 
#ABCD AGE vs 
#순정 transfer 와 비교 ()
#worker수 20명해도된다
#막 써도됝다 => 눈칫껏
