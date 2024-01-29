from pathlib import Path
import os
from glob import glob
import subprocess
import shutil

dir_to_clean = "/scratch/connectome/dyhan316/VAE_ADHD/junbeom_finetuning/finetune_results_yAware_hyperparam_tuning"
#"/scratch/connectome/dyhan316/VAE_ADHD/junbeom_finetuning/"


print(os.listdir(dir_to_clean))

seed_dirs = glob(str(Path(dir_to_clean) / "**/seed*"), recursive = True) #add in Path and str to make more robust
seed_dirs = [Path(seed_pth) for seed_pth in seed_dirs]


#count the lines for a given txt file 
line_length = [int(subprocess.run(f"cat {one_seed_dir}/eval*| wc -l" , capture_output = True, shell = True).stdout.decode()) for one_seed_dir in seed_dirs]


print(len(line_length)) #count number of lines in eval* and if less than 0, remove that tree
assert len(line_length) == len(seed_dirs), "sth wrong!!"


print(line_length)
#import pdb;pdb.set_trace()

for i,dir in enumerate(seed_dirs):
    print(dir)
    print(line_length[i])
    if line_length[i] == 0:
        shutil.rmtree(dir)
    
    
print("======DONE======")



#diff = (set(seed_dirs).difference(seed_keep_dirs))




#print(seed_dirs[0], "\n=======\n",seed_keep_dirs[0])

#seed_keep_dirs = glob(str(Path(dir_to_clean) / "**/ROC*"), recursive = True) #add in Path and str to make more robust

#print(subprocess.run(f"cat {seed_dirs[0]}/eval*" , capture_output = True, shell = True).stdout.decode())
#seed_keep_dirs = [Path(roc_pth).parent for roc_pth in seed_keep_dirs]
#print(len(seed_keep_dirs))