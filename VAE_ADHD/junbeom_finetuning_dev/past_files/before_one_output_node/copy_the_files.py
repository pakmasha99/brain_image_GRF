import os
import shutil
import pandas as pd 
import pdb
import numpy as np 

sub_list = pd.read_csv("/scratch/connectome/dyhan316/VAE_ADHD/junbeom_finetuning/csv/fsdat_baseline.csv")
file_dir = "/storage/bigdata/ADNI/adni_registration_all"#/storage/bigdata/ADNI/adni_registration_all

sub_list = sub_list["File_name"].values
#pdb.set_trace()
for i in sub_list:
    print(i)
    shutil.copytree(os.path.join(file_dir, i), f'/scratch/connectome/study_group/VAE_ADHD/data/{i}')
    #break

#pdb.set_trace()
