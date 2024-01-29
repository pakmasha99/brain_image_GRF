import numpy as np
import monai
import pandas as pd
import os
from monai.data import ImageDataset
from monai.transforms import *
from pathlib import Path

##define dataset (through kinda wrapping over the ImageDataset)
def UKB_T1_sexclass(data_path, split, transform, **kwargs):
    """주의 : train/test split rate : 0.75로 preset (희환쌤 코드 처럼 분리할 수도 있을 것 같으나, 이것은 나중에 하자 )"""
    
    #list imgs list inside the directory, with suffixes ending in .gz (list of nii.gz files) ==> sorted 
    imgs = sorted([f for f in Path(data_path).iterdir() if f.suffix == '.gz']) # f.suffixes로 하면 nii.gz까지 다 있으나, .crop도 suffix로 인정되어 복잡해짐..
    
    #TRAIN_TEST SPLIT (take first 75% or last 25% of iamge list depending on split)
    imgs = imgs[:int(len(imgs)*0.75)] if split == "train" else imgs[int(len(imgs)*0.75): ] #i.e. 앞의 75% if train/뒤의 25% if test
                        #list of subject directories (PosixPath) to use
    
    
    """labels도 위의 두 줄처럼럼 어떻게 해서 하기
        #imgs와 order 이 같아야하는 것을 주의하면서! 근데 원래 CELEBA도 40짜리 label을 가져서, 우리도 label찾아서 해야할 듯!
        ordering이 맞아야한다는 것을 주의하기!
        일단은 그냥 똑같은 shape를 가진 0으로된np  array로 하자"""

    file_list = os.listdir(data_path) 
    phenotype = pd.read_csv("/scratch/connectome/mieuxmin/UKB_t1_MNI/UKB_phenotype.csv") ###이거는 위치에 따라 달라질 수 있음 
    phenotype_real = phenotype[["eid","sex"]]
    phenotype_real = phenotype_real.astype({"eid":'str'})
    lbls_list = []
    for i in file_list:
        number = i[:7]
        #index = phenotype_real.index[phenotype_real["eid"]==number]
        index2 = np.where(phenotype_real["eid"]==number)
        select_indices = list(index2)[0]
        select_df = phenotype_real.iloc[select_indices]
        a = list(select_df["sex"])[0]
        a = int(a)
        lbls_list.append(a)
    lbls = np.array(lbls_list)

    return ImageDataset(image_files=imgs, labels = lbls, transform = transform)

class Transform_yAware:
    def __init__(self):
        
        self.transform = monai.transforms.Compose([
            #normalize , flip, blur, noise, cutout, crop
            ScaleIntensity(), AddChannel(),
            RandFlip(prob = 0.5),
            RandGaussianSharpen(sigma1_x=(0.1, 1.0), sigma1_y=(0.1, 1.0), sigma1_z=(0.1, 1.0), sigma2_x=0.1, sigma2_y=0.1, sigma2_z=0.1,prob=0.5),
            ResizeWithPadOrCrop(spatial_size =  (182,218,182), method = "symmetric", mode = "constant"),
            NormalizeIntensity()
            ,ToTensor()
        ])
        
   
        #어디에 뭐가 들어있는지 확인하기 위해서, (182,20,182)로 함
        self.transform_prime = monai.transforms.Compose([
            #normalize , flip, blur, noise, cutout, crop
            ScaleIntensity(), AddChannel(),
            RandFlip(prob = 0.5),
            RandGaussianSharpen(sigma1_x=(0.1, 1.0), sigma1_y=(0.1, 1.0), sigma1_z=(0.1, 1.0), sigma2_x=0.1, sigma2_y=0.1, sigma2_z=0.1,prob=0.5),
            ResizeWithPadOrCrop(spatial_size =  (182,218,182), method = "symmetric", mode = "constant"),
            NormalizeIntensity()
            ,ToTensor()
        ])
        
         
            
    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1,y2    
    