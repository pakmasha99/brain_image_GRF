import numpy as np
from monai.data import ImageDataset
from monai.transforms import *

class Transform:
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

