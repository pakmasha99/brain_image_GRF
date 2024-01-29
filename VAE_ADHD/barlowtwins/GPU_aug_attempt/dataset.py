##TRYING TO CREATE NEW DATASET
import numpy as np
import monai
from monai.data import ImageDataset
from monai.transforms import ScaleIntensity, ResizeWithPadOrCrop, NormalizeIntensity, ToTensor, EnsureChannelFirst, ToNumpy
from pathlib import Path
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import from_numpy
from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip
import torch

class basic_transform():
    """
    basic transform for creating the dataset in the first place 
    (MNI로 fixed input size 면 not resize하고 등등)
    
    MNI : if true, does not apply ResizeWithPadOrCrop, but if false, applies the spatial size that was given by **kwargs
    """
    def __init__(self, MNI = True, **kwargs):
        transform_list = [EnsureChannelFirst()]  #removed scale intensity
                    
        #add padding/cropping if not MNI (i.e. shape varies)
        shape_no_channel = kwargs['shape'][1:] #shape without the channel dimension, as that should be used when doing Resizewithpadorcrop
        transform_list.append(ResizeWithPadOrCrop(spatial_size =  shape_no_channel, method = "symmetric", mode = "edge")) if MNI == False else None

        #adding the normalize intensity To Tensor things 
        transform_list = transform_list + [NormalizeIntensity(), ToNumpy()] 
            #device = "cuda:0"이렇게 하면 CUDA out of memory 뜸! (because, have to store too many things in VRAM
            ##ToNumpy instead of tensor because transform yAware is implemnetned in numpy
        
        self.transform = monai.transforms.Compose(transform_list)        

    def __call__(self, x):
        return self.transform(x)          

##changed function UKB_T1 to MRI_dataset
class MRI_dataset(Dataset):
    """
    split = 'train' or 'test'
    split_prop = '0~1'사이, how much to take as training for splitting
    transform = transform to be performed
    shape = shape of the image (MUST INCLUDE CHANNEL)
        * if MNI==True, does not do cropping or anything
        * if MNI==False, apply ResizeWithPadOrCrop
    MNI = fixed shape or not
    """
    
    def __init__(self,data_path, split, split_prop, transform, shape, MNI= True, gpu = 0): #add gpu as input
        assert len(shape) == 4, "shape must be given WITH the channel too! (C, H, W, D)"
        self.device = f"cuda:{gpu}"
        self.shape = shape #make it into channel
        self.split = split #split여부 tracking하기 
        
        
        ##getting list of img paths 
        imgs = sorted([f for f in Path(data_path).iterdir() if f.suffix != ".txt"]) #txt가 들어있는 경우가 있어서 제외
        
        ##GETTING LABELS : NOT IMPLEMENTED YET, SO ZERO... BUT FOLLOWING yAware's way 
        lbls = np.zeros(len(imgs))
        lbls = lbls.astype(int)
        
        ##TRAIN TEST SPLIT
        imgs = imgs[:int(len(imgs)*0.75)] if split == "train" else imgs[int(len(imgs)*0.75):] if split =="test" else print("wrong")

        self.dataset = ImageDataset(image_files=imgs, labels = lbls, transform=basic_transform(MNI = MNI,shape=self.shape ))
                
        if self.split == 'train':
            self.transform = transform(shape = self.shape) #the transform to use 
            self.transform_prime = transform(shape = self.shape) 

    #def collate_fn():#see later (dataloader할때 batch가 이상하면 이쪽을 고쳐야할수도)
    
    def __getitem__(self,idx):
        sub_data = self.dataset[idx]
        sub_img, sub_label = self.dataset[idx] #해당 idx subject의 img뽑기
        
        if self.split == 'train':
            """below : major revision, so check again (copy 안해도?)"""            
            y1 = self.transform(from_numpy(sub_img).float().to(torch.cuda.current_device())) #load the dataloader worker to the correct gpu (each gpu does its own augmentation)
            y2 = self.transform_prime(from_numpy(sub_img).float().to(torch.cuda.current_device()))
            #y1 = from_numpy(self.transform(sub_img).copy()).float() #change from numpy to tensor, minding the strider errors and dtype errors
            #        #copy done to avoid Torch.from_numpy not support negative strides errors
            #        #.float to avoid dtype mismatch https://stackoverflow.com/questions/44717100/pytorch-convert-floattensor-into-doubletensor
            #y2 = from_numpy(self.transform_prime(sub_img).copy()).float()
            #print("=========success, note that Crop in augmenrtations.py is still done on cpu.. couldn't find version that deos 3D wihtout spending lots of time", type(y1), type(y2)) #돌려보니 여기까지는 안오는 걸로 봐서, y1저기서 에러가 바로 뜨는 듯 (앵 근데 return은 나중에 하잖아...)
            #y1 = y1.to('cpu') #이렇게 다시 cpu로 옮기려고 해도 out of memory 
            #y2 = y2.to('cpu')
            """===================="""
            return (y1, y2), sub_label
        
        elif self.split == "test":
            return from_numpy(sub_img.copy()).float(), sub_label
    
    #define __len__ because needed when loading into dataloader
    def __len__(self):
        return len(self.dataset)
    
###this is done as a basic transform that is performed before creating the dataset
#즉, dataset creation using ImageDataset에서는 basic_transform을 거치고, 
#유일하게 다른 것인 어떤 "transform"을 받았냐는 것 (after creating Image Dataset)


#===========yAware augmentations============#
class transform_yAware_all():
    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.transforms = Transformer()
        
        ##register the transformations
        self.transforms.register(Flip(), probability=0.5)
        self.transforms.register(Blur(sigma=(0.1, 1)), probability=0.5)
        self.transforms.register(Noise(sigma=(0.1, 1)), probability=0.5)
        self.transforms.register(Cutout(patch_size=np.ceil(np.array(self.shape)/4)), probability=0.5)
        self.transforms.register(Crop(np.ceil(0.75*np.array(self.shape)), "random", resize=True), probability=0.5)
    
    def __call__(self, x):
        return self.transforms(x) #so far, numpy array
    
class transform_yAware_crop():
    def __init__(self, shape, **kwargs):
        self.transforms = Transformer()
        
        ##register the transformations
        self.transforms.register(Crop(np.ceil(0.75*np.array(shape)), "random", resize=True), probability=0.5)
    
    def __call__(self, x):
        return self.transforms(x)

class transform_yAware_cutout():
    def __init__(self, shape, **kwargs):
        self.transforms = Transformer()
        
        ##register the transformations
        self.transforms.register(Cutout(patch_size=np.ceil(np.array(shape)/4)), probability=1)

    def __call__(self, x):
        return self.transforms(x)


