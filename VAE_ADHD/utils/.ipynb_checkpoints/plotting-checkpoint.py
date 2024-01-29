import json
from pathlib import Path
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy
import numpy as np 



#nonlinear heatmap using Rbf interpolation
def nonuniform_imshow(x, y, z, aspect=1, cmap=plt.cm.rainbow, vmin = None, vmax = None ):
    """
    from : https://stackoverflow.com/questions/39034797/heatmap-for-nonuniformly-spaced-data
    * x, y, z : 1d array (x,y: 좌표, z: heatmap값
    * vmin, vmax : min,max value to set for imshow (default, None)
    
    """
    
    # Create regular grid
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate missing data
    rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(xi, yi)
  
    _, ax = plt.subplots(figsize=(6, 6))
  
    #hm = ax.imshow(zi, interpolation='nearest', cmap=cmap,
    #               extent=[x.min(), x.max(), y.max(), y.min()]) 
    hm = ax.imshow(zi, interpolation='nearest', cmap=cmap,
                   extent=[x.min(), x.max(), y.max(), y.min()], vmin= vmin, vmax=vmax) 
    #ax.scatter(x, y)
    ax.scatter(x, y, c = 'black', alpha = 0.3)
    ax.set_aspect(aspect)
    
    return hm 




#=====밑 : primiarly was used for BT pretraining====#
def open_stat_txt(txt_dir):
    """
    if stats.txt is provided, use that 
    if the file name above is provided, use that
    
    returns the meta_data and df (loaded dataframe)
    """
    if os.path.isdir(txt_dir):
        txt_dir = txt_dir / "stats.txt"
    else :  #i.e. when that stats file itself is provided
        pass
    file = open(txt_dir, 'r')
    total_list=[] 
    for i, line in enumerate(file):
        try:  #json형태가 아닐떄가 있어서, 이런식으로 try로 해서 에러가 뜨면 meta_data로 저장하기 
            line_dict = json.loads(line) 
            total_list.append(line_dict)
        except :
            meta_data = line
    df = pd.DataFrame(total_list)
    return meta_data, df

def avg_epoch_wise(df):
    """
    gets the df from above and returns its epoch wise avearges
    """
    epoch_df = pd.DataFrame()
    num_epoch = df['epoch'].max()
    for i in range(num_epoch):
        a = df[df['epoch']==i].mean(0)
        a_frame = a.to_frame().T
        epoch_df = pd.concat([epoch_df, a_frame], axis = 0)
    epoch_df = epoch_df.set_index('epoch')
    return epoch_df

def std_epoch_wise(df):
    """
    gets the df from above and returns its epoch wise avearges
    """
    epoch_df = pd.DataFrame()
    num_epoch = df['epoch'].max()
    for i in range(num_epoch):
        a = df[df['epoch']==i].std(0)
        a_frame = a.to_frame().T
        epoch_df = pd.concat([epoch_df, a_frame], axis = 0)
    epoch_df = epoch_df.set_index('epoch')
    return epoch_df


def plot_error_bars(ckpt_dict, xlim,  x_label, sub_title ,yscale = 'linear', what_to_plot = "loss"):
    plot_num = len(ckpt_dict)
    fig, axs = plt.subplots(plot_num, sharex = True, sharey = True, figsize = (4,2*plot_num))
    for i, (key,path) in enumerate(ckpt_dict.items()):
        #i : iteration index, key : name, path : the path
        meta, df = open_stat_txt(path)
        epoch_average_loss = avg_epoch_wise(df)
        epoch_std_loss = std_epoch_wise(df)
        
        #axs[i].plot(df['loss'])
        #axs[i].plot(epoch_average_loss['loss'])
        axs[i].errorbar(epoch_average_loss.index, epoch_average_loss[what_to_plot], 
                       epoch_std_loss[what_to_plot])
        
        axs[i].set_xlim(0,xlim)
        axs[i].set_yscale(yscale)
        #axs[i].set_ylim(bottom = y_min)
        axs[i].set_title(key)
        
        fig.tight_layout()
        fig.suptitle(sub_title, fontweight = "bold")
        
        plt.xlabel(x_label, fontweight = "bold")
        #print(i, key, path)  
    return fig, axs
    #axs[i].set_xticks(range(0,10)) #remove later

def plot_loss(ckpt_pth, sub_count = int(0.75*7091), *args, **kwargs):
    meta, df = open_stat_txt(ckpt_pth)
    where = meta.find("print-freq")
    print_every = int(meta[where:].split(" ")[1]) #+1 because 10 도 가능하니, leave rookm
    
    where = meta.find('batch-size')+11
    batch_size = int(meta[where:].split(" ")[0])
    return plt.plot(df['step']*batch_size/sub_count, df['loss'],*args, **kwargs)