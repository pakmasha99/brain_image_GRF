import os 
import numpy as np 
import pandas as pd
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
import re
import json

import ast 



np.seterr(all='raise')
#https://stackoverflow.com/questions/15933741/how-do-i-catch-a-numpy-warning-like-its-an-exception-not-just-for-testing
#=========================================================================================#
#=====BELOW : about extracting statistics and etc after the `extract_json` (below)========#
#=========================================================================================#

##stats_pd가 extract_case, arrayize_df를 쓰기는 하지만, arrayize_df독자적으로 쓰여야할 때도 있을듯 (특히, mean, std가 아닌 array자체가 필요할때)(df 자체로는 안되서 이렇게라도함)

def extract_case(df, task_name, param_name):
    """
    given a df and task name and paramname, extract the relevant rows (i.e. only take the 6 seeds with the same conditions)
    """
    mask = (df["Taskname"] == task_name) & (df['param']==param_name)
    df_extract = df[mask] #extract df with all same conditions, except the seed numbers
    return df_extract



def arrayize_df(df, column_names) :
    """
    for column names, take the thing and arrayize it (i.e. shape 5x6 : 5 (n_CV) x 6 (number of seeds) then add them up as dict
    with corresponding key values being the column name
    
    output : dict for example, {'loss', 5x6 aray, 'acc', 5x6 array ...}
    """
    
    aggregate_dict = {}
    for column_name in column_names:
        arr = np.array([df[column_name +"_"+ str(i)].to_numpy() for i in  range(5)]) #5 becasue we did 5 CV
        aggregate_dict[column_name] = arr
    aggregate_dict
    return aggregate_dict


def stats_pd(df, cols_to_include, mode):
    """
    creates pd stats file that removes the original thigns and instead creates a pd dataframe (for later heatmap visiulaization)
    
    INPUT :
    * df : original dataframe
    * cols_to_include : cols to include in the stats_pd that is made 
        * (loss 값들은 여기에 추가하지 말기!)(이거는 mode에서 알아서 infer하도록 할것임)
    * mode : 'cls' or 'reg'
    
    OUTPUT : 
    * final_pd : pandas df of with `col_to_include` and the mean, std of the metric_cols (predefined below)
    """
    
    #choose which metric cols to track depending on the mode
    metric_cols = ["loss", "acc", "auroc"]  if mode == 'cls' else ["mse", "mae", "rmse", "r2"] if mode == 'reg' else print("wrong")

    #====attempt===== #add epoch to both reg and cls thigns (baseline)
    metric_cols.append('epoch')
    
    
    task_name_set, param_name_set = set(df.Taskname) , set(df.param)
    
    stats_row_list = [] #list of each combination of task, param에 대한 stats_row
    
    for task_name in task_name_set: 
        for param_name in param_name_set:
            
            extracted_df = extract_case(df, task_name, param_name) #extract a specific param task name set (i.e. same condition, jsut different seed values
            
            
            #sometimes the extracted_df does not have the shape we want (i.e. empty) => 이경우 에러날 수도 있으니, skip하기
            if 0 in extracted_df.shape :
                print(f"skipped iteration {task_name}, {param_name} because it was empty")
                continue
            
            aggregate_dict = arrayize_df(extracted_df, metric_cols) #dictionary형태로 metric들의 2d array뽑아오기(나중에 평균취함)
            
            #only include cols that we want to use  (cols to include가 들어간 이유)
            
            stats_row = extracted_df[cols_to_include].iloc[0] #take sample from the first row , final series (which will be appended to create the final_pd
            
            #print("FIX HERE 밑에파트 inner for loop인지 뭔지떄문에 에러뜸.. ")
            
            for key, value in aggregate_dict.items():
                #add the mean and std to the things
                
                #https://stackoverflow.com/questions/15933741/how-do-i-catch-a-numpy-warning-like-its-an-exception-not-just-for-testing
                try :
                    #print(task_name, param_name, key, value, np.std(value))
                    stats_row[key + "_mean"] = np.mean(value)
                    stats_row[key + "_std"] = np.nanstd(value)
                    
                except FloatingPointError as e: #numpy going to infinity는 warning으로 뜨기때문에, floatingpointerror을 잡아서 하자 
                    #import pdb; pdb.set_trace()
                    print("error occured, probably due to the std being too large that it diverges to inf")
                    print(e)
                    print('=====passing this iteration========')
                    continue
                          
            stats_row_list.append(stats_row)
    return pd.DataFrame(stats_row_list)




#================================================================================#
#=====BELOW PART OF THE CODE : about extracting json from eval_stats.txt)========#
#================================================================================#

def obj2sep(df, column, new_name): 
    """
    ### obj2sep 
    원래 [d d d ]이런식으로 리스트로 했는데, pandas가 그것을 못함
    따라서, 이것을 그냥 CV 별로 0~5로 해서 column을 새로 만들자 (기존것을 없애고)
    (from : https://stackoverflow.com/questions/46523319/pandas-column-dtype-of-array) 
    seperate panmdas dtypes being obj to a columns (seperate them)
    given dataframe and columns
    (from : https://stackoverflow.com/questions/46523319/pandas-column-dtype-of-array) 
    """
    
    removed_df = df.drop(column, axis=1) #mse빠진 원래것 넣기 
    
    if type(df[column][0]) == str:
        df_list = [eval(i) for i in df[column].tolist()]
    elif  type(df[column][0]) == list:
        df_list = df[column].tolist()
        
    add_df = pd.DataFrame(df_list, dtype=np.float32, columns = [new_name +'_' +str(i) for i in range(5)])
    return pd.concat([removed_df, add_df], axis = 1)



def extract_json(directory):
    file_list = os.listdir(directory) 
    file_name = file_list[:]

    result_df_in_list_reg = []
    result_columns_reg = []
    n_reg = 0 
    result_df_in_list_cls = []
    result_columns_cls = []
    n_cls= 0 
    
    for i in range(len(file_name)):
        print("<<<<<<<< FILE NAME is ",file_name[i],">>>>>>>>")
        add_lv1 = directory+file_name[i]
        add_1_list = os.listdir(add_lv1) 
        if "-reg" in file_name[i]:
            for j in range(len(add_1_list)):
                print("<<<<<<<< param ",add_1_list[j],">>>>>>>>")
                add_lv2 = add_lv1+'/'+add_1_list[j]
                add_2_list = os.listdir(add_lv2) 
                for m in range(len(add_2_list)):
                    print("<<<<<<<< model",add_2_list[m],">>>>>>>>")
                    add_lv3 = add_lv2+'/'+add_2_list[m]
                    add_3_list = os.listdir(add_lv3) 
                    add_3_list.sort()
                    for s in range(len(add_3_list)):
                        seed = add_3_list[s]
                        add_lv4 = add_lv3+'/'+seed+'/'
                        add_4_list = os.listdir(add_lv4)
                        # dem 만들기 
                        T = []
                        Taskname = file_name[i]+'_'+add_1_list[j]
                        T.append(Taskname)
                        P=[]
                        param = add_2_list[m]
                        P.append(param)
                        S=[]
                        S.append(seed)
                        #dem 생성 
                        dem = pd.DataFrame({'Taskname':T, 'param':P, 'seed':S})
                
                        #eval_list끌어오기 
                        address = add_lv4+'eval_stats.txt'
                        #aprint(address)
                        try : 
                            file_size = os.path.getsize(address)
                        except : 
                            print(f"{address} does not exist, so skipping this iteration on")
                            continue
                        if file_size == 0 : 
                            print(address,"is empty!!!")
                        else: 
                            eval_stats = pd.read_csv(add_lv4+'eval_stats.txt', header = None,sep = "\t")
                        # line 0처리
                            line_0 = json.loads(eval_stats.iloc[0][0])
                            input_size_val = line_0.get('input_size')
                            line_0.update(input_size = str(input_size_val))
                            line_00 = []
                            line_00.append(line_0)
                            line_0_df= pd.DataFrame(line_00)
                
                        #line_1처리
                            line_1 = json.loads(eval_stats.iloc[1][0])
                            line_11 = []
                            line_11.append(line_1)
                            line_1_df = pd.DataFrame(line_11)
    
                        # line0~1 concat  & basic_dem
                            final = pd.concat([dem,line_0_df],axis=1)
                            final = pd.concat([final,line_1_df],axis=1)
                            
                            final_col_reg = final.loc[0].tolist()
                            
                            if n_reg == 0:
                                result_columns_reg.append(final.columns.tolist())
                                n_reg += 1
                            else:
                                pass
                            
                            #print(a)
                            result_df_in_list_reg.append(final_col_reg)
                        
        elif "-cls" in file_name[i]: 
            for j in range(len(add_1_list)):
                print("<<<<<<<< param ",add_1_list[j],">>>>>>>>")
                add_lv2 = add_lv1+'/'+add_1_list[j]
                add_2_list = os.listdir(add_lv2) 
                for m in range(len(add_2_list)):
                    print("<<<<<<<< model",add_2_list[m],">>>>>>>>")
                    add_lv3 = add_lv2+'/'+add_2_list[m]
                    add_3_list = os.listdir(add_lv3) 
                    add_3_list.sort()
                    for s in range(len(add_3_list)):
                        seed = add_3_list[s]
                        add_lv4 = add_lv3+'/'+seed+'/'
                        add_4_list = os.listdir(add_lv4)
                        # dem 만들기 
                        T = []
                        Taskname = file_name[i]+'_'+add_1_list[j]
                        T.append(Taskname)
                        P=[]
                        param = add_2_list[m]
                        P.append(param)
                        S=[]
                        S.append(seed)
                        #dem 생성 
                        dem = pd.DataFrame({'Taskname':T, 'param':P, 'seed':S})
                
                        #eval_list끌어오기 
                        address = add_lv4+'eval_stats.txt'
                        
                        #print(address)
                        try : 
                            file_size = os.path.getsize(address)
                        except : 
                            print(f"{address} does not exist, so skipping this iteration on")
                            continue
                        if file_size == 0 : 
                            print(address,"is empty!!!")
                        else: 
                            eval_stats = pd.read_csv(add_lv4+'eval_stats.txt', header = None,sep = "\t")
                            # line 0처리
                            line_0 = json.loads(eval_stats.iloc[0][0])
                            input_size_val = line_0.get('input_size')
                            line_0.update(input_size = str(input_size_val))
                            line_00 = []
                            line_00.append(line_0)
                            line_0_df= pd.DataFrame(line_00)
                
                            #line_1처리
                            line_1 = json.loads(eval_stats.iloc[1][0])
                            line_11 = []
                            line_11.append(line_1)
                            line_1_df = pd.DataFrame(line_11)
                         
                            # line0~2 concat  & basic_dem
                            final = pd.concat([dem,line_0_df],axis=1)
                            final = pd.concat([final,line_1_df],axis=1)
                            
                            final_col_cls = final.loc[0].tolist()
                            
                            if n_cls == 0:
                                result_columns_cls.append(final.columns.tolist())
                                n_cls += 1
                            else:
                                pass
                        
                           # print(a)
                            result_df_in_list_cls.append(final_col_cls)
    #print("완료")   
    FINAL_cls = pd.DataFrame(result_df_in_list_cls,columns = result_columns_cls[0])
    FINAL_reg = pd.DataFrame(result_df_in_list_reg,columns = result_columns_reg[0])
    
    
    ##doing some touchup 
    FINAL_cls = obj2sep(FINAL_cls, 'loss_list', 'loss')
    FINAL_cls = obj2sep(FINAL_cls, 'acc_list', 'acc')
    FINAL_cls = obj2sep(FINAL_cls, 'mean_auroc', 'auroc')
    FINAL_cls = obj2sep(FINAL_cls, 'std_auroc', 'trash')

    FINAL_reg = obj2sep(FINAL_reg, 'mse_list ', 'mse')
    FINAL_reg = obj2sep(FINAL_reg, 'mae_list ', 'mae')
    FINAL_reg = obj2sep(FINAL_reg, 'rmse_list', 'rmse')
    FINAL_reg = obj2sep(FINAL_reg, 'r2_list', 'r2')
    
    #also add the last epoch thing for reg! (making code to implement this btw)
    FINAL_cls =  obj2sep(FINAL_cls, 'last_epoch_list', 'epoch')
    FINAL_reg = obj2sep(FINAL_reg, 'last_epoch_list', 'epoch')
    return FINAL_cls, FINAL_reg



