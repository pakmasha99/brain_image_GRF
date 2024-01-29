from cmath import nan
import os 

PRETRAINING = 0
FINE_TUNING = 1

class Config:

    def __init__(self, mode, task,run_where):
        assert mode in {PRETRAINING, FINE_TUNING}, "Unknown mode: %i"%mode

        self.mode = mode
        self.run_where = run_where
        ##depending on where run_where is, adjust accordingly
        if self.run_where == "lab":
            self.DATA = "/scratch/connectome/dyhan316/CL_MRI_MISC/DATA"
            raise NotImplementedError("lab must be checked if it runs properly!")
        elif self.run_where == "KISTI":
            self.DATA = "/scratch/x2519a02/CL_MRI_MISC/DATA"
        else :
            raise NotImplementedError(f"run_where {run_where} not implemented yet")


        if self.mode == PRETRAINING:
            raise NotImplementedError("shouldn't use this as pretarining part of this code is wrong ")

        
        ##trash values (only for debugging)
        elif self.mode == FINE_TUNING:
            ## We assume a classification task here
            ##default, true for all
            self.nb_epochs_per_saving = 10
            self.pin_mem = True
            self.cuda = True
            self.task = task # for future reference

            ##training hyperparams
            self.tf = 'cutout' # ADNI #option for none? 
            self.model = 'DenseNet' # 'UNet'
            self.valid_ratio = 0.25 # valid set ratio compared to training set (i.e. if 5 fold CV, ratio should be 0.25 (becasue train : valid = 4:1)
            
            if task == "test":
                self.data = os.path.join(self.DATA, 'ADNI_past') #'./adni_t1s_baseline' # 
                self.label = './csv/fsdat_baseline.csv' # where the label file is 
                self.task_type = 'cls' # ADNI # 'cls' or 'reg' #####
                self.label_name = 'Dx.new' # ADNI # `Dx.new` #####
                self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####
                self.task_name = "AD/CN"
                self.iter_strat_label = {"binary" : ['PTGENDER'], "multiclass" : [] ,"continuous" : ["PTAGE"]} #MUST BE IN LIST FORM 
                #PTGENDER
                #or do "multiclass"
                
            elif task == "test_age":
                self.data = os.path.join(self.DATA, 'ADNI_past') #'./adni_t1s_baseline' # 
                self.label = './csv/fsdat_baseline.csv' # where the label file is 
                self.task_type = 'reg' # ADNI # 'cls' or 'reg' #####
                self.label_name = 'PTAGE' # ADNI # `Dx.new` #####
                self.num_classes = 1 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####
                self.task_name = "AGE"

            ###task 정해주기###
            elif "ADNI" in task: #i.e. if ADNI
                ### ADNI
                self.data = os.path.join(self.DATA, 'ADNI_past') #'./adni_t1s_baseline' # ADNI #ADNI dataset path 
                self.label = './csv/fsdat_baseline.csv' # where the label file is 
                self.iter_strat_label = {"binary" : ['PTGENDER'], "multiclass" : [] ,"continuous" : ["PTAGE"]} 
                if task == "ADNI_ALZ_ADCN":
                    self.task_type = 'cls' # ADNI # 'cls' or 'reg' #####
                    self.label_name = 'Dx.new' # ADNI # `Dx.new` #####
                    self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####
                    self.task_name = "AD/CN"
                    
                elif task == "ADNI_ALZ_ADMCI":
                    self.task_type = 'cls' # ADNI # 'cls' or 'reg' #####
                    self.label_name = 'Dx.new' # ADNI # `Dx.new` #####
                    self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####
                    self.task_name = "AD/MCI"
                    
                elif task == "ADNI_ALZ_CNMCI":
                    self.task_type = 'cls' # ADNI # 'cls' or 'reg' #####
                    self.label_name = 'Dx.new' # ADNI # `Dx.new` #####
                    self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####
                    self.task_name = "CN/MCI"
                ###could get more variants here###    
                elif task == "ADNI_sex":
                    self.task_type = 'cls' # ADNI # 'cls' or 'reg' #####
                    self.label_name = 'PTGENDER' # ADNI # `Dx.new` #####
                    self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####
                    self.task_name = "M/F"
                #not sure if it works, but whatever haha 
                elif task == "ADNI_age":
                    self.task_type = 'reg'
                    self.label_name = "PTAGE"
                    self.num_classes = 1
                    self.task_name = "AGE"
                    
            
            ##possible ABCD options
            ##not sure if doing it like this is a good idea!! but whatever !
            #"ABCD" + "HC" (or not) + "sex" or "ADHD" (이들의 조합으로 만들기 가능)
            elif "ABCD" in task : #if ABCD task 
                self.iter_strat_label = {"binary" : ['sex'], "multiclass" : [] ,"continuous" : ["age", "nihtbx_totalcomp_uncorrected"]} #MUST BE IN LIST FORM, 이 label들을 정한이유 : 
                #https://wandb.ai/dyhan316/selecting_ABCD_iterative_strat_label?workspace=user-dyhan316  https://teams.microsoft.com/l/message/19:1048f2a9f9ca4a5c9f0ed3cb64e236b6@thread.tacv2/1677152741957?tenantId=4a0e8042-883b-4758-9607-b068648ad6c4&groupId=834aefdc-0ca0-4c85-84e8-22352729e113&parentMessageId=1677152741957&teamName=SNU%20Connectome%20Group&channelName=Study%20with%20me%20-%20CVAE(Autism)&createdTime=1677152741957&allowXTenantAccess=false
                
                
                print("\n======\n=======reshape will be used, but since should make it so that the method depends on whether we use BT or yAware!!!!\n======\n=======")
                if "HC" in task : #i.e. if including healthy subs only
                    ###
                    raise NotImplementedError("healthy only version notimplemented yet")
                elif "talairch" in task : 
                    self.data =  os.path.join(self.DATA, "ABCD_talairch")
                    self.label = './csv/ABCD_csv/ABCD_phenotype_total_ONLY_MRI.csv'

                else : #i.e. include all subjects,
                    raise NotImplementedError("may need to look into this part again!")
                    self.data = '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
                    self.label = './csv/ABCD_csv/ABCD_phenotype_total_ONLY_MRI.csv'
                if "sex" in task : #4191 if iterstrat, 7266 if strat
                    self.task_type = 'cls'
                    self.label_name = 'sex'
                    self.num_classes = 2
                    self.task_name = "1.0/2.0"
                if "ADHD_strict" in task : #train_num = 7265 |||| is max (if we take out intelligence from the iter strat label)
                    self.iter_strat_label = {"binary" : ['sex'], "multiclass" : [] ,"continuous" : ["age"]} #removed intelligence because it took out too much samples (NAN)
                    self.label = './csv/ABCD_csv/BT_ABCD_dataset_brain_cropped.csv' #not sure if exahustive
                    self.task_type = 'cls'
                    self.label_name = 'ADHD'
                    self.num_classes = 2
                    self.task_name = "HC/ADHD"
                if "ABCD_BMI_pred_cls" in task : #?? 모름 몇명이나 될지
                    self.iter_strat_label = {"binary" : ['sex'], "multiclass" : [] ,"continuous" : ["age"]} #removed intelligence to keep as much as possible
                    self.label = './csv/ABCD_csv/sex_age_added_ABCD_phenotype_total_1years_become_overweight_10PS_stratified_partitioned_5fold.csv' 
                    self.task_type = 'cls'
                    self.label_name = 'become_overweight'
                    self.num_classes = 2
                    self.task_name = "0.0/1.0"
                if "ABCD_BMI_pred_reg" in task :
                    raise NotImplementedError("BMI_pred_reg not implemented yet")
                    
                    
            elif "UKB" in task : 
                raise NotImplementedError("UKB not done yet, also BT, yAware weights에 따라 resize method등등 다르게 해서 하기!")
                
                
            else:
                raise NotImplementedError(f"not implemented for {task}")
            

            # elif "CHA" in task : 
            #     self.data = '/storage/bigdata/CHA_bigdata/sMRI_brain'
            #     self.label = '/storage/bigdata/CHA_bigdata/metadata/CHA_sMRI_brain.csv'
            #     self.iter_strat_label = {"binary" : ['sex'], "multiclass" : [] ,"continuous" : ["age(days)"]}
                
            #     if "ASDGDD" in task : 
            #         self.task_type = 'cls'
            #         self.num_classes = 2
            #         self.label_name = "ASDvsGDD"
            #         self.task_name = "ASD/GDD"