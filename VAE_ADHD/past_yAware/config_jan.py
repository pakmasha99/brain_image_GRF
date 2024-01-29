#config 
PRETRAINING = 0
FINE_TUNING = 1

class Config:

    def __init__(self, mode):
        assert mode in {PRETRAINING, FINE_TUNING}, "Unknown mode: %i"%mode

        self.mode = mode

        if self.mode == PRETRAINING:
            self.batch_size = 32 #64
            self.nb_epochs_per_saving = 1
            self.pin_mem = True
            self.num_cpu_workers = 8
            self.nb_epochs = 100
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5
            # Hyperparameters for our y-Aware InfoNCE Loss
            self.sigma = 5 # depends on the meta-data at hand
            self.temperature = 0.1
            self.tf = "all_tf"
            self.model = "DenseNet"


            # Paths to the data
            self.data_train = "/scratch/connectome/mieuxmin/UKB_t1_MNI/npy_unzip_mni_whole.npy" #npy로 싹다 묶었더니 안돌아감
            self.label_train = "/scratch/connectome/mieuxmin/UKB_t1_MNI/npy_unzip_mni_100_sex.csv"

            self.data_val = "/scratch/connectome/mieuxmin/UKB_t1_MNI/npy_val_unzip_mni_whole.npy"
            self.label_val = "/scratch/connectome/mieuxmin/UKB_t1_MNI/npy_val_unzip_mni_100_sex.csv"


            self.input_size = (1, 182, 218, 182)
            self.label_name = "sex"

            self.checkpoint_dir = "/scratch/connectome/mieuxmin/UKB_t1_MNI/checkpoint_yAware/"
            
            

        elif self.mode == FINE_TUNING:
            ## We assume a classification task here
            self.batch_size = 32
            self.nb_epochs_per_saving = 10
            self.pin_mem = True
            self.num_cpu_workers = 1
            self.nb_epochs = 100
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5

            self.pretrained_path = "/path/to/model.pth"
            self.num_classes = 2
            self.model = "DenseNet"
