
import torch
from experiments.gmn_dev.tasks import Task, RosenbrockTask, MNISTTask, QuadraticTask
class Config:
    def __init__(self):
        self.seed = 11
        
        self.meta_lr = 0.001
        self.meta_batch_size = 4
        self.meta_iterations = 10000
        
        self.downsample_size =8
        self.inner_batch_size = 128
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 1
    def load(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.key = value
                self.__dict__[key] = value