import torch
from tasks import Task, RosenbrockTask, MNISTTask, QuadraticTask
class Config:
    def __init__(self):
        self.seed = 11
        # Meta-optimizer settings
        self.meta_lr = 0.001
        self.meta_batch_size = 4
        self.meta_iterations = 10000
        
        # Inner optimization settings
        self.unroll_steps = 50
        self.eval_unroll_steps = 50
        self.truncated_backprop_steps = 15
        
        # Task settings
        self.task = QuadraticTask
        self.task_kwargs = {}
        # self.task = MNISTTask
        # self.task_kwargs = {}
        self.inner_batch_size = 128
        self.downsample_size = 16
        self.hidden_sizes = [32]
        
        # Logging and saving
        self.eval_interval = 100
        self.save_interval = 500
        self.log_interval = 10
        self.save_dir = "learned_optimizer_checkpoints"
        
        # LSTM optimizer settings
        self.hidden_size = 40
        
        # Adam initialization settings
        self.mimic_adam_epochs = 0  # Number of epochs to mimic Adam
        self.adam_mimic_lr = 0.001   # Adam learning rate to mimic
        self.mimic_loss_weight = 1.0 # Weight for mimic loss
        
        # Device settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")