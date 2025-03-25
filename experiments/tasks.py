import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

class Task:
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        self.setup_data()
        
    def setup_data(self):
        raise NotImplementedError
        
    def get_batch(self):
        # Get next batch, create new iterator if we've exhausted the current one
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        
        # Move batch to device
        inputs, targets = batch
        return inputs.to(self.config.device), targets.to(self.config.device)
    def loss(self, outputs, targets):
        return NotImplementedError
    
class QuadraticTask(Task):
    # a simple quadratic task
    def __init__(self, config, train=True):
        super().__init__(config, train)
        self.loss_fn = nn.MSELoss()
    def setup_data(self):
        # create a grid of points in the range [-2, 2] x [-1, 3]
        num_samples = 10000 if self.train else 1000    
        x = torch.linspace(-2, 2, num_samples)
        y = torch.linspace(-1, 3, num_samples)
        X, Y = torch.meshgrid(x, y)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        # put them together
        self.inputs = torch.cat([X, Y], 1)
        # compute the quadratic function
        self.targets = X**2 + Y**2
        self.dataset = torch.utils.data.TensorDataset(self.inputs, self.targets)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.inner_batch_size, shuffle=True)
        self.data_iter = iter(self.dataloader)
    def loss(self, outputs, targets):
        loss = F.mse_loss(outputs, targets)
        return loss
class RosenbrockTask(Task):
    def _rosenbrock(self, x):
        '''Compute the Rosenbrock function'''
        return 100 * (x[:, 1] - x[:, 0]**2)**2 + (1 - x[:, 0])**2
    def __init__(self, config, train=True):
        super().__init__(config, train)
        self.loss_fn = nn.MSELoss()
    def setup_data(self):
        # create a grid of points in the range [-2, 2] x [-1, 3]
        num_samples = 10000 if self.train else 1000
        x = torch.linspace(-2, 2, num_samples)
        y = torch.linspace(-1, 3, num_samples)
        X, Y = torch.meshgrid(x, y)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        # put them together
        self.inputs = torch.cat([X, Y], 1)
        # compute the Rosenbrock function
        self.targets = self._rosenbrock(self.inputs).float().to(self.config.device)
        self.dataset = torch.utils.data.TensorDataset(self.inputs, self.targets)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.inner_batch_size, shuffle=True)
        self.data_iter = iter(self.dataloader)
    def loss(self, outputs, targets):
        loss = F.mse_loss(outputs, targets)
        return loss
        

class MNISTTask(Task):
    def __init__(self, config, train=True):
        super().__init__(config, train)
        self.loss_fn = F.cross_entropy
    def setup_data(self):
        # Define transforms: resize to 16x16 and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((self.config.downsample_size, self.config.downsample_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset
        dataset = torchvision.datasets.MNIST(
            root='./.data', 
            train=self.train, 
            download=True, 
            transform=transform
        )
        
        # Create data loader
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.config.inner_batch_size,
            shuffle=True, 
            num_workers=2
        )
        
        # Create iterator to cycle through batches
        self.data_iter = iter(self.dataloader)
    

    
    def loss(self, outputs, targets):
        loss = F.cross_entropy(outputs, targets)
        return loss
    
if __name__ == '__main__':
    from config import Config
    config = Config()
    task = RosenbrockTask(config)
    inputs, targets = task.get_batch()
    print(inputs.shape, targets.shape)
    task = MNISTTask(config)
    inputs, targets = task.get_batch()
    print(inputs.shape, targets.shape)
