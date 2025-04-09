from experiments.gmn_dev.optimizees import MLP
from experiments.gmn_dev.tasks import QuadraticTask, MNISTTask
from experiments.gmn_dev.config import  Config
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
import uuid

config = Config()
train_task = QuadraticTask(
    config,
    train=True
)
test_task = QuadraticTask(
    config,
    train=False
)

train_loader = train_task.dataloader
test_loader = test_task.dataloader


mlp_base = MLP(
    input_size=2, hidden_sizes=[1], output_size=1
).to(config.device)
optimizers = [optim.Adam(mlp_base.parameters(), lr=0.01),
             optim.SGD(mlp_base.parameters(), lr=0.01),
             optim.RMSprop(mlp_base.parameters(), lr=0.01),
            ]

loss_fn = train_task.loss_fn
SAVE_PATH = './experiments/gmn_dev/optimizer_trajectories'

def train(model, optimizer, train_loader, test_loader, num_epochs=15):
    print(f"\nTraining with {optimizer.__class__.__name__} optimizer")
    checkpoints = []
    test_losses = []
    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            # if i % 100 == 0:
            #     print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}, Loss: {loss.item():.4f}")
        checkpoints.append(deepcopy(model).to('cpu').state_dict())
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(config.device), y.to(config.device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}")
    return checkpoints, test_losses
def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(config.device), y.to(config.device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss

def init_weights(model):
    weight_init_fns = [
        nn.init.xavier_uniform_,
        nn.init.xavier_normal_,
        nn.init.kaiming_uniform_,
        nn.init.kaiming_normal_,
        nn.init.orthogonal_,
        nn.init.uniform_,
        nn.init.normal_,
        ]
    
    weight_init_fn = random.choice(weight_init_fns)
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            weight_init_fn(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
def create_optimizers(model):
    lr_range = [1e-4, 1e-1]
    
    # randomly choose a learning rate for each optimizer
    optimizers = [
        optim.Adam(model.parameters(), lr=random.uniform(lr_range[0], lr_range[1])),
        optim.SGD(model.parameters(), lr=random.uniform(lr_range[0], lr_range[1]), momentum=random.uniform(0.5, 0.9)),
        optim.RMSprop(model.parameters(), random.uniform(lr_range[0], lr_range[1]), momentum=random.uniform(0.5, 0.9)),
    ]

    return optimizers

def main():
    for i in range(1000):
        mlp_base.apply(init_weights)
        original_model = deepcopy(mlp_base.state_dict())
        run_id = str(uuid.uuid4())

        optimizers = create_optimizers(mlp_base)
        run_results = {}
        for optimizer in optimizers:
            # Reset the model to its original state before training with a new optimizer
            mlp_base.load_state_dict(original_model)
            
            # Train the model with the current optimizer
            checkpoints, test_losses = train(mlp_base, optimizer, train_loader, test_loader)

            curr_result = {
                'optimizer': optimizer.__class__.__name__,
                'checkpoints': checkpoints,
                'test_losses': test_losses
            }
            
            run_results[optimizer.__class__.__name__] = curr_result
        # save run_results
        torch.save(run_results, f"{SAVE_PATH}/run_{run_id}.pt")
        print(f"Run {run_id} completed and saved.")

if __name__ == "__main__":
    main()