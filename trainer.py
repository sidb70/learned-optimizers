from data.dataloader import load_dataset
from models.models import load_model
from meta_models.sequence_models import UnrolledRNN
from copy import deepcopy
import torch
import torch.nn as nn
import uuid
import os
from utils import flatten_model
checkpoint_dir = '/mnt/home/bhatta70/Documents/learned-optimizers/data/checkpoints/mnist-linear/'
training_id = uuid.uuid4()
checkpoint_dir = os.path.join(checkpoint_dir, str(training_id))
os.makedirs(os.path.join(checkpoint_dir))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_params(params):
    ''' normalize to range 0-1'''
    params = params - params.min()
    params = params / params.max()
    return params

# Load the dataset
trainset, testset = load_dataset("mnist")
# split train into train and validation
trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - 10000, 10000])
# dataloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
model = load_model("MNISTLinear").to(device)

# Train the model
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

h0, _ = flatten_model(model,grad=False)
# mask out half of the params
# num_masked = int(h0.size(0) * (3/4))
# mask = torch.randperm(h0.size(0))[:num_masked]
# h0[mask] = 0

all_val_losses = []
all_train_losses = []

num_params = sum(p.numel() for p in model.parameters())
meta_model = UnrolledRNN(input_size=num_params, hidden_size = num_params, output_size = num_params, seq_len = 10).to(device)
meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
meta_criterion = nn.MSELoss()

for epoch in range(10):
    train_losses = []
    model_params = []  
    meta_losses = []

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        train_losses.append(loss.item())
        params, grads = flatten_model(model, grad=True)

        # get gradients 
        


        params = normalize_params(params)
        grads = normalize_params(grads)
        # # randomly mask half 
        # indices = torch.randperm(params.size(0))[:num_masked]
        # params[indices] = 0
        # grads[indices] = 0

        model_params.append(params)

        # meta update
        meta_optimizer.zero_grad()
        params = torch.stack(model_params, dim=0).unsqueeze(0)
        model_grads = grads.unsqueeze(0)
        # normalize params and grads

        outputs = meta_model(h0=h0, x=params)
        # print("outputs", outputs.flatten()[:10])
        # print("model_grads", model_grads.flatten()[:10])
        loss = meta_criterion(outputs, model_grads)
        loss.backward()
        meta_losses.append(loss.item())
        # print(f"Meta Loss: {loss.item()}")
        meta_optimizer.step()



        optimizer.step()


    torch.save(deepcopy(model).to('cpu'), os.path.join(checkpoint_dir, f'{epoch}.pt'))

    train_loss = sum(train_losses) / len(train_losses)
    print(f"Epoch {epoch}, Loss: {train_loss}")
    print(f"Meta Loss: {sum(meta_losses) / len(meta_losses)}")
    # validation loss
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(valloader)
    print(f"Validation Loss: {val_loss}")
    all_val_losses.append(val_loss)
    all_train_losses.append(train_loss)
    if val_loss > train_loss:
        break

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")



