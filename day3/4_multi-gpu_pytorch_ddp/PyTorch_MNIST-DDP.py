import os
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

batch_size = 128
epochs = 5

use_gpu = True
dataset_loc = './mnist_data'

rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "6108"

if rank == 0:
    print(torch.__version__)

print(local_rank, rank, world_size)
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

transform = transforms.Compose([
    transforms.ToTensor() # convert and scale
])

train_dataset = datasets.MNIST(dataset_loc,
                               download=True,
                               train=True,
                               transform=transform)
test_dataset = datasets.MNIST(dataset_loc,
                              download=True,
                              train=False,
                              transform=transform)

train_sampler = DistributedSampler(train_dataset,
                                   num_replicas=world_size, # number of all GPUs
                                   rank=rank,               # (global) ID of GPU
                                   shuffle=True,
                                   seed=42)
test_sampler = DistributedSampler(test_dataset,
                                 num_replicas=world_size, # number of all GPUs
                                 rank=rank,               # (global) ID of GPU
                                 shuffle=False,
                                 seed=42)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=False, # sampler does it
                          num_workers=4,
                          sampler=train_sampler,
                          pin_memory=True)
val_loader =   DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4,
                          sampler=test_sampler,
                          pin_memory=True)

def create_model():
    model = nn.Sequential(
        nn.Linear(28*28, 128),  # Input: 28x28(x1) pixels
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10, bias=False)  # Output: 10 classes
    )
    return model

model = create_model()

if use_gpu:
    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)

model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()

# Main training loop
for i in range(epochs):
    model.train()
    #train_loader.sampler.set_epoch(i)

    # Training steps per epoch
    epoch_loss = 0
    pbar = tqdm(train_loader)
    for x, y in pbar:
        if use_gpu:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        x = x.view(x.shape[0], -1) # flatten
        optimizer.zero_grad()
        y_hat = model(x)
        batch_loss = loss(y_hat, y)
        batch_loss.backward()
        optimizer.step()
        batch_loss_scalar = batch_loss.item()
        epoch_loss += batch_loss_scalar / x.shape[0]
        pbar.set_description(f'training batch_loss={batch_loss_scalar:.4f}')

    # Run validation at the end of each epoch
    with torch.no_grad():
        model.eval()
        val_loss = 0
        pbar = tqdm(val_loader)
        for x, y in pbar:
            if use_gpu:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            x = x.view(x.shape[0], -1) # flatten
            y_hat = model(x)
            batch_loss = loss(y_hat, y)
            batch_loss_scalar = batch_loss.item()

            val_loss += batch_loss_scalar / x.shape[0]
            pbar.set_description(f'validation batch_loss={batch_loss_scalar:.4f}')

    print(f"Epoch={i}, train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

if rank == 0:
    torch.save(model.state_dict(), 'model.pt')
