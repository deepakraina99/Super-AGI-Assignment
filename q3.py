import torch
import torch.nn as nn
import torch.optim as optim
from model2 import GPT2

# Define training module here
# 0 - single gpu training loop
# 1 - distributed data parallel
# 2 - Fully Sharded Data Parallel
train_type = 0

# Define your GPT2 model and dataset
vocab_size = 50257  # Replace with actual vocabulary size
embed_size = 768   # Replace with desired embedding size
max_len = 512      # Maximum sequence length
num_heads = 12       # Number of attention heads
ff_hidden_size = 1024  # Hidden layer size of feed-forward network
num_layers = 12      # Number of transformer layers
window_size = 5

# GPT-2 model
model = GPT2(vocab_size, window_size, embed_size, max_len, num_heads, ff_hidden_size, num_layers)
dataset = None


if train_type==0:
    from torch.utils.data import DataLoader, Dataset


    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop for single GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    num_epochs=100
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item()}")

if train_type==1:
    import torch.multiprocessing as mp
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset

    def train(rank, world_size):
        # Initialize process group
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        # Create model and DDP wrapper
        model = DDP(model, device_ids=[rank])

        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Initialize DataLoader with DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)

        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(rank), targets.to(rank)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item()}")

    if __name__ == '__main__':
        world_size = 4  # Number of GPUs
        mp.spawn(train, args=(world_size,), nprocs=world_size)

if train_type==2:
    import fairscale

    # Create FSDP model and optimizer
    model = fairscale.nn.FullyShardedDataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop for FSDP
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item()}")
