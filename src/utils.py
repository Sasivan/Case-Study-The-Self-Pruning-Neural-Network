import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def compute_sparsity(model, threshold):
    total_params = 0
    pruned_params = 0
    gates = model.get_all_gates()
    for g in gates:
        total_params += g.numel()
        pruned_params += (g < threshold).sum().item()
    return pruned_params / total_params if total_params > 0 else 0

def plot_gate_distribution(model, lmbda, output_dir):
    gates = torch.cat([g.flatten() for g in model.get_all_gates()]).detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(gates, bins=100, color='skyblue', edgecolor='black')
    plt.title(f'Gate Distribution (Lambda={lmbda})')
    plt.xlabel('Gate Value (Sigmoid Output)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    path = os.path.join(output_dir, f'gates_lambda_{lmbda}.png')
    plt.savefig(path)
    plt.close()
