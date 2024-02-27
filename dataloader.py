import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloader():
    # Load the dataset
    data = np.load('data.npy')
    data = data.reshape(8565, 50, 50)
    data_tensor = torch.Tensor(data).unsqueeze(1)  # Add channel dimension
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader