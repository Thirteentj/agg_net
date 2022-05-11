
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import csv


class CMetricDataset(Dataset):
    def __init__(self):
        self.data = np.load('observations.npy')[2:, :, :]
        self.labels = np.load('labels.npy')
        print(f'data shape: {self.data.shape}')
        print(f'labels shape: {self.labels.shape}')


    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        t = transforms.ToTensor()
        obsv = torch.tensor(self.data[idx])
        label = torch.tensor(self.labels[idx])
        return((torch.squeeze(obsv), torch.squeeze(label)))


def load_data(num_workers=0, batch_size=32):
    dataset = CMetricDataset()
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)
