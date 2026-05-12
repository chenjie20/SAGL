import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, datasets, labels):
        self.datasets = datasets
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):

        # data = [dataset[idx] for dataset in self.datasets]
        data = tuple(dataset[idx] for dataset in self.datasets)
        labels = self.labels[idx]

        return data, labels


class CustomSimplifiedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return self.datasets[0].shape[0]

    def __getitem__(self, idx):

        # data = [dataset[idx] for dataset in self.datasets]
        data = tuple(dataset[idx] for dataset in self.datasets)

        return data
