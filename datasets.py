import torch
from torchvision import datasets, transforms
import numpy as np
import copy

def get_spiral_datasets(dir=''):
    trainset = SpiralDataset(dir, train=True)
    trainset_track = SpiralDataset(dir, train=True)
    testset = SpiralDataset(dir, train=False)
    return trainset, trainset_track, testset

def get_mnist_datasets(dir):
    trainset = datasets.MNIST(
        root=dir, train=True, download=True, transform=transforms.ToTensor())
    trainset_track = datasets.MNIST(
        root=dir, train=True, transform=transforms.ToTensor())
    testset = datasets.MNIST(
        root=dir, train=False, transform=transforms.ToTensor())
    return trainset, trainset_track, testset

class SpiralDataset(torch.utils.data.Dataset):
    def __init__(self, dir, n_points=2000, noise=0.5, train=True):
        super().__init__()
        self.data, self.targets = generate_two_spirals_dataset(n_points, noise=noise)
        self.indices = torch.arange(0, self.data.shape[0])
        np.random.shuffle(self.indices)

        if train:
            self.indices = self.indices[:int(0.8 * len(self.indices))]
        else:
            self.indices = self.indices[int(0.8 * len(self.indices)):]

    def __getitem__(self, index):
        return self.data[self.indices[index]], self.targets[self.indices[index]]

    def __len__(self):
        return len(self.indices)

def generate_two_spirals_dataset(n_points, noise=.5):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points), np.ones(n_points)))
    )
