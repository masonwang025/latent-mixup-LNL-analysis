import torch
from torchvision import datasets, transforms
import numpy as np

def get_spiral_datasets(dir):
    full_dataset = SpiralsDataset(dir)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, torch.clone(train_dataset), test_dataset

def get_mnist_datasets(dir):
    trainset = datasets.MNIST(
        root=dir, train=True, download=True, transform=transforms.ToTensor())
    trainset_track = datasets.MNIST(
        root=dir, train=True, transform=transforms.ToTensor())
    testset = datasets.MNIST(
        root=dir, train=False, transform=transforms.ToTensor())
    return trainset, trainset_track, testset

class SpiralDataset(torch.utils.data.Dataset):
    def __init__(self, n_points, noise=0.5):
        self.x, self.y = generate_two_spirals_dataset(n_points, noise=noise)
        self.indices = np.random.shuffle(torch.range(self.x.shape[0]))

    def __getitem__(self, index):
        return self.x[self.indices[index]], self.y[self.indices[index]]

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
