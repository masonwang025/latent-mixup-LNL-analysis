import torch
from torchvision import datasets, transforms
import numpy as np
import os
import math


def get_spiral_datasets(dir):
    trainset = SpiralDataset(dir, train=True)
    trainset_track = SpiralDataset(dir, train=True)
    testset = SpiralDataset(dir, train=False)
    return trainset, trainset_track, testset


class SpiralDataset(torch.utils.data.Dataset):
    def __init__(self, dir, train=True):
        # code for generating this dataset is in spiral_dataset_generator.ipynb
        super().__init__()

        dataset_file = os.path.join(
            dir, 'spiral', 'train_dataset.txt' if train else 'test_dataset.txt')

        self.data, self.targets = [], []

        with open(dataset_file, 'r') as f:
            for line in f:
                data_x, data_y, target = line.split(' ')
                data_x, data_y = float(data_x), float(data_y)
                self.data.append([data_x, data_y, float(
                    math.cos(data_x)), float(math.sin(data_y))])
                self.targets.append(float(target))

        self.data, self.targets = torch.tensor(
            self.data), torch.tensor(self.targets)

    def __getitem__(self, index):
        # print(type(self.targets[index]))
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)
