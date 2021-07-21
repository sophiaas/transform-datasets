import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MNIST(Dataset):
    """
    Dataset object for the MNIST dataset.
    Takes the MNIST file path, then loads, standardizes, and saves it internally.
    """

    def __init__(self, path, ordered=False):

        super().__init__()

        self.name = "mnist"
        self.dim = 28 ** 2
        self.img_size = (28, 28)
        self.ordered = ordered

        mnist = np.array(pd.read_csv(path))

        labels = mnist[:, 0]
        mnist = mnist[:, 1:]
        mnist = mnist / 255
        mnist = mnist - mnist.mean(axis=1, keepdims=True)
        mnist = mnist / mnist.std(axis=1, keepdims=True)
        #         mnist = mnist.reshape((len(mnist), 28, 28))

        if ordered:
            sort_idx = np.argsort(labels)
            mnist = mnist[sort_idx]
            labels = labels[sort_idx]

        self.data = torch.Tensor(mnist)
        self.labels = torch.Tensor(labels).long()

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
