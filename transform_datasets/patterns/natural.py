import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MNIST(Dataset):
    '''
    Dataset object for the MNIST dataset.
    Takes the MNIST file path, then loads, standardizes, and saves it internally.
    '''
    
    def __init__(
        self,
        path,
        ordered=False
    ):

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
        mnist = mnist.reshape((len(mnist), 28, 28))

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
    
    
class Omniglot(Dataset):
    def __init__(
        self,
        path,
        ravel=True,
    ):

        super().__init__()

        self.name = "omniglot"
        self.ravel = ravel

        omniglot = pd.read_pickle(path)
        labels = np.array(list(omniglot.labels))
        alphabet_labels = np.array(list(omniglot.alphabet_labels))
        imgs = np.array(list(omniglot.imgs))

        self.dim = 900
        self.img_size = (30, 30)
        imgs -= imgs.mean(axis=(1, 2), keepdims=True)
        imgs /= imgs.std(axis=(1, 2), keepdims=True)

        if not ravel:
            self.channels = 1
            imgs = imgs.reshape((imgs.shape[0], 1, 30, 30))
        else:
            imgs = imgs.reshape((imgs.shape[0], -1))
            
        self.data = torch.Tensor(imgs)
        self.labels = torch.Tensor(labels)
        self.alphabet_labels = torch.Tensor(alphabet_labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
