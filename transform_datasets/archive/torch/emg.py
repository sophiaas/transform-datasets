import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import itertools
import os
from transform_datasets.torch import utils


class Gesture(Dataset):
    def __init__(
        self,
        test=False
    ):
        super().__init__()
        self.test = test
        self.name = 'emg-gesture'
        
        if test:
            emg = pd.read_pickle(os.path.expanduser('~/datasets/emg/dataset_test.p'))
        else:
            emg = pd.read_pickle(os.path.expanduser('~/datasets/emg/dataset_train.p'))
        
        self.data = torch.Tensor(list(emg.data))
        self.data = self.data.reshape(self.data.shape[0], -1)
        self.dim = self.data.shape[-1]
        
        self.labels = torch.Tensor(list(emg.labels))
        self.label_names = list(emg.gesture)
        self.experimenter = list(emg.exp)
   
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
