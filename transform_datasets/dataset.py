import torch
from torch.utils.data import Dataset
from collections import OrderedDict


class TransformDataset(Dataset):
    def __init__(self, dataset, transforms):
        """
        Arguments
        ---------
        pattern_generator (obj):
            object from patterns.natural or patterns.synthetic
        transforms (list of obj):
            list of objects from transformations. the order of the objects
            determines the order in which they are applied.
        n_classes (int):
            number of classes to generate
        """
        self.transforms = transforms
        self.gen_transformations(dataset)

    def gen_transformations(self, dataset):
        data, labels, transformations = [], [], []
        transformations = OrderedDict({x.name: [] for x in self.transforms})
        for x, y in dataset:
            xt = x.copy()
            for transform in self.transforms:
                xt, t = transform(xt)
                transformations[self.transforms.name] += t
            data += xt
            labels += [y] * len(xt)
        self.data = data
        self.labels = labels
        self.transformations = transformations

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
