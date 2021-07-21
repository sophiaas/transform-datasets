import torch
from torch.utils.data import Dataset
from collections import OrderedDict


class TransformDataset(Dataset):
    def __init__(self, dataset, transforms, return_tlabels=False):
        """
        Arguments
        ---------
        pattern_generator (obj):
            Object from patterns.natural or patterns.synthetic
        transforms (list of obj):
            List of objects from transformations. the order of the objects
            determines the order in which they are applied.
        n_classes (int):
            Number of classes to generate
        return_tlabels (bool):
            Whether to return transform labels on the __getitem__ method.
        """
        if type(transforms) != list:
            transforms = [transforms]
        self.transforms = transforms
        self.return_tlabels = return_tlabels
        self.gen_transformations(dataset)

    def gen_transformations(self, dataset):
        transform_dict = OrderedDict({x.name: [] for x in self.transforms})
        transformed_data = dataset.data.clone()
        new_labels = dataset.labels.clone()
        for transform in self.transforms:
            transformed_data, new_labels, t = transform(transformed_data, new_labels)
            transform_dict[transform.name] += t
        self.data = transformed_data
        self.labels = new_labels
        self.transform_labels = transform_dict

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.return_tlabels:
            t = self.transform_labels[idx]
            return x, y, t
        else:
            return x, y

    def __len__(self):
        return len(self.data)
