import torch
from torch.utils.data import Dataset
from collections import OrderedDict


class TransformDataset(Dataset):
    def __init__(self, dataset, transforms):
        """
        Arguments
        ---------
        dataset (obj):
            Object from patterns.natural or patterns.synthetic
        transforms (list of obj):
            List of objects from transformations. The order of the objects
            determines the order in which they are applied.
        """
        if type(transforms) != list:
            transforms = [transforms]
        self.transforms = transforms
        self.gen_transformations(dataset)

    def gen_transformations(self, dataset):
        transform_dict = OrderedDict()
        transformed_data = dataset.data.clone()
        new_labels = dataset.labels.clone()
        for transform in self.transforms:
            transformed_data, new_labels, transform_dict, t = transform(
                transformed_data, new_labels, transform_dict
            )
            transform_dict[transform.name] = t
        self.data = transformed_data
        self.labels = new_labels
        self.transform_labels = transform_dict

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
