import torch
from torch.utils.data import Dataset


class TransformDataset:
    def __init__(self, pattern_generator, transforms, n_classes):
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
        self.pattern_generator = pattern_generator
        self.transforms = transforms
        self.n_classes = n_classes

    def gen_dataset(self):
        data, labels, transformations = [], [], []
        canonical = {}
        for c in range(self.n_classes):
            x = self.pattern_generator.gen_pattern()
            canonical[c] = x
            xt = x.copy()
            ts = []
            for transform in self.transforms:
                xt, t = transform(xt)
                ts.append(t)
            data.append(xt)
            labels.append(c)
            transformations.append(ts)
        self.data = data
        self.labels = labels
        self.transformations = transformations
        self.canonical = canonical
