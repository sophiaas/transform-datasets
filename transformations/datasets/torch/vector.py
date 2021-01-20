import numpy as np
import torch
from torch.utils.data import Dataset
from harmonics.groups.hierarchical_reflection import Reflection


class Translation(Dataset):
    
    def __init__(self,
                 n_classes=100,
                 transformation='translation',
                 max_transformation_steps=10,
                 dim=25,
                 noise=0.2,
                 seed=0):
                
        np.random.seed(seed)

        random_classes = np.random.uniform(-1, 1, size=(n_classes, dim))
        random_classes -= np.mean(random_classes, axis=1, keepdims=True)
        dataset, labels, transformations = [], [], []

        for i, c in enumerate(random_classes):
            for t in range(max_transformation_steps):
                datapoint = self.translate(c, t, max_transformation_steps)
                dataset.append(datapoint)
                labels.append(i)
                transformations.append(t)

                # Negative translation
                datapoint = translate(c, -t, max_transformation_steps)
                dataset.append(datapoint)
                labels.append(i)
                transformations.append(-t)
                
        self.data = torch.Tensor(dataset)
        self.labels = torch.Tensor(labels)
        self.transformation = torch.Tensor(transformations)
        self.dim = self.data.shape[1]
        self.n_classes = n_classes
    
    def translate(self, 
                  x, 
                  translation, 
                  max_transformation):
        
        new_x = np.zeros(max_transformation * 2 + len(x))
        start = max_transformation + translation
        new_x[start:start+len(x)] = x
        return new_x
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    
    
class CyclicTranslation(Dataset):
    
    def __init__(self,
                 n_classes=100,
                 dim=32,
                 noise=0.2,
                 seed=0,
                 percent_transformations=1.0):
                
        np.random.seed(seed)

        random_classes = np.random.uniform(-1, 1, size=(n_classes, dim))
        random_classes -= np.mean(random_classes, axis=1, keepdims=True)
        dataset, labels, transformations = [], [], []
        
        all_transformations = np.arange(dim)
        n_transformations = int(percent_transformations * len(all_transformations))
        np.random.shuffle(all_transformations)
        select_transformations = all_transformations[:n_transformations]

        for i, c in enumerate(random_classes):
            for t in select_transformations:
                datapoint = self.translate(c)
                n = np.random.uniform(-noise, noise, size=dim)
                datapoint += n
                dataset.append(datapoint)
                labels.append(i)
                transformations.append(t)
                
        self.data = torch.Tensor(dataset)
        self.labels = torch.Tensor(labels)
        self.transformation = torch.Tensor(transformations)
        self.dim = self.data.shape[1]
        self.n_classes = n_classes
    
    def translate(self, x):
        x = list(x)
        last = x.pop()
        x = [last] + x
        return np.array(x)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    
class HierarchicalReflection(Dataset):
    
    def __init__(self,
                 n_classes=100,
                 n_transformations=100,
                 dim=32,
                 noise=0.0,
                 seed=0):
                
        np.random.seed(seed)
        
        bits = np.log2(dim)
        if not bits.is_integer():
            raise ValueError('dim must be a power of 2')

        self.group = Reflection(int(bits))
        
        random_classes = np.random.uniform(-1, 1, size=(n_classes, dim))
        random_classes -= np.mean(random_classes, axis=1, keepdims=True)
        dataset, labels = [], []

        for i, c in enumerate(random_classes):
            for t in range(n_transformations):
                datapoint = self.group.rand_element(c)
                n = np.random.uniform(-noise, noise, size=dim)
                datapoint += n
                dataset.append(datapoint)
                labels.append(i)
                
        self.data = torch.Tensor(dataset)
        self.labels = torch.Tensor(labels)
        self.dim = self.data.shape[1]
        self.n_classes = n_classes
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)

    