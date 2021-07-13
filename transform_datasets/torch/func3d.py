import torch
from torch.utils.data import Dataset
import numpy as np
import pyshtools as pysh
import itertools


class HarmonicPatternsS2(Dataset):
    def __init__(
        self,
        l_max_dim=50,
        l_max_sample=20,
        n_classes=100,
        n_harmonics=10,
        kind='real',
        seed=0,
        grid_type='GLQ',
        ravel=True,
    ):
        self.l_max_dim = l_max_dim
        self.l_max_sample = l_max_sample
        self.n_classes = n_classes
        self.n_harmonics = n_harmonics
        self.kind = kind
        self.seed = seed
        self.grid_type = grid_type
        self.ravel = ravel
        np.random.seed(seed)
        
        self.gen_dataset()
        
        self.img_size = self.data.shape[1:]
        
        if self.ravel:
            self.data = self.data.reshape(self.data.shape[0], -1)
            self.dim = self.data.shape[-1]
        else:
            self.dim = self.img_size
            
        self.name = 'harmonic-patterns-s2'
            
    def random_coeffs(self):
        coeffs = pysh.SHCoeffs.from_zeros(self.l_max_dim, 
                                          kind=self.kind, 
                                          normalization='ortho')
        vals, l, m = [], [], []
        for i in range(self.n_harmonics):
            vals.append(np.random.uniform())
            l.append(np.random.randint(0, self.l_max_sample + 1)) #degree
            m.append(np.random.randint(-l[i], l[i]+1)) #order
        coeffs.set_coeffs(vals, l, m)
        return coeffs
            
    def gen_dataset(self):
        data, labels = [], []
        for n in range(self.n_classes):
            coeffs = self.random_coeffs()
            signal = coeffs.expand(grid=self.grid_type)
            labels.append(n)
            data.append(signal.data)
        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels)
            
    @property
    def data_img(self):
        return self.data.reshape((self.data.shape[0],) + self.img_size)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    def plot3d(self, 
               idx, 
               elevation=20, 
               azimuth=30, 
               cmap='viridis',
               cmap_limits=None, 
               cmap_reverse=False, 
               title=False,
               titlesize=None, 
               scale=4., 
               ax=None, 
               show=True, 
               fname=None):
        
        grid = pysh.SHGrid.from_array(self.data_img[idx].numpy(), grid=self.grid_type)
        fig, ax = grid.plot3d(elevation=elevation,
                              azimuth=azimuth,
                              cmap=cmap,
                              cmap_limits=cmap_limits,
                              title=title,
                              titlesize=titlesize,
                              ax=ax,
                              show=show,
                              fname=fname)
    
    
class HarmonicPatternsS2Orbit(HarmonicPatternsS2):
    
    def __init__(self,
                 l_max_dim=50,
                 l_max_sample=20,
                 n_classes=100,
                 n_harmonics=10,
                 n_axis_rotations=10,
                 kind='real',
                 seed=0,
                 grid_type='GLQ',
                 return_transform=False,
                 ravel=True):
        
        self.return_transform = return_transform
        self.n_axis_rotations = n_axis_rotations

        self.alpha = np.arange(0, 360, 360 / n_axis_rotations)        
        self.beta = np.arange(0, 180, 180 / n_axis_rotations)
        self.gamma = np.arange(0, 360, 360 / n_axis_rotations)

        self.transformations = list(
            itertools.product(
                self.alpha,
                self.beta,
                self.gamma
            )
        ) 
        
        super().__init__(l_max_dim=l_max_dim,
                         l_max_sample=l_max_sample,
                         n_classes=n_classes,
                         n_harmonics=n_harmonics,
                         kind=kind,
                         seed=seed,
                         grid_type=grid_type,
                         ravel=ravel)
        
        self.name = 'harmonic-patterns-s2-orbit'
        
    def gen_dataset(self):
        data, labels = [], []
        canonical = {}
        for n in range(self.n_classes):
            coeffs = self.random_coeffs()
            grid = coeffs.expand(grid=self.grid_type)
            canonical[n] = torch.Tensor(grid.data)
            for t in self.transformations:
                rot = coeffs.rotate(t[0], t[1], t[2])
                signal = rot.expand(grid=self.grid_type)
                data.append(signal.data)
                labels.append(n)
        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels) 
        self.canonical = canonical
        
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.return_transform:
            x0 = self.canonical[y]
            t = self.transformations[idx]
            return x, y, x0, t
        else:
            return x, y
        