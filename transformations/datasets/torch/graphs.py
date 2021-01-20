import numpy as np
import math
import torch
from torch.utils.data import Dataset


class PermutedMatrices(Dataset):
    
    def __init__(self,
                 dim=6,
                 n_exemplars=100,
                 percent_transformations=0.3,
                 noise = 0.1,
                 seed = 0):

        np.random.seed(seed)
        
        self.dim = dim ** 2
        
        all_permutations = np.math.factorial(dim)
        select_permutations = int(all_permutations * percent_transformations)
        
        data = []
        labels = []
        exemplar_labels = []
        
        ex_idx = 0
        
        matrices = np.random.uniform(-1, 1, size=(n_exemplars, dim, dim))
        
        l = 0
        for mat in matrices:
            # Permute matrices + Add noise
            for i in range(select_permutations): 
                # NB: Generates a random permutation each time, may be redundant
                perm = np.random.permutation(dim)
                permuted = mat[perm][:, perm].reshape(self.dim)
                n = np.random.uniform(-noise, noise, size=self.dim)
                permuted = permuted + n
                data.append(permuted)
                labels.append(l)                    
            l += 1

                    
        data = np.array(data)
        self.data = torch.Tensor(data)
        self.labels = labels

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)