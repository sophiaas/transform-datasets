import numpy as np
import math
import torch
from torch.utils.data import Dataset
import itertools
from scipy.special import comb
from scipy.spatial.distance import squareform



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
    
    
class GraphOrbits(Dataset):
    
    def __init__(self,
                 n_nodes=5,
                 n_repeats=1):
        
        self.n_nodes = n_nodes
        self.dim = n_nodes ** 2
        self.name = 'graph-orbits'
        self.n_repeats = n_repeats
        self.img_size = (n_nodes, n_nodes)
        
        graphs = self.all_n_node_graphs(self.n_nodes)
        orbits = self.find_isomorphisms(graphs)
        labels = []
        data = []
        for i, o in enumerate(orbits):
            data += o * n_repeats 
            labels += [i] * len(o) * n_repeats            
            
        self.data = torch.Tensor(np.array(data))
        self.data = self.data.reshape(self.data.shape[0], -1)
        self.labels = torch.Tensor(labels)
        
    def all_n_node_graphs(self, n):
        """
        Returns all symmetric n by n adjacency matrices.
        Parameters
        ----------
        n: int
            size of the adjacency matrices
        Returns
        -------
        adj_mats: lst
            List of all n by n symmetric adjacency matrices.
        """
        nodes = np.arange(n)
        configs = [tup for tup in itertools.product([0, 1], repeat=int(comb(len(nodes), 2)))]
        configs = sorted(configs, key=lambda x: sum(x))
        adj_mats = []
        for config in configs:
            t = squareform(config)
            if np.count_nonzero(np.sum(t, axis=0)) >= n and np.count_nonzero(np.sum(t, axis=1)) >= n:
                if n == 2 or not np.array_equal(np.sum(t, axis=0), np.ones(n)):
                    adj_mats.append(t)
        return adj_mats
    
    def transform(self, adj_mats):
        """
        Returns a list containing all permutations of the adjacency
        matrices in adj_mats.
        Parameters
        ----------
        adj_mats: lst
            Output of all_n_node_graphs
        Returns
        -------
        transforms: lst
            List the same length as adj_mats, where transforms[i] is a
            list containing all permutations of adj_mats[i]
        """
        n = np.arange(len(adj_mats[0]))
        perm = list(itertools.permutations(n))
        perm_rules = [list(zip(n, i)) for i in perm]
        transforms = []
        for mat in adj_mats:
            mat_transforms = []
            for rule in perm_rules:
                transform = mat.copy()
                for tup in rule:
                    transform[:, tup[0]] = mat[:, tup[1]]
                ref = transform.copy()
                for tup in rule:
                    transform[tup[0], :] = ref[tup[1], :]
                mat_transforms.append(transform)
            transforms.append(mat_transforms)
        return transforms

    def find_isomorphisms(self, adj_mats):
        """
        Groups all isomorphic graphs in a list of adjacency matrices. 
        Returns a list containing the non-redundant orbits.
        Tractable only for graphs with fewer than 6 nodes.
        Parameters
        ----------
        adj_mats: int
            List of adjacency matrices on n nodes
        Returns
        -------
        orbits: lst
            List of orbits. orbits[i] indexes orbit i and contains every element (permutation) of orbit i.
        """
        transforms = self.transform(adj_mats)
        match = np.zeros((len(adj_mats), len(adj_mats)))
        for i, mat_1 in enumerate(adj_mats):
            for j, mat_2 in enumerate(transforms):
                n = len([x for x in mat_2 if (x == mat_1).all()])
                if n > 0:
                    match[i, j] = 1
        m = [list(np.nonzero(x)[0]) for x in match]
        m.sort()
        m = list(orbits for orbits,_ in itertools.groupby(m))
        orbits = [[adj_mats[i] for i in n] for n in m]
        return orbits

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)