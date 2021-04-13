import torch
from torch.utils.data import Dataset
import numpy as np
import svis
from scipy.special import sph_harm


class SphericalSums(Dataset):
    def __init__(
        self,
        n=25,
        n_classes=100,
        n_harmonics=5,
        l_max=10,
        complex_valued=False,
        seed=0,
        name="spherical-sums",
    ):
        if n % 2 == 0:
            raise ValueError("n must be odd")
            
        self.grid_size = (n, 2 * n)
        self.n = n
        self.n_harmonics = n_harmonics
        self.l_max = l_max
        self.complex_valued = complex_valued
        self.n_classes = n_classes
        self.seed = seed
        self.name = name
        self.dim = self.grid_size[0] * self.grid_size[1] * 3

        np.random.seed(seed)
        torch.manual_seed(seed)

        data = []
        labels = []
        for n in range(n_classes):
            self.theta, self.phi = self.meshgrid(n=self.n)
            r = np.zeros(self.theta.shape, dtype=np.complex128)
            for i in range(self.n_harmonics):
                l = np.random.randint(0, self.l_max+1) #degree
                m = np.random.randint(-l, l+1) #order
                amplitude = np.random.uniform(0, 1)
                harmonic = sph_harm(m, l, self.phi, self.theta)
                r += amplitude * harmonic
            if not self.complex_valued:
                r = abs(r)
            r /= r.max()
            r = r.ravel()
            labels.append(n)
            data.append(np.vstack([r, self.theta.ravel(), self.phi.ravel()]).T)

        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels)
            
        
    def linspace(self, n):
        phi = np.arange(2. * n) * np.pi / (n - 1)
        theta = np.arange(n) * np.pi / (n - 1)
        return theta, phi

    def meshgrid(self, 
                 n):

        theta_bins, phi_bins = self.linspace(n)
        theta_grid, phi_grid = np.meshgrid(theta_bins, phi_bins, indexing='ij')

        return theta_grid, phi_grid
    

    def change_coordinates(self, coords, to="cartesian"):
        if to == "cartesian":
            r, theta, phi = coords.T
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            new_coords = np.vstack([x, y, z]).T

        elif to == "spherical":
            x, y, z = coords.T
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            theta = np.arccos(z / (r + 1e-10))
            phi = np.arctan2(y, x)
            new_coords = np.vstack([r, theta, phi]).T

        else:
            raise ValueError

        return torch.tensor(new_coords)
            
    def unvectorize(self, coords):
        grid_size = self.grid_size
        coords = coords.reshape((coords.shape[0], grid_size[0], grid_size[1], 3))
        return coords
    
    def unravel(self, coords):
        grid_size = self.grid_size
        coords = coords.reshape((coords.shape[0], grid_size[0] * grid_size[1], 3))
        return coords
    
    @property
    def data_cartesian(self):
        return self.change_coordinates(self.unvectorize(self.data), to="cartesian")

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
import itertools


class RotatedSphericalSums(SphericalSums):
    def __init__(
        self,
        n=25,
        n_classes=100,
        n_harmonics=5,
        l_max=10,
        complex_valued=False,
        seed=0,
        n_transformations=4000,
        n_repeats=1,
        ordered=False,
        name="rotated-spherical-sums",
        equivariant=False,
    ):

        super().__init__(
            n=n,
            n_classes=n_classes,
            n_harmonics=n_harmonics,
            l_max=l_max,
            complex_valued=complex_valued,
            seed=seed,
            name=name,
        )

        self.ordered = ordered
        self.n_transformations = n_transformations
        self.n_repeats = n_repeats
        self.name = name
        self.equivariant = equivariant
        
        phi_range = self.phi[0]
        theta_range = self.theta[:, 0]

        data = []
        labels = []
        s = []
        
        if equivariant:
            raise NotImplementedError
            x0 = []
            
        from geomstats.geometry.special_orthogonal import SpecialOrthogonal
        so3 = SpecialOrthogonal(n=3, point_type='matrix')

        from tqdm import tqdm
        print('Generating rotations...')
        for i, d in enumerate(self.data):
            for j in tqdm(range(n_transformations)):
                cartesian = self.change_coordinates(d, to="cartesian")
                R = so3.random_uniform(n_samples=1)
                rotated = cartesian @ R
                rotated_sph = self.change_coordinates(rotated, to="spherical")
                data.append(rotated_sph.ravel())
                labels.append(self.labels[i])
                s.append(R)
                if equivariant:
                    x0.append(d)

        self.data = torch.stack(data)
        self.labels = torch.Tensor(labels)
        self.s = s
                         
        if equivariant:
            raise NotImplementedError
            self.x0 = x0

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.equivariant:
            raise NotImplementedError
            s = self.s[idx]
            x0 = self.x0[idx]
            return x, y, s, x0
        else:
            return x, y

    def __len__(self):
        return len(self.data)