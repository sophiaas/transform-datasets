import numpy as np
import torch
from torch.utils.data import Dataset
from transform_datasets.transforms.functional import translate1d, translate2d
import pyshtools as pysh
from PIL import Image
import os
import math


class PatternDataset:
    def __init__(self, **kwargs):
        assert "name" in kwargs, "Keyword arguments must include keyword 'name'."
        self.set_attributes(kwargs)

    def set_attributes(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def gen_pattern(self):
        raise NotImplementedError

    def gen_dataset(self):
        data = []
        labels = []
        for y in range(self.n_classes):
            x = self.gen_pattern()
            data.append(x)
            labels.append(y)
        self.data = torch.tensor(np.array(data))
        self.labels = torch.tensor(np.array(labels))

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
    
    def __len__(self):
        return len(self.data)
    

class BinaryGraphs(PatternDataset):
    def __init__(
        self,
        name="binary_graphs",
        dim=32,
        sparsity=0.7,
        n_classes=10,
    ):

        super().__init__(
            name=name,
            dim=dim,
            n_classes=n_classes,
        )
        self.sparsity = sparsity
        self.gen_dataset()

    def gen_pattern(self):
        graph = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                x = np.random.uniform(0, 1)
                if x >= self.sparsity:
                    graph[i, j] = 1.0
                    graph[j, i] = 1.0
#         x = np.random.uniform(0, 1, size=(self.dim, self.dim))
#         x[x < self.sparsity] = 0.0
#         x[x >= self.sparsity] = 1.0

#         x -= np.mean(x)
#         x /= np.std(abs(x))
        return graph
    

class HarmonicsS1(PatternDataset):
    def __init__(
        self,
        name="harmonics-s1",
        dim=32,
        n_harmonics=5,
        max_frequency=16,
        real=True,
        n_classes=10,
    ):

        super().__init__(
            name=name,
            dim=dim,
            n_harmonics=n_harmonics,
            max_frequency=max_frequency,
            real=real,
            n_classes=n_classes,
        )
        self.coordinates = np.arange(0, np.pi * 2, np.pi * 2 / self.dim)
        self.gen_dataset()

    def gen_pattern(self):
        x = np.zeros(self.dim, dtype=np.complex64)
        for i in range(self.n_harmonics):
            omega = np.random.randint(self.max_frequency + 1)
            phase = np.random.randint(self.dim)
            amplitude = np.random.uniform(0, 1)
            coords = translate1d(self.coordinates, phase)
            f = amplitude * np.exp(1j * omega * coords)
            x += f
        if self.real:
            x = x.real
        x -= np.mean(x)
        x /= np.max(abs(x))
        return x


class HarmonicsS1xS1(PatternDataset):
    def __init__(
        self,
        name="harmonics-s1xs1",
        img_size=(32, 32),
        n_classes=10,
        n_harmonics=5,
        max_frequency=5,
        real=True,
    ):
        super().__init__(
            name=name,
            img_size=img_size,
            n_classes=n_classes,
            n_harmonics=n_harmonics,
            max_frequency=max_frequency,
            real=real,
        )

        self.coordinates_v = np.linspace(0, np.pi * 2, self.img_size[0], endpoint=False)
        self.coordinates_h = np.linspace(0, np.pi * 2, self.img_size[1], endpoint=False)
        self.grid_h, self.grid_v = np.meshgrid(self.coordinates_h, self.coordinates_v)
        self.dim = img_size[0] * img_size[1]
        self.gen_dataset()

    def gen_pattern(self):
        x = np.zeros(self.img_size, dtype=np.complex64)
        for i in range(self.n_harmonics):
            omega_h, omega_v = (
                np.random.randint(-self.max_frequency, self.max_frequency + 1),
                np.random.randint(-self.max_frequency, self.max_frequency + 1),
            )
            phase_h, phase_v = np.random.randint(self.img_size[1]), np.random.randint(
                self.img_size[0]
            )
            amplitude = np.random.uniform(0, 1)
            coords_h, coords_v = (
                translate2d(self.grid_h, phase_v, phase_h),
                translate2d(self.grid_v, phase_v, phase_h),
            )
            f = np.cos(coords_h * omega_h + coords_v * omega_v) + 1j * np.sin(
                coords_h * omega_h + coords_v * omega_v
            )
            x += f
        if self.real:
            x = x.real
        x -= np.mean(x)
        x /= np.max(abs(x))
        return x


class HarmonicsS2(PatternDataset):
    def __init__(
        self,
        name="harmonics-s2",
        l_max_dim=50,
        l_max_sample=20,
        n_classes=10,
        n_harmonics=10,
        kind="real",
        grid_type="GLQ",
    ):
        super().__init__(
            name=name,
            l_max_dim=l_max_dim,
            l_max_sample=l_max_sample,
            n_classes=n_classes,
            n_harmonics=n_harmonics,
            kind=kind,
            grid_type=grid_type,
        )
        self.gen_dataset()

    def gen_dataset(self):
        data = []
        labels = []
        for y in range(self.n_classes):
            x = self.gen_pattern()
            data.append(x)
            labels.append(y)
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)

    def gen_pattern(self):
        coeffs = pysh.SHCoeffs.from_zeros(
            self.l_max_dim, kind=self.kind, normalization="ortho"
        )
        vals, l, m = [], [], []
        for i in range(self.n_harmonics):
            vals.append(np.random.uniform())
            l.append(np.random.randint(0, self.l_max_sample + 1))  # degree
            m.append(np.random.randint(-l[i], l[i] + 1))  # order
        coeffs.set_coeffs(vals, l, m)
        pattern = coeffs.expand(grid=self.grid_type)
        return pattern.data


class HarmonicsDisk:
    def __init__(self):
        raise NotImplementedError

    def gen_pattern(self):
        raise NotImplementedError


class RandomUniform(PatternDataset):
    def __init__(
        self, name="random-uniform", size=(32,), magnitude=1.0, n_classes=10
    ):
        super().__init__(
            name=name, size=size, magnitude=magnitude, n_classes=n_classes
        )
        self.gen_dataset()

    def gen_pattern(self):
        pattern = np.random.uniform(-self.magnitude, self.magnitude, size=self.size)
        return pattern


class RandomNormal(PatternDataset):
    def __init__(
        self, name="random-normal", size=(32,), mean=0.0, std=1.0, n_classes=10
    ):

        super().__init__(
            name=name, size=size, mean=mean, std=std, n_classes=n_classes
        )
        self.gen_dataset()

    def gen_pattern(self):
        pattern = np.random.normal(loc=self.mean, scale=self.std, size=self.size)
        pattern -= pattern.mean(keepdims=True)
        pattern /= pattern.std(keepdims=True)
        return pattern
    
    
class UniformPhasors(PatternDataset):
    
    def __init__(self,
                 size=(32,),
                 n_classes=10,
                 name="uniform-phasors"):
        
        super().__init__(size=size, 
                         n_classes=n_classes,
                         name=name)
        
        self.gen_dataset()
        
    def gen_pattern(self):
        phase = np.random.uniform(-np.pi, np.pi, size=self.size)
        return np.exp(1j * phase)
    
    
class WhiteNoise1D(PatternDataset):
    
    def __init__(self,
                 dim=32,
                 n_classes=10,
                 real=True,
                 zero_mean=True,
                 smooth=False,
                 sigma=None,
                 name="white-noise-1d"):
        
        super().__init__(dim=dim, 
                         n_classes=n_classes,
                         name=name,
                         real=real,
                         smooth=smooth,
                         sigma=sigma,
                         zero_mean=zero_mean)
        
        self.gen_dataset()
        
    def gaussian_kernel_1d(self):
        if self.sigma is None:
            self.sigma = self.dim / 8
        x = np.linspace(-(self.dim - 1) / 2., (self.dim - 1) / 2., self.dim)
        kernel = np.exp(-0.5 * (x / self.sigma) ** 2) 
        return kernel / np.max(kernel)
        
    def gen_pattern(self):
        max_freq = math.floor(self.dim / 2)
        even = (self.dim % 2) == 0
        if self.smooth:
            kernel = self.gaussian_kernel_1d()
        else:
            kernel = np.ones(self.dim)
        if self.real:   
            if even:
                neg = np.random.uniform(-np.pi, np.pi, max_freq)
                zero = np.random.uniform(-np.pi, np.pi, 1)
                pos = -(neg[1:][::-1])
                phase_pattern = np.concatenate([neg, zero, pos])
            else:
                neg = np.random.uniform(-np.pi, np.pi, max_freq)
                zero = np.random.uniform(-np.pi, np.pi, 1)
                pos = -(neg[::-1])
                phase_pattern = np.concatenate([neg, zero, pos])
            f_coeffs = kernel * np.exp(1j * phase_pattern)
            signal = np.fft.ifft(np.fft.ifftshift(f_coeffs)).real
        else:
            phase_pattern = np.random.uniform(-np.pi, np.pi, self.dim)
            f_coeffs = kernel * np.exp(1j * phase_pattern)
            signal = np.fft.ifft(np.fft.ifftshift(f_coeffs))
        if self.zero_mean:
            signal = signal - np.mean(signal, keepdims=True)
            signal = signal / np.std(signal, keepdims=True)
        return signal


class NaturalImageSlices(PatternDataset):
    
    def __init__(self,
                 name="natural-image-slices",
                 path=os.path.expanduser("~/data/curated-natural-images/images/"),
                 n_classes=10,
                 slice_length=256,
                 images=range(94),
                 color=False,
                 normalize=True,
                 min_contrast=0.2):
        
        super().__init__(name=name,
                         path=path,
                         n_classes=n_classes,
                         slice_length=slice_length,
                         images=images,
                         color=color,
                         normalize=normalize,
                         min_contrast=min_contrast)
        
        self.img_size = (512, 512)
        
        if self.img_size[0] - slice_length < 0 or self.img_size[1] - slice_length < 0:
            raise ValueError("Slice length must be <= 512")
            
        self.max_start = (self.img_size[0] - slice_length, self.img_size[1] - slice_length)
        self.gen_dataset()
        
    def gen_pattern(self):
        idx = np.random.choice(self.images)
        n_zeros = 4 - len(str(idx))
        str_idx = "0" * n_zeros + str(idx)
        img = np.asarray(Image.open(self.path + "{}.png".format(str_idx)))

        if not self.color:
            img = img.mean(axis=-1)

        slice_orientation = np.random.choice(range(2))
        
        low_contrast = True
        max_attempts = 50
        attempt = 0
        while low_contrast:
            if attempt >= max_attempts:
                print("Max attempts reached for given min_contrast level. Try lowering min_contrast.")
                break
            if slice_orientation == 0:
                row = np.random.randint(self.img_size[0])
                slice_start = np.random.choice(range(self.max_start[0]))
                img_slice = img[row, slice_start:slice_start+self.slice_length]
            else:
                col = np.random.randint(self.img_size[1])
                slice_start = np.random.choice(range(self.max_start[1]))
                img_slice = img[slice_start:slice_start+self.slice_length, col]
            if img_slice.std() >= self.min_contrast:
                low_contrast = False
            attempt += 1

        if self.normalize:
            img_slice -= img_slice.mean()
            img_slice /= img_slice.std()
            
        return img_slice