import numpy as np
import torch
from torch.utils.data import Dataset

from harmonics.groups.hierarchical_reflection import Reflection


class HarmonicPatternsS1(Dataset):
    
    def __init__(self,
                 dim=32,
                 n_classes=10,
                 n_harmonics=5,
                 max_frequency=16,
                 seed=0,
                 real=True):
        
        super().__init__()
        np.random.seed(seed)
        self.dim = dim
        self.n_classes = n_classes
        self.n_harmonics = n_harmonics
        self.max_frequency = max_frequency
        self.seed = seed
        self.real = real

        self.name = "harmonic-patterns-s1"
        self.coordinates = np.arange(0, np.pi * 2, np.pi * 2 / dim)
        self.gen_dataset()
        
    def gen_dataset(self):
        data = []
        labels = []
        for c in range(self.n_classes):
            d = self.random_signal()
            data.append(d)
            labels.append(c)
        self.data = torch.tensor(np.array(data))
        self.labels = torch.tensor(labels)
        
    def random_signal(self):
        d = np.zeros(self.dim, dtype=np.complex64)
        for i in range(self.n_harmonics):
            omega = np.random.randint(self.max_frequency + 1)
            phase = np.random.randint(self.dim)
            amplitude = np.random.uniform(0, 1)
            coords = self.translate(self.coordinates, phase)
            f = amplitude * np.exp(1j * omega * coords)
            d += f
        if self.real:
            d = d.real
        d -= np.mean(d)
        d /= np.max(abs(d))
        return d

    def translate(self, x, t):
        new_x = list(x)
        for i in range(t):
            last = new_x.pop()
            new_x = [last] + new_x
        return np.array(new_x)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    
class HarmonicPatternsS1Orbit(HarmonicPatternsS1):
    
    def __init__(self,
                 dim=32,
                 n_classes=10,
                 n_harmonics=5,
                 percent_transformations=1.0,
                 ordered=False,
                 max_frequency=16,
                 seed=0,
                 real=True,
                 equivariant=False):
        
        self.percent_transformations = percent_transformations
        self.n_transformations = int(dim * percent_transformations)
        self.ordered = ordered
        self.equivariant = equivariant
        
        super().__init__(dim=dim,
                         n_classes=n_classes,
                         n_harmonics=n_harmonics,
                         max_frequency=max_frequency,
                         seed=seed,
                         real=real)
        
        self.name = 'harmonic-patterns-s1-orbit'
        
    def gen_dataset(self):
        data = []
        labels = []
        for c in range(self.n_classes):
            signal = self.random_signal()
            orbit = self.gen_orbit(signal)
            data += orbit  
            labels += [c] * len(orbit)
        self.data = torch.tensor(np.array(data))
        self.labels = torch.tensor(labels)
        
    def gen_orbit(self, signal):
        orbit = []
        all_transformations = np.arange(self.dim)
        if not self.ordered:
             np.random.shuffle(all_transformations)
        select_transformations = all_transformations[:self.n_transformations]
        for g in select_transformations:
            signal_t = self.translate(signal, g)
            orbit.append(signal_t) 
        return orbit
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class Translation(Dataset):
    def __init__(
        self, n_classes=100, max_transformation_steps=10, dim=25, noise=0.2, seed=0
    ):

        np.random.seed(seed)
        
        random_classes = np.random.uniform(
            -1, 1, size=(n_classes, dim - max_transformation_steps * 2)
        )
        random_classes -= np.mean(random_classes, axis=1, keepdims=True)
        random_classes /= np.std(random_classes, axis=1, keepdims=True)
        
        dataset, labels, transformations = [], [], []

        for i, c in enumerate(random_classes):
            for t in range(max_transformation_steps):
                datapoint = self.translate(c, t, max_transformation_steps)
                dataset.append(datapoint)
                labels.append(i)
                transformations.append(t)

                # Negative translation
                datapoint = self.translate(c, -t, max_transformation_steps)
                dataset.append(datapoint)
                labels.append(i)
                transformations.append(-t)

        self.data = torch.Tensor(dataset)
        self.labels = torch.Tensor(labels)
        self.transformation = torch.Tensor(transformations)
        self.dim = self.data.shape[1]
        self.n_classes = n_classes

    def translate(self, x, translation, max_transformation):

        new_x = np.zeros(max_transformation * 2 + len(x))
        start = max_transformation + translation
        new_x[start : start + len(x)] = x
        return new_x

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class CyclicTranslation(Dataset):
    def __init__(
        self,
        n_classes=100,
        dim=32,
        noise=0.2,
        seed=0,
        percent_transformations=1.0,
        ordered=False,
    ):

        np.random.seed(seed)
        self.name = "cyclic-translation"
        random_classes = np.random.uniform(-1, 1, size=(n_classes, dim))
        random_classes -= np.mean(random_classes, axis=1, keepdims=True)
        dataset, labels, transformations = [], [], []

        all_transformations = np.arange(dim)
        n_transformations = int(percent_transformations * len(all_transformations))

        for i, c in enumerate(random_classes):
            if not ordered:
                np.random.shuffle(all_transformations)
            select_transformations = all_transformations[:n_transformations]
            for t in select_transformations:
                datapoint = self.translate(c, t)
                n = np.random.uniform(-noise, noise, size=dim)
                datapoint += n
                dataset.append(datapoint)
                labels.append(i)
                transformations.append(t)

        self.data = torch.Tensor(dataset)
        self.labels = torch.Tensor(labels)
        self.transformation = torch.Tensor(transformations)
        self.dim = dim
        self.n_classes = n_classes
        self.noise = noise
        self.seed = seed
        self.percent_transformations = percent_transformations
        self.ordered = ordered

    def translate(self, x, t):
        new_x = list(x)
        for i in range(t):
            last = new_x.pop()
            new_x = [last] + new_x
        return np.array(new_x)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class HierarchicalReflection(Dataset):
    def __init__(self, n_classes=100, n_transformations=100, dim=32, noise=0.0, seed=0):

        np.random.seed(seed)

        bits = np.log2(dim)
        if not bits.is_integer():
            raise ValueError("dim must be a power of 2")

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


def gen_1D_fourier_variant(
    n_samples,
    dim,
    complex_input=False,
    project_input=False,
    project_output=False,
    variant="fourier",
    standardize=False,
    seed=0,
):

    # TODO: Either convert this into proper datasets or delete

    np.random.seed(seed)

    X = np.random.uniform(-1, 1, size=(n_samples, dim))

    if complex_input:
        X += 1j * np.random.uniform(-1, 1, size=(n_samples, dim))

    F = np.fft.fft(X, axis=-1)

    if complex_input and project_input:
        X = np.hstack([X.real, X.imag])

    if variant == "fourier":
        if project_output:
            F = np.hstack([F.real, F.imag])
        return X, F

    elif variant == "power-spectrum":
        PS = np.abs(F) ** 2
        return X, PS

    elif variant == "bispectrum":
        B = []
        for f in F:
            b = np.zeros(dim + 1, dtype=np.complex)
            b[0] = np.conj(f[0] * f[0]) * f[0]
            b[1:-1] = np.conj(f[:-1] * f[1]) * f[1:]
            b[dim] = np.conj(f[dim - 1] * f[1]) * f[0]
            B.append(b)
        B = np.array(B)
        if project_output:
            B = np.hstack([B.real, B.imag])
        return X, B

    else:
        raise ValueError("Invalid dataset type")
