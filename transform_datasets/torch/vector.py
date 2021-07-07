import numpy as np
import torch
from torch.utils.data import Dataset
# from torch_tools.data import Dataset

from harmonics.groups.hierarchical_reflection import Reflection


class HarmonicPatternsS1(Dataset):
    
    def __init__(self,
                 dim=32,
                 n_classes=10,
                 n_harmonics=5,
                 max_frequency=16,
                 seed=0,
                 noise=0.0,
                 n_samples=1,
                 real=True):
        
        #TODO: Implement noise
        
        super().__init__()
        np.random.seed(seed)
        self.dim = dim
        self.n_classes = n_classes
        self.n_harmonics = n_harmonics
        self.max_frequency = max_frequency
        self.seed = seed
        self.real = real
        self.noise = noise
        self.n_samples = n_samples

        self.name = "harmonic-patterns-s1"
        self.coordinates = np.arange(0, np.pi * 2, np.pi * 2 / dim)
        self.gen_dataset()
        
    def gen_dataset(self):
        data = []
        labels = []
        for c in range(self.n_classes):
            d = self.random_signal()
            for s in range(n_samples):
                n = np.random.uniform(-self.noise, self.noise, size=self.dim)
                d_ = d + n
                data.append(d_)
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
                 noise=0.0,
                 n_samples=1,
                 noise_before_transformation=False,
                 real=True,
                 equivariant=False):
                
        self.percent_transformations = percent_transformations
        self.n_transformations = int(dim * percent_transformations)
        self.ordered = ordered
        self.equivariant = equivariant
        self.noise = noise
        self.noise_before_transformation = noise_before_transformation
        self.n_samples = n_samples
        
        if noise_before_transformation:
            class_noise = noise
        else:
            class_noise = 0.0
        
        super().__init__(dim=dim,
                         n_classes=n_classes,
                         n_harmonics=n_harmonics,
                         max_frequency=max_frequency,
                         noise=class_noise,
                         n_samples=n_samples,
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
            if not self.noise_before_transformation:
                for s in range(self.n_samples):
                    n = np.random.uniform(-self.noise, self.noise, size=self.dim)
                    signal_t_ = signal_t + n
                    orbit.append(signal_t_) 
                else:
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
    
class Cyclic1DTranslation(Dataset):
    def __init__(
        self,
        n_classes=100,
        dim=32,
        noise=0.2,
        noise_before_transformation=True,
        n_samples=100,
        seed=0,
        percent_transformations=1.0,
        ordered=False,
        equivariant=False,
    ):

        np.random.seed(seed)
        self.name = "cyclic-1d-translation"
        random_classes = np.random.uniform(-1, 1, size=(n_classes, dim))
        random_classes -= np.mean(random_classes, axis=1, keepdims=True)
        dataset, labels, s = [], [], []
        if equivariant:
            x0 = []

        all_transformations = np.arange(dim)
        n_transformations = int(percent_transformations * len(all_transformations))
        
        if noise_before_transformation:      
            for i, c in enumerate(random_classes):
                if not ordered:
                    np.random.shuffle(all_transformations)
                select_transformations = all_transformations[:n_transformations]
                for s in range(n_samples):
                    n = np.random.uniform(-noise, noise, size=dim)
                    c_ = c + n
                    for t in select_transformations:
                        datapoint = self.translate(c_, t)
                        dataset.append(datapoint)
                        labels.append(i)
                        s.append(t)
                        
        else:
            for i, c in enumerate(random_classes):
                if not ordered:
                    np.random.shuffle(all_transformations)
                select_transformations = all_transformations[:n_transformations]
                for t in select_transformations:
                    datapoint = self.translate(c, t)
                    for s in range(n_samples):
                        n = np.random.uniform(-noise, noise, size=dim)
                        datapoint_ = datapoint + n
                        dataset.append(datapoint_)
                        labels.append(i)
                        transformations.append(t)

        self.data = torch.Tensor(dataset)
        self.labels = torch.Tensor(labels)
        self.s = torch.Tensor(s)
        self.dim = dim
        self.n_classes = n_classes
        self.noise = noise
        self.noise_before_transformation = noise_before_transformation
        self.seed = seed
        self.percent_transformations = percent_transformations
        self.ordered = ordered
        self.equivariant = equivariant

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
