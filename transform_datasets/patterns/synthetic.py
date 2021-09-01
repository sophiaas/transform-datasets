import numpy as np
import torch
from torch.utils.data import Dataset
from transform_datasets.transforms.functional import translate1d, translate2d
import pyshtools as pysh


class PatternDataset:
    def __init__(self, **kwargs):
        assert "name" in kwargs, "Keyword arguments must include keyword 'name'."
        self.set_attributes(kwargs)

    def set_attributes(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            if k == "seed":
                np.random.seed(v)

    def gen_pattern(self):
        raise NotImplementedError

    def gen_dataset(self):
        data = []
        labels = []
        for y in range(self.n_classes):
            x = self.gen_pattern()
            data.append(x)
            labels.append(y)
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


class HarmonicsS1(PatternDataset):
    def __init__(
        self,
        name="harmonics-s1",
        dim=32,
        n_harmonics=5,
        max_frequency=16,
        real=True,
        seed=0,
        n_classes=10,
    ):

        super().__init__(
            name=name,
            dim=dim,
            n_harmonics=n_harmonics,
            max_frequency=max_frequency,
            real=real,
            seed=seed,
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
        seed=0,
        real=True,
    ):
        super().__init__(
            name=name,
            img_size=img_size,
            n_classes=n_classes,
            n_harmonics=n_harmonics,
            max_frequency=max_frequency,
            seed=seed,
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
        seed=0,
        grid_type="GLQ",
    ):
        super().__init__(
            name=name,
            l_max_dim=l_max_dim,
            l_max_sample=l_max_sample,
            n_classes=n_classes,
            n_harmonics=n_harmonics,
            kind=kind,
            seed=seed,
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
        self, name="random-uniform", size=(32,), seed=0, magnitude=1.0, n_classes=10
    ):
        super().__init__(
            name=name, size=size, seed=seed, magnitude=magnitude, n_classes=n_classes
        )
        self.gen_dataset()

    def gen_pattern(self):
        return np.random.uniform(-self.magnitude, self.magnitude, size=self.size)


class RandomNormal(PatternDataset):
    def __init__(
        self, name="random-normal", size=(32,), seed=0, mean=0.0, std=1.0, n_classes=10
    ):

        super().__init__(
            name=name, size=size, seed=seed, mean=mean, std=std, n_classes=n_classes
        )
        self.gen_dataset()

    def gen_pattern(self):
        return np.random.normal(loc=self.mean, scale=self.std, size=self.size)
