import numpy as np
import torch


class HarmonicsS1:
    def __init__(
        self,
        dim=32,
        n_harmonics=5,
        max_frequency=16,
        real=True,
        seed=0,
    ):

        super().__init__()
        np.random.seed(seed)
        self.dim = dim
        self.n_harmonics = n_harmonics
        self.max_frequency = max_frequency
        self.seed = seed
        self.real = real
        self.name = "harmonics-s1"
        self.coordinates = np.arange(0, np.pi * 2, np.pi * 2 / dim)

    def gen_pattern(self):
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


class HarmonicsS1xS1:
    def __init__(self):
        raise NotImplementedError

    def gen_pattern(self):
        raise NotImplementedError


class HarmonicsS2:
    def __init__(self):
        raise NotImplementedError

    def gen_pattern(self):
        raise NotImplementedError


class HarmonicsDisk:
    def __init__(self):
        raise NotImplementedError

    def gen_pattern(self):
        raise NotImplementedError


class RandomUniform:
    def __init__(self, size=(32,), seed=0, min=-1.0, max=1.0):

        np.random.seed(seed)
        self.name = "random-uniform"
        self.size = size
        self.min = min
        self.max = max

    def gen_pattern(self):
        return np.random.uniform(self.min, self.max, size=self.size)


class RandomNormal:
    def __init__(self, size=(32,), seed=0, mean=0.0, std=1.0):

        np.random.seed(seed)
        self.name = "random-normal"
        self.size = size
        self.mean = mean
        self.std = std

    def gen_pattern(self):
        return np.random.normal(loc=self.mean, scale=self.std, size=self.size)
