import numpy as np
import torch
from scipy import ndimage
from transform_datasets.transforms.functional import translate1d, translate2d, rescale
from skimage.transform import rotate
import pyshtools as pysh
import itertools
from collections import OrderedDict
from cplxmodule.cplx import Cplx


class Transform:
    def __init__(self):
        self.name = None

    def define_containers(self, tlabels):
        transformed_data, transforms, new_labels = [], [], []
        new_tlabels = OrderedDict({k: [] for k in tlabels.keys()})
        return transformed_data, new_labels, new_tlabels, transforms

    def reformat(self, transformed_data, new_labels, new_tlabels, transforms):
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        new_labels = torch.tensor(new_labels)
        for k in new_tlabels.keys():
            new_tlabels[k] = torch.stack(new_tlabels[k])
        return transformed_data, new_labels, new_tlabels, transforms
    
    
class Permutation(Transform):
    def __init__(self,
                 percent_transformations=0.1):
        super().__init__()
        self.name = "permutation"
        self.percent_transformations = percent_transformations
        
    def __call__(self, data, labels, tlabels):
        if len(data.shape) != 3:
            raise ValueError("Data must be (k, n, m)")
        dim = data.shape[1]
        all_permutations = np.math.factorial(dim)
        select_permutations = int(all_permutations * self.percent_transformations)
        transformed_data = []
        all_perms = []
        for mat in data:
            # Permute matrices
            for i in range(select_permutations): 
                # NB: Generates a random permutation each time, may be redundant
                perm = np.random.permutation(dim)
                permuted = mat[perm][:, perm]
                transformed_data.append(permuted) 
                all_perms.append(perm)
        transformed_data = torch.stack(transformed_data)
        return transformed_data, labels, tlabels, all_perms
    
class CenterMean(Transform):
    def __init__(self):
        super().__init__()
        self.name = "center-mean"
        
    def __call__(self, data, labels, tlabels):
        if len(data.shape) == 2:
            axis = -1
        elif len(data.shape) == 3:
            axis = (-1, -2)
        else:
            raise ValueError("Operation is not defined for data of dimension {}".format(len(data.shape)))
        means = data.mean(axis=axis, keepdims=True)
        transformed_data = data - means
        return transformed_data, labels, tlabels, means
    
class UnitStd(Transform):
    def __init__(self):
        super().__init__()
        self.name = "unit-std"
        
    def __call__(self, data, labels, tlabels):
        if len(data.shape) == 2:
            axis = -1
        elif len(data.shape) == 3:
            axis = (-1, -2)
        else:
            raise ValueError("Operation is not defined for data of dimension {}".format(len(data.shape)))
        stds = data.std(axis=axis, keepdims=True)
        transformed_data = data / stds
        return transformed_data, labels, tlabels, stds


class UniformNoise(Transform):
    def __init__(self, magnitude=1.0, n_samples=10):
        super().__init__()
        self.name = "uniform-noise"
        self.magnitude = magnitude
        self.n_samples = n_samples

    def __call__(self, data, labels, tlabels):
        size = data.shape[1:]
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )
        for i, x in enumerate(data):
            for j in range(self.n_samples):
                noise = np.random.uniform(-self.magnitude, self.magnitude, size=size)
                xt = x.numpy().copy() + noise
                transformed_data.append(xt)
                transforms.append(self.magnitude)
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms
    
    
class GaussianNoise(Transform):
    def __init__(self, loc=0.0, scale=1.0, n_samples=10):
        super().__init__()
        self.name = "gaussian-noise"
        self.loc = loc
        self.scale = scale
        self.n_samples = n_samples

    def __call__(self, data, labels, tlabels):
        size = data.shape[1:]
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )
        for i, x in enumerate(data):
            for j in range(self.n_samples):
                noise = np.random.normal(loc=self.loc, scale=self.scale, size=size)
                xt = x.numpy().copy() + noise
                transformed_data.append(xt)
                transforms.append((self.loc, self.scale))
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms
    
    
class VonMisesNoise(Transform):
    
    """
    Assumes the data it is applied to is complex.
    """
    
    def __init__(self, mu=0.0, kappa=10.0, n_samples=10):
        super().__init__()
        self.name = "von-mises-noise"
        self.mu = mu
        self.kappa = kappa
        self.n_samples = n_samples
        
    def __call__(self, data, labels, tlabels):
        size = data.shape[1:]
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )
        for i, x in enumerate(data):
            for j in range(self.n_samples):
                noise = np.random.vonmises(mu=self.mu, kappa=self.kappa, size=x.shape)
                xt = x * np.exp(1j * noise)
                transformed_data.append(xt.numpy())
                transforms.append((self.mu, self.kappa))
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms
    
    
class Fourier1D(Transform):
    def __init__(self):
        super().__init__()
        self.name = "fourier-1d"

    def __call__(self, data, labels, tlabels):
        transformed_data = torch.fft.fft(data)
        transforms = torch.zeros(len(transformed_data))
        new_labels = labels
        new_tlabels = tlabels
        return transformed_data, new_labels, new_tlabels, transforms
    
    
class Fourier2D(Transform):
    def __init__(self):
        super().__init__()
        self.name = "fourier-2d"

    def __call__(self, data, labels, tlabels):
        transformed_data = torch.fft.fft2(data)
        transforms = torch.zeros(len(transformed_data))
        new_labels = labels
        new_tlabels = tlabels
        return transformed_data, new_labels, new_tlabels, transforms
    
    
class PhaseRotation(Transform):
    
    """
    Assumes the data it is applied to is in complex Fourier space.
    """
    
    def __init__(self, n_transformations):
        super().__init__()
        self.name = "phase-rotation"
        self.n_transformations = n_transformations
        
        
    def __call__(self, data, labels, tlabels):
        dim = data.shape[-1]
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

        frequencies = np.fft.fftfreq(dim, d=1/dim)        
        shifts = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_transformations)
        
        for i, x in enumerate(data):
            for s in shifts:
                transformed_data.append(x.numpy() * np.exp(1j * frequencies * s))
                transforms.append(s)
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms
    
    
class PhaseRotation2D(Transform):
    
    """
    Assumes the data it is applied to is 2D and in complex Fourier space.
    """
    
    def __init__(self, n_transformations):
        super().__init__()
        self.name = "phase-rotation"
        self.n_transformations = n_transformations
        
        
    def __call__(self, data, labels, tlabels):
        img_size = data.shape[1:]
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

#         frequencies_r = np.fft.fftfreq(img_size[0], d=1/img_size[0])
#         frequencies_c = np.fft.fftfreq(img_size[1], d=1/img_size[1])
        freqs = np.array(list(itertools.product(np.fft.fftfreq(img_size[0], d=1/img_size[0]), np.fft.fftfreq(img_size[1], d=1/img_size[1])))).reshape(img_size+(2,))
        shifts_r = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_transformations)
        shifts_c = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_transformations)
        shifts = list(itertools.product(shifts_r, shifts_c))
        
        for i, x in enumerate(data):
            for s in shifts:
                transformed_data.append(x.numpy() * np.exp(1j * (freqs[:, :, 0] * s[0] + freqs[:, :, 1] * s[1])))
                transforms.append(s)
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms
    
    
class Bispectrum1DLabels(Transform):

    def __init__(self, normalize=True):
        super().__init__()
        self.name = "bispectrum-1d-labels"     
        self.normalize = normalize
        
    def __call__(self, data, labels, tlabels):
        img_size = data.shape[1:]
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )
        
        n = data.shape[-1]
        
        FT = torch.fft.fft(data)
        FT = FT.reshape((FT.shape[0], -1))
        
        rolled_real = torch.stack([torch.roll(FT.real, -i, dims=1) for i in range(n)])
        rolled_imag = torch.stack([torch.roll(FT.imag, -i, dims=1) for i in range(n)])
      
        t = rolled_real + 1j * rolled_imag
        t = t.permute(1, 0, -1)
        FT = FT.unsqueeze(-1)
        BS = (FT @ FT.permute(0, -1, 1) * t.conj())
        BS = BS.reshape(BS.shape[0], -1)

        new_labels = torch.cat([BS.real, BS.imag], axis=-1).float()
        if self.normalize:
            new_labels -= new_labels.mean(axis=(0, 1), keepdims=True)
            new_labels /= new_labels.std(axis=(0, 1), keepdims=True)
        new_tlabels = tlabels
        transformed_data = data
        transforms = torch.zeros(len(transformed_data))
        
        return transformed_data, new_labels, new_tlabels, transforms
    
    
class UnitNorm(Transform):
    def __init__(self, axis=-1):
        super().__init__()
        self.name = "unit-norm"
        self.axis = axis

    def __call__(self, data, labels, tlabels):
        transformed_data = data / (torch.linalg.norm(data, axis=self.axis, keepdims=True) + 1e-10)
        transforms = torch.zeros(len(transformed_data))
        new_labels = labels
        new_tlabels = tlabels
        return transformed_data, new_labels, new_tlabels, transforms
    
class UnitMagnitude(Transform):
    def __init__(self):
        super().__init__()
        self.name = "unit-magnitude"

    def __call__(self, data, labels, tlabels):
        transformed_data = data / abs(data) #TODO: Maybe add epsilon to avoid div by zero?
        transforms = torch.zeros(len(transformed_data))
        new_labels = labels
        new_tlabels = tlabels
        return transformed_data, new_labels, new_tlabels, transforms
    

class CyclicTranslation1D(Transform):
    def __init__(self, fraction_transforms=0.1, sample_method="linspace"):
        super().__init__()
        assert sample_method in [
            "linspace",
            "random",
        ], "sample_method must be one of ['linspace', 'random']"
        self.fraction_transforms = fraction_transforms
        self.sample_method = sample_method
        self.name = "cyclic-translation-1d"

    def get_samples(self, dim):
        n_transforms = int(self.fraction_transforms * dim)
        if self.sample_method == "linspace":
            return [int(x) for x in np.linspace(0, dim - 1, n_transforms)]
        else:
            select_transforms = np.random.choice(
                np.arange(dim), size=n_transforms, replace=False
            )
            select_transforms = sorted(select_transforms)
            return select_transforms

    def __call__(self, data, labels, tlabels):
        assert len(data.shape) == 2, "Data must have shape (n_datapoints, dim)"

        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

        dim = data.shape[-1]
        select_transforms = self.get_samples(dim)
        for i, x in enumerate(data):
            if self.sample_method == "random" and self.fraction_transforms != 1.0:
                select_transforms = self.get_samples(dim)
            for t in select_transforms:
                xt = translate1d(x, t)
                transformed_data.append(xt)
                transforms.append(t)
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms


class CyclicTranslation2D(Transform):
    def __init__(self, fraction_transforms=0.1, sample_method="linspace"):
        super().__init__()
        assert sample_method in [
            "linspace",
            "random",
        ], "sample_method must be one of ['linspace', 'random']"
        self.fraction_transforms = fraction_transforms
        self.sample_method = sample_method
        self.name = "cyclic-translation-2d"

    def get_samples(self, dim_v, dim_h):
        n_transforms = int(self.fraction_transforms * dim_v * dim_h)
        if self.sample_method == "linspace":
            return [
                (int(v), int(h))
                for v, h in zip(
                    np.linspace(0, dim_v - 1, n_transforms),
                    np.linspace(0, dim_h - 1, n_transforms),
                )
            ]
        else:
            all_transforms = list(
                itertools.product(
                    np.arange(dim_v),
                    np.arange(dim_h),
                )
            )
            select_transforms_idx = np.random.choice(
                range(n_transforms), size=n_transforms, replace=False
            )
            select_transforms = [
                all_transforms[x] for x in sorted(select_transforms_idx)
            ]
            return select_transforms

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

        dim_v, dim_h = data.shape[-2:]
        select_transforms = self.get_samples(dim_v, dim_h)
        for i, x in enumerate(data):
            if self.sample_method == "random" and self.fraction_transforms != 1.0:
                select_transforms = self.get_samples(dim_v, dim_h)
            for tv, th in select_transforms:
                xt = translate2d(x, tv, th)
                transformed_data.append(xt)
                transforms.append((tv, th))
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms


class GaussianBlur(Transform):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma
        self.name = "gaussian-blur"

    def __call__(self, data, labels, tlabels):
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

        for i, x in enumerate(data):
            xt = ndimage.gaussian_filter(x, sigma=self.sigma)
            transformed_data.append(xt)
            transforms.append(self.sigma)
            new_labels.append(labels[i])
            for k in new_tlabels.keys():
                new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms


class SO2(Transform):
    def __init__(self, fraction_transforms=0.1, sample_method="linspace"):
        super().__init__()
        assert sample_method in [
            "linspace",
            "random",
        ], "sample_method must be one of ['linspace', 'random']"
        self.fraction_transforms = fraction_transforms
        self.sample_method = sample_method
        self.name = "so2"

    def get_samples(self):
        n_transforms = int(self.fraction_transforms * 360)
        if self.sample_method == "linspace":
            return np.linspace(0, 359, n_transforms)
        else:
            select_transforms = np.random.choice(
                np.arange(360), size=n_transforms, replace=False
            )
            select_transforms = sorted(select_transforms)
            return select_transforms

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

        select_transforms = self.get_samples()
        for i, x in enumerate(data):
            if self.sample_method == "random":
                select_transforms = self.get_samples()
            for t in select_transforms:
                xt = rotate(x, t)
                transformed_data.append(xt)
                transforms.append(t)
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms


class SO3(Transform):
    def __init__(
        self, n_samples=10, grid_type="GLQ", sample_method="linspace"
    ):
        """
        TODO: Currently encountering a bug when input is complex
        """
        super().__init__()
        assert sample_method in [
            "linspace",
            "random",
        ], "sample_method must be one of ['linspace', 'random']"
        self.n_samples = n_samples
        self.grid_type = grid_type
        self.sample_method = sample_method
        self.name = "so3"

    def get_samples(self):
        if self.sample_method == "linspace":
            samples_per_axis = int(np.cbrt(self.n_samples))
#             alpha = np.arange(0, 360, 360 / samples_per_axis)
            alpha = np.arange(0, 360, 360 / self.n_samples)
            beta = np.arange(0, 1)
            gamma = np.arange(0, 1)
#             beta = np.arange(0, 180, 180 / samples_per_axis)
#             gamma = np.arange(0, 360, 360 / samples_per_axis)
            select_transforms = list(itertools.product(alpha, beta, gamma))
            return select_transforms

        else:
            alpha = np.random.uniform(0, 360, size=self.n_samples)
            beta = np.random.uniform(0, 180, size=self.n_samples)
            gamma = np.random.uniform(0, 360, size=self.n_samples)
            select_transforms = list(zip(alpha, beta, gamma))
            return select_transforms

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

        select_transforms = self.get_samples()
        for i, x in enumerate(data):
            if self.sample_method == "random":
                select_transforms = self.get_samples()
            for t in select_transforms:
                grid = pysh.SHGrid.from_array(x.numpy(), grid=self.grid_type)
                coeffs = grid.expand()
                coeffs_t = coeffs.rotate(t[0], t[1], t[2])
                xt = coeffs_t.expand(grid=self.grid_type)
                transformed_data.append(xt.data)
                transforms.append(t)
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms


class C4(Transform):
    def __init__(self):
        super().__init__()
        self.name = "c4"

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )
        all_transforms = np.arange(4)
        for i, x in enumerate(data):
            for t in all_transforms:
                xt = np.rot90(x, t)
                transformed_data.append(xt)
                transforms.append(t)
                new_labels.append(labels[i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms


class Scaling(Transform):
    def __init__(
        self,
        range_min=0.5,
        range_max=1.0,
        n_samples=10,
        sample_method="linspace",
    ):
        super().__init__()
        assert sample_method in [
            "linspace",
            "random",
        ], "sample_method must be one of ['linspace', 'random']"
        self.n_samples = n_samples
        self.range_min = range_min
        self.range_max = range_max
        self.sample_method = sample_method
        self.name = "scaling"

    def get_samples(self):
        if self.sample_method == "linspace":
            return np.linspace(self.range_min, self.range_max, self.n_samples)
        else:
            return sorted(
                np.random.uniform(self.range_min, self.range_max, size=self.n_samples)
            )

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )
        select_transforms = self.get_samples()
        for i, x in enumerate(data):
            if self.sample_method == "random":
                select_transforms = self.get_samples()
            for t in select_transforms:
                xt = rescale(x, t, data.shape[-1])
                transformed_data.append(xt)
                transforms.append(t)
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms


class CircleCrop(Transform):
    def __init__(self):
        super().__init__()
        self.name = "circle-crop"

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        img_size = data.shape[1:]

        v, h = np.mgrid[: img_size[0], : img_size[1]]
        equation = (v - ((img_size[0] - 1) / 2)) ** 2 + (
            h - ((img_size[1] - 1) / 2)
        ) ** 2
        circle = equation < (equation.max() / 2)

        transformed_data = data.clone()
        transformed_data[:, ~circle] = 0.0
        transforms = torch.zeros(len(data))

        return transformed_data, labels, tlabels, transforms

    
class Ravel(Transform):
    def __init__(self):
        super().__init__()
        self.name = 'ravel'
        
    def __call__(self, data, labels, tlabels):
        transformed_data = data.reshape(data.shape[0], -1)
        transforms = torch.zeros(len(data))
        return transformed_data, labels, tlabels, transforms
    

class HierarchicalReflection:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError
