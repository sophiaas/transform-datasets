import numpy as np
import torch
from scipy import ndimage

from transform_datasets.transforms.functional import rescale
from transform_datasets.transforms.groups import Commutative
from skimage.transform import rotate, resize
import itertools
from collections import OrderedDict
from cplxmodule.cplx import Cplx
import copy


class Transform:
    def __init__(self):
        self.name = None

    def define_containers(self, tlabels):
        transformed_data, transforms, new_labels = [], [], []
        new_tlabels = OrderedDict({k: [] for k in tlabels.keys()})
        return transformed_data, new_labels, new_tlabels, transforms

    def reformat(self, transformed_data, new_labels, new_tlabels, transforms):
        try:
            transformed_data = torch.stack(transformed_data)
        except:
            transformed_data = torch.tensor(np.array(transformed_data))
        transforms = torch.tensor(transforms)
        # new_labels = torch.tensor(new_labels)
        new_labels = torch.stack(new_labels)
        for k in new_tlabels.keys():
            new_tlabels[k] = torch.stack(new_tlabels[k])
        return transformed_data, new_labels, new_tlabels, transforms


class Permutation(Transform):
    """
    Has bugs, needs to be fixed
    """

    def __init__(self, fraction_transforms=0.1):
        super().__init__()
        self.name = "permutation"
        self.fraction_transforms = fraction_transforms

    def __call__(self, data, labels, tlabels):
        if len(data.shape) != 3:
            raise ValueError("Data must be (k, n, m)")
        dim = data.shape[1]
        all_permutations = np.math.factorial(dim)
        select_permutations = int(all_permutations * self.fraction_transforms)
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )
        for i, mat in enumerate(data):
            # Permute matrices
            for j in range(select_permutations):
                # NB: Generates a random permutation each time, may be redundant
                perm = np.random.permutation(dim)
                permuted = mat[perm][:, perm]
                transformed_data.append(permuted)
                transforms.append(perm)
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])
        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, labels, tlabels, all_perms


class CenterMean(Transform):
    def __init__(self, samplewise=True):
        super().__init__()
        self.name = "center-mean"
        self.samplewise = samplewise

    def __call__(self, data, labels, tlabels):
        if self.samplewise:
            if len(data.shape) == 2:
                axis = -1
            elif len(data.shape) == 3:
                axis = (-1, -2)
            else:
                raise ValueError(
                    "Operation is not defined for data of dimension {}".format(
                        len(data.shape)
                    )
            )
            means = data.mean(dim=axis, keepdim=True)
        else:
            means = data.mean()
        transformed_data = data - means
        if not self.samplewise:
            means = torch.tile(means, (len(data),))
        return transformed_data, labels, tlabels, means


class UnitStd(Transform):
    def __init__(self, samplewise=True):
        super().__init__()
        self.name = "unit-std"
        self.samplewise = samplewise

    def __call__(self, data, labels, tlabels):
        if self.samplewise:
            if len(data.shape) == 2:
                axis = -1
            elif len(data.shape) == 3:
                axis = (-1, -2)
            else:
                raise ValueError(
                    "Operation is not defined for data of dimension {}".format(
                        len(data.shape)
                    )
                )
            stds = data.std(dim=axis, keepdim=True)
        else:
            stds = data.std()
        transformed_data = data / (stds + 1e-10)
        if not self.samplewise:
            stds = torch.tile(stds, (len(data),))
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

    def __init__(self, mu=0.0, kappa=10.0, n_samples=1):
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
                transformed_data.append(xt)
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


class Phase(Transform):
    def __init__(self):
        super().__init__()
        self.name = "phase"

    def __call__(self, data, labels, tlabels):
        transformed_data = np.angle(data)
        transforms = torch.zeros(len(transformed_data))
        new_labels = labels
        new_tlabels = tlabels
        return transformed_data, new_labels, new_tlabels, transforms


class Phasor(Transform):
    def __init__(self):
        super().__init__()
        self.name = "phase"

    def __call__(self, data, labels, tlabels):
        transformed_data = np.exp(1j * np.angle(data))
        transforms = torch.zeros(len(transformed_data))
        new_labels = labels
        new_tlabels = tlabels
        return transformed_data, new_labels, new_tlabels, transforms


class WindowDelete(Transform):
    def __init__(self, window_size=(10, 10)):
        super().__init__()
        self.name = "window-delete"
        self.window_size = window_size

    def __call__(self, data, labels, tlabels):
        img_size = data.shape[1:]
        start_h_idx = int(img_size[1] / 2) - int(self.window_size[1] / 2)
        start_v_idx = int(img_size[0] / 2) - int(self.window_size[0] / 2)
        end_h_idx = start_h_idx + self.window_size[1]
        end_v_idx = start_v_idx + self.window_size[0]
        transformed_data = copy.deepcopy(data)
        transformed_data[:, start_v_idx:end_v_idx, start_h_idx:end_h_idx] = 0.0
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

        frequencies = np.fft.fftfreq(dim, d=1 / dim)
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
        freqs = np.array(
            list(
                itertools.product(
                    np.fft.fftfreq(img_size[0], d=1 / img_size[0]),
                    np.fft.fftfreq(img_size[1], d=1 / img_size[1]),
                )
            )
        ).reshape(img_size + (2,))
        shifts_r = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_transformations)
        shifts_c = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_transformations)
        shifts = list(itertools.product(shifts_r, shifts_c))

        for i, x in enumerate(data):
            for s in shifts:
                transformed_data.append(
                    x.numpy()
                    * np.exp(1j * (freqs[:, :, 0] * s[0] + freqs[:, :, 1] * s[1]))
                )
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
        BS = FT @ FT.permute(0, -1, 1) * t.conj()
        BS = BS.reshape(BS.shape[0], -1)

        new_labels = torch.cat([BS.real, BS.imag], axis=-1).float()
        if self.normalize:
            new_labels -= new_labels.mean(axis=(0, 1), keepdims=True)
            new_labels /= new_labels.std(axis=(0, 1), keepdims=True)
        new_tlabels = tlabels
        transformed_data = data
        transforms = torch.zeros(len(transformed_data))

        return transformed_data, new_labels, new_tlabels, transforms


class CyclicTranslation1DLabels(Transform):
    def __init__(self, translate_by=1):
        super().__init__()
        self.name = "cyclic-translation-1d-labels"
        self.translate_by = translate_by

    def __call__(self, data, labels, tlabels):
        img_size = data.shape[1:]
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

        new_labels = torch.roll(data, self.translate_by, dims=-1)
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
        transformed_data = data / (
            torch.linalg.norm(data, axis=self.axis, keepdims=True) + 1e-10
        )
        transforms = torch.zeros(len(transformed_data))
        new_labels = labels
        new_tlabels = tlabels
        return transformed_data, new_labels, new_tlabels, transforms


class UnitMagnitude(Transform):
    def __init__(self):
        super().__init__()
        self.name = "unit-magnitude"

    def __call__(self, data, labels, tlabels):
        transformed_data = data / abs(
            data
        )  # TODO: Maybe add epsilon to avoid div by zero?
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
            unit = dim / n_transforms
            return [int(x) for x in np.arange(0, dim, unit)]
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
                xt = torch.roll(x, t)
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
        n_transforms = int(self.fraction_transforms * dim_h * dim_v)
        if self.sample_method == "linspace":
            unit_v = dim_v / n_transforms
            unit_h = dim_h / n_transforms
            return [
                (int(v), int(h))
                for v, h in zip(
                    np.arange(0, dim_v, unit_v),
                    np.arange(0, dim_h, unit_h),
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
                range(len(all_transforms)), size=n_transforms, replace=False
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
                xt = torch.roll(x, (tv, th), dims=(-2, -1))
                transformed_data.append(xt)
                transforms.append((int(tv), int(th)))
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
    def __init__(self, n=1, sample_method="linspace"):
        super().__init__()
        assert sample_method in [
            "linspace",
            "random",
        ], "sample_method must be one of ['linspace', 'random']"
        self.n = n
        self.sample_method = sample_method
        self.name = "so2"

    def get_samples(self):
        rot = 360 / self.n
        if self.sample_method == "linspace":
            rotations = np.array([rot * i for i in range(self.n)])
        else:
            rotations = np.random.choice(
                np.arange(360), size=self.n, replace=False
            )
        return rotations

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
    
    
class O2(Transform):
    def __init__(self, n=1, sample_method="linspace"):
        """
        If sample_method == "linspace", rotations will be linspaced. However, flips will be randomized.
        """
        super().__init__()
        assert sample_method in [
            "linspace",
            "random",
        ], "sample_method must be one of ['linspace', 'random']"
        self.n = n
        self.sample_method = sample_method
        self.name = "so2"
        
    def get_transforms(self):
        rot = 360 / self.n
        if self.sample_method == "linspace":
            rot_list = np.array([rot * i for i in range(self.n)])
            rotations = np.hstack([rot_list, rot_list])
            flips = np.hstack([np.zeros(self.n), np.ones(self.n)])
        else:
            rotations = np.random.choice(
                np.arange(360), size=self.n, replace=False
            )
            flips = np.random.randint(low=0, high=2, size=(self.n,))
        return rotations, flips

    def get_flips(self):
        if self.sample_method == "linspace":
            n_transforms = int(self.n)
            flips = np.hstack([np.zeros(n_transforms), np.ones(n_transforms)])
        else:
            flips = np.random.randint(low=0, high=2, size=(n_transforms,))
        return flips

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )
        
        for i, x in enumerate(data):
            rotations, flips = self.get_transforms()
            for j in range(len(rotations)):
                if bool(flips[j]):
                    x_flip = torch.flip(x, dims=(0,))
                else:
                    x_flip = x
                xt = rotate(x_flip, rotations[j])
                transformed_data.append(xt)
                transforms.append((flips[j], rotations[j]))
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms
    
    
    
class Resize(Transform):
    
    def __init__(self, new_size):
        super().__init__()
        self.new_size = new_size
        self.name = "resize"

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

        for i, x in enumerate(data):
            transformed_data.append(resize(x, self.new_size))
            
        transformed_data = torch.tensor(np.array(transformed_data))
        transforms = torch.tensor([self.new_size] * len(transformed_data))

        return transformed_data, labels, tlabels, transforms

    
    
class SO3(Transform):
    def __init__(self, n_samples=10, grid_type="GLQ", sample_method="linspace"):
        """
        TODO: Currently encountering a bug when input is complex
        """
        import pyshtools as pysh

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
        self.name = "ravel"

    def __call__(self, data, labels, tlabels):
        transformed_data = data.reshape(data.shape[0], -1)
        transforms = torch.zeros(len(data))
        return transformed_data, labels, tlabels, transforms
    
class AddChannelDim(Transform):
    def __init__(self):
        super().__init__()
        self.name = "add-channel-dim"

    def __call__(self, data, labels, tlabels):
        transformed_data = torch.unsqueeze(data, 1)
        transforms = torch.zeros(len(data))
        return transformed_data, labels, tlabels, transforms
    
    
class ProductGroupCommutative(Transform):
    def __init__(self,
                 M=[2, 2],
                 fraction_transforms=1.0,
                 sample_method="random"):
        super().__init__()
        self.group = Commutative(M=M)
        self.fraction_transforms = fraction_transforms
        self.sample_method = sample_method
        
    def get_samples(self):
        n_transforms = int(self.fraction_transforms * self.group.n)
        if self.sample_method == "linspace":
            unit = self.group.n / n_transforms
            return [int(x) for x in np.arange(0, self.group.n, unit)]
        else:
            select_transforms = np.random.choice(
                np.arange(self.group.n), size=n_transforms, replace=False
            )
            select_transforms = sorted(select_transforms)
            return select_transforms
        
    def __call__(self, data, labels, tlabels):
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )
        select_transforms = self.get_samples()
        for i, x in enumerate(data):
            for t in select_transforms:
                g = self.group.element_at_index(t)
                xt = self.group.act(g, x)
                transformed_data.append(xt)
                transforms.append(g)
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])
                
        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms

class HierarchicalReflection:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError
        
        
class OctahedralRotation(Transform):
    
    def __init__(self,
                 full=False,
                 sample_method="random"):
        super().__init__()
        self.full = full
        self.sample_method = sample_method
        
    def all_rotations(self, polycube):
        """List all 24 rotations of the given 3d array"""
        def rotations4(polycube, axes):
            """List the four rotations of the given 3d array in the plane spanned by the given axes."""
            for i in range(4):
                 yield np.rot90(polycube, i, axes)

        # imagine shape is pointing in axis 0 (up)

        # 4 rotations about axis 0
        yield from rotations4(polycube, (1,2))

        # rotate 180 about axis 1, now shape is pointing down in axis 0
        # 4 rotations about axis 0
        yield from rotations4(np.rot90(polycube, 2, axes=(0,2)), (1,2))

        # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
        # 8 rotations about axis 2
        yield from rotations4(np.rot90(polycube, axes=(0,2)), (0,1))
        yield from rotations4(np.rot90(polycube, -1, axes=(0,2)), (0,1))

        # rotate about axis 2, now shape is pointing in axis 1
        # 8 rotations about axis 1
        yield from rotations4(np.rot90(polycube, axes=(0,1)), (0,2))
        yield from rotations4(np.rot90(polycube, -1, axes=(0,1)), (0,2))
        
    def random_rotation(self, x):
        inner_axes = [
            (0, 2),
            (0, 1)
        ]
        outer_axes = [
            (1, 2),
            (0, 1),
            (0, 2)
        ]
        inner_n = [-1, 2]

        inner_rot = np.random.randint(2)
        if bool(inner_rot):
            ax_idx = np.random.randint(2)
            n_idx = np.random.randint(2)
            x = np.rot90(x, inner_n[n_idx], axes=inner_axes[ax_idx])
        outer_n = np.random.randint(4)
        ax_idx = np.random.randint(3)
        return np.rot90(x, outer_n, outer_axes[ax_idx])
    
    def __call__(self, data, labels, tlabels):
        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )
        
        for i, x in enumerate(data):
            if self.sample_method == "random":
                xt = x
                if self.full:
                    flip = np.random.randint(2)
                    if bool(flip):
                        xt = torch.flip(xt, (0,))
                xt = self.random_rotation(xt)
                transformed_data.append(xt)
                transforms.append(0) #TODO: Return actual transform
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])
            else:
                rots = self.all_rotations(x)
                if self.full:
                    rots_flip = self.all_rotations(torch.flip(x, (0,)))
                    rots = list(rots) + list(rots_flip)
                for xt in rots:
                    transformed_data.append(xt)
                    transforms.append(0) #TODO: Return actual transform
                    new_labels.append(labels[i])
                    for k in new_tlabels.keys():
                        new_tlabels[k].append(tlabels[k][i])
                
        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms
