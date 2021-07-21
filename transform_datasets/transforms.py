import numpy as np
import torch
from scipy import ndimage
from transform_datasets.utils import translate1d, translate2d, rescale
from skimage.transform import rotate
import pyshtools as pysh
import itertools


class UniformNoise:
    def __init__(self, 
                 seed=0, 
                 magnitude=1.0, 
                 n_samples=10):
        super().__init__()
        np.random.seed(seed)
        self.name = "random-uniform"
        self.magnitude = magnitude
        self.n_samples = n_samples

    def __call__(self, data, labels):
        size = data.shape[1:]
        transformed_data, transforms, new_labels = [], [], []
        for i, x in enumerate(data):
            for j in range(self.n_samples):
                noise = torch.tensor(np.random.uniform(-self.magnitude, self.magnitude, size=size))
                xt = x + noise
                transformed_data.append(xt)
                transforms.append(self.magnitude)
                new_labels.append(labels[i])     
        transformed_data = torch.stack(transformed_data)
        transforms = torch.tensor(transforms)
        new_labels = torch.tensor(new_labels)
        return transformed_data, new_labels, transforms
    

class CyclicTranslation1D:
    def __init__(self, fraction_transforms=0.1):
        self.fraction_transforms = fraction_transforms
        self.name = 'cyclic-translation-1d'

    def __call__(self, data, labels):
        assert len(data.shape) == 2, "Data must have shape (n_datapoints, dim)"

        transformed_data, transforms, new_labels = [], [], []
        dim = data.shape[-1]
        all_transforms = np.arange(dim)
        n_transforms = int(self.fraction_transforms * len(all_transforms))
        self.orbit_size = n_transforms
        for i, x in enumerate(data):
            select_transforms = np.random.choice(
                all_transforms, size=n_transforms, replace=False
            )
            select_transforms = sorted(select_transforms)
            for t in select_transforms:
                xt = translate1d(x, t)
                transformed_data.append(xt)
                transforms.append(t)
                new_labels.append(labels[i])             
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        new_labels = torch.tensor(new_labels)
        return transformed_data, new_labels, transforms


class CyclicTranslation2D:
    def __init__(self, fraction_transforms=0.1):
        self.fraction_transforms = fraction_transforms
        self.name = 'cyclic-translation-2d'

    def __call__(self, data, labels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, transforms, new_labels = [], [], []
        dim_v, dim_h = data.shape[-2:]
        all_transforms = list(
            itertools.product(
                np.arange(dim_v),
                np.arange(dim_h),
            )
        )
        n_transforms = int(self.fraction_transforms * len(all_transforms))
        self.orbit_size = n_transforms
        for i, x in enumerate(data):
            select_transforms_idx = np.random.choice(
                range(n_transforms), size=n_transforms, replace=False
            )
            select_transforms = [all_transforms[x] for x in sorted(select_transforms_idx)]
            for tv, th in select_transforms:
                xt = translate2d(x, tv, th)
                transformed_data.append(xt)
                transforms.append((tv, th))
                new_labels.append(labels[i])             
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        new_labels = torch.tensor(new_labels)
        return transformed_data, new_labels, transforms


class GaussianBlur:
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = 'gaussian-blur'

    def __call__(self, data, labels):
        transformed_data, transforms, new_labels = [], [], []
        for i, x in enumerate(data):
            xt = ndimage.gaussian_filter(x, sigma=self.sigma)
            transformed_data.append(xt)
            transforms.append(self.sigma)
            new_labels.append(labels[i])
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        new_labels = torch.tensor(new_labels)
        return transformed_data, new_labels, transforms


class SO2:
    def __init__(self, fraction_transforms=0.1):
        self.fraction_transforms = fraction_transforms
        self.name = 'so2'

    def __call__(self, data, labels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, transforms, new_labels = [], [], []
        all_transforms = np.arange(360)
        n_transforms = int(self.fraction_transforms * len(all_transforms))
        for i, x in enumerate(data):
            select_transforms = np.random.choice(
                all_transforms, size=n_transforms, replace=False
            )
            select_transforms = sorted(select_transforms)
            for t in select_transforms:
                xt = rotate(x, t)
                transformed_data.append(xt)
                transforms.append(t)
                new_labels.append(labels[i])
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        new_labels = torch.tensor(new_labels)
        return transformed_data, new_labels, transforms


class SO3:
    def __init__(self, n_axis_rotations=10, grid_type="GLQ"):
        self.n_axis_rotations = n_axis_rotations
        self.grid_type = grid_type
        self.name = 'so3'

        self.alpha = np.arange(0, 360, 360 / n_axis_rotations)
        self.beta = np.arange(0, 180, 180 / n_axis_rotations)
        self.gamma = np.arange(0, 360, 360 / n_axis_rotations)

    def __call__(self, data, labels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"
        transformed_data, transforms, new_labels = [], [], []
        select_transforms = list(itertools.product(self.alpha, self.beta, self.gamma))
        for i, x in enumerate(data):
            for t in select_transforms:
                grid = pysh.SHGrid.from_array(x.numpy(), grid=self.grid_type)
                coeffs = grid.expand()
                coeffs_t = coeffs.rotate(t[0], t[1], t[2])
                xt = coeffs_t.expand(grid=self.grid_type)
                transformed_data.append(xt.data)
                transforms.append(t)
                new_labels.append(labels[i])
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        new_labels = torch.tensor(new_labels)
        return transformed_data, new_labels, transforms


class C4:
    def __init__(self):
        self.name = 'c4'
        
    def __call__(self, data, labels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, transforms, new_labels = [], [], []
        all_transforms = np.arange(4)
        for i, x in enumerate(data):
            for t in all_transforms:
                xt = np.rot90(x, t)
                transformed_data.append(xt)
                transforms.append(t)
                new_labels.append(labels[i])
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        new_labels = torch.tensor(new_labels)
        return transformed_data, new_labels, transforms


class Scaling:
    def __init__(self, range_min=0.5, range_max=1.0, n_transforms=10):
        self.n_transforms = n_transforms
        self.range_min = range_min
        self.range_max = range_max
        self.name = 'scaling'

    def __call__(self, data, labels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, transforms, new_labels = [], [], []
        select_transforms = np.linspace(self.range_min, self.range_max, self.n_transforms)
        for i, x in enumerate(data):
            for t in select_transforms:
                xt = rescale(x, t, data.shape[-1])
                transformed_data.append(xt)
                transforms.append(t)
                new_labels.append(labels[i])
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        new_labels = torch.tensor(new_labels)
        return transformed_data, new_labels, transforms


class CircleCrop:
    def __init__(self):
        self.name = 'circle-crop'

    def __call__(self, data, labels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        img_size = data.shape[1:]

        assert (img_size[0] % 2 != 0) and (
            img_size[1] % 2 != 0
        ), "Image size should be a tuple of odd numbers to ensure centered rotational symmetry."

        v, h = np.mgrid[: img_size[0], : img_size[1]]
        equation = (v - ((img_size[0] - 1) / 2)) ** 2 + (
            h - ((img_size[1] - 1) / 2)
        ) ** 2
        circle = equation < (equation.max() / 2)
        
        transformed_data = data.clone()
        transformed_data[:, ~circle] = 0.0
        transforms = torch.zeros(len(data))

        return transformed_data, labels, transforms


class HierarchicalReflection:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError
