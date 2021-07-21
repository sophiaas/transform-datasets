import numpy as np
import torch
from scipy import ndimage
from utils import translate1d, translate2d
import skimage


class CyclicTranslation1D:
    def __init__(self, fraction_transforms):
        self.fraction_transforms = fraction_transforms

    def __call__(self, data):
        assert len(data.shape) == 2, "Data must have shape (n_datapoints, dim)"

        transformed_data, transforms = [], []
        dim = data.shape[-1]
        all_transforms = np.arange(dim)
        n_transforms = int(self.fraction_transforms * len(all_transforms))
        self.orbit_size = n_transforms
        for x in data:
            transforms = np.random.sample(
                all_transforms, size=n_transforms, replace=False
            )
            transforms = sorted(transforms)
            for t in transforms:
                xt = translate1d(x, t)
                transformed_data.append(xt)
                transforms.append(t)
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        return transformed_data, transforms


class CyclicTranslation2D:
    def __init__(self, fraction_transforms):
        self.fraction_transforms = fraction_transforms

    def __call__(self):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, transforms = [], []
        dim_v, dim_h = data.shape[-2:]
        all_transforms = list(
            itertools.product(
                np.arange(dim_v),
                np.arange(dim_h),
            )
        )
        n_transforms = int(self.fraction_transforms * len(all_transforms))
        self.orbit_size = n_transforms
        for x in data:
            transforms = np.random.sample(
                np.arange(n_transforms), size=n_transforms, replace=False
            )
            transforms = all_transforms[sorted(transforms)]
            for tv, th in transforms:
                xt = translate2d(x, tv, th)
                transformed_data.append(xt)
                transforms.append((tv, th))
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        return transformed_data, transforms


class GaussianBlur:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        transformed_data, transforms = [], []
        for x in data:
            xt = ndimage.gaussian_filter(x, sigma=self.sigma)
            transformed_data.append(xt)
            transforms.append(self.sigma)
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        return transformed_data, transforms


class SO2:
    def __init__(self, fraction_transforms):
        self.fraction_transforms = fraction_transforms

    def __call__(self, data):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, transforms = [], []
        all_transforms = np.arange(360)
        n_transforms = int(self.fraction_transforms * len(all_transforms))
        for x in data:
            transforms = np.random.sample(
                all_transforms, size=n_transforms, replace=False
            )
            transforms = sorted(transforms)
            for t in transforms:
                xt = skimage.transform.rotate(x, t)
                transformed_data.append(xt)
                transforms.append(t)
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        return transformed_data, transforms


class SO3:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class C4:
    def __call__(self, data):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, transforms = [], []
        transforms = np.arange(4)
        for x in data:
            for t in transforms:
                xt = np.rot90(x, t)
                transformed_data.append(xt)
                transforms.append(t)
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        return transformed_data, transforms


class Scaling:
    def __init__(self, min=0.5, max=1.0, n_transforms=10):
        self.n_transforms = n_transforms
        self.min = min
        self.max = max

    def __call__(self, data):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, transforms = [], []
        transforms = np.linspace(self.min, self.max, self.n_transforms)
        for x in data:
            for t in transforms:
                xt = rescale(x, t, data.shape[-1])
                transformed_data.append(x)
                transforms.append(t)
        transformed_data = torch.tensor(transformed_data)
        transforms = torch.tensor(transforms)
        return transformed_data, transforms


class HierarchicalReflection:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class CircleCrop:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, data):
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

        data[~circle] = 0.0
        transformed_data = torch.tensor(data)
        transforms = torch.zeros(len(data))

        return transformed_data, transforms
