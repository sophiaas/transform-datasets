import numpy as np
import torch


class CyclicTranslation1D:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class CyclicTranslation2D:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class GaussianBlur:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self):
        raise NotImplementedError


class SO2:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class SO3:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class C4:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class Scaling:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class HierarchicalReflection:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class CircleCrop:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError
