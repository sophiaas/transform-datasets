import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from scipy import ndimage
from skimage import transform


def split_dataset(dataset, 
                  batch_size,
                  validation_split=0.2, 
                  seed=0):
    
    validation_split = 0.2

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=batch_size, 
                                               sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=batch_size,
                                                    sampler=valid_sampler)
    
    return train_loader, validation_loader

 
def compute_disk_pixels(img_size):
    if img_size[0] % 2 == 0 or img_size[1] % 2 == 0:
        raise ValueError(
            "Image size should be a tuple of odd numbers to ensure centered rotational symmetry."
        )
    v, h = np.mgrid[: img_size[0], : img_size[1]]
    equation = (v - ((img_size[0] - 1) / 2)) ** 2 + (
        h - ((img_size[1] - 1) / 2)
    ) ** 2
    circle = equation < (equation.max() / 2)
    return circle


def translate(img, x=0, y=0):
    """
    Given an image and offset x, y, returns a translated image up in y and right in x
    """
    new_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            oldi = (i + y) % img.shape[0]
            oldj = (j - x) % img.shape[1]
            new_img[i, j] = img[oldi, oldj]
    return new_img

def rescale(x, scale, img_size):
    compute_shift = lambda img_size, scale: (img_size/2) - (img_size*scale/2)
    gen_transform = lambda scale, shift: transform.AffineTransform(scale=scale, translation=(shift,shift))
    return transform.warp(x, gen_transform(scale=scale, shift=compute_shift(img_size,scale)).inverse)


def gen_random_data(img_size, n_classes, group_type, n_transformations, blur_parameter=1, percent_translations=0.1, scale_range=(1,0.5)):
    
    X = np.random.uniform(-1, 1, size=(n_classes, img_size[0], img_size[1]))
    
    if group_type == 'rotation':
        X = np.array([ndimage.gaussian_filter(xi, sigma=blur_parameter) for xi in X])
        in_circle = compute_disk_pixels(img_size)
        X[:,~in_circle] = 0
        
        thetas = np.linspace(0, 360, n_transformations)
        # Remember that the outer loop is the 'inner loop'
        X = np.array([transform.rotate(xi, th) for xi in X for th in thetas])
        
    elif group_type == 'translation':
        step_size = int(percent_translations*img_size[0])
        transformations = list(
            itertools.product(
                np.arange(0, img_size[0], step_size),
                np.arange(0, img_size[0], step_size),
            )
        )
        X = np.array([translate(xi, x=xx, y=yy) for xi in X for (xx,yy) in transformations])
        X = np.array([ndimage.gaussian_filter(xi, sigma=blur_parameter) for xi in X])
        
    elif group_type == 'scaling':
        X = np.array([ndimage.gaussian_filter(xi, sigma=blur_parameter) for xi in X])
        scales = np.linspace(scale_range[0], scale_range[1], n_transformations)
        X = np.array([rescale(xi, sci, img_size[0]) for xi in X for sci in scales])

    X -= np.mean(X, axis=(1, 2), keepdims=True)
    X /= np.std(X, axis=(1, 2), keepdims=True)
    n_transformations_from_X = X.shape[0] // n_classes
    Y = np.concatenate([np.array([i]*n_transformations_from_X) for i in range(n_classes)])
    
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    return X,Y