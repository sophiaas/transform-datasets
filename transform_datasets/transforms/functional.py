import numpy as np
import skimage


def translate1d(x, t):
    """
    Given a 1D signal x and translation t, returns a signal translated cyclically by t positions.
    """
    new_x = list(x)
    for i in range(t):
        last = new_x.pop()
        new_x = [last] + new_x
    return np.array(new_x)


def translate2d(img, v=0, h=0):
    """
    Given an image and offset v, h returns a cyclically translated image up in v and right in h.
    """
    new_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            oldi = (i - v) % img.shape[0]
            oldj = (j - h) % img.shape[1]
            new_img[i, j] = img[oldi, oldj]
    return new_img


def rescale(x, scale, img_size):
    """
    Given...
    """
    compute_shift = lambda img_size, scale: (img_size / 2) - (img_size * scale / 2)
    gen_transform = lambda scale, shift: skimage.transform.AffineTransform(
        scale=scale, translation=(shift, shift)
    )
    return skimage.transform.warp(
        x, gen_transform(scale=scale, shift=compute_shift(img_size, scale)).inverse,  mode='edge'
    )

def circle_crop(data):
    """
    assumes input is batch x height x width
    """
    img_size = (data.shape[-2], data.shape[-1])

    v, h = np.mgrid[: img_size[0], : img_size[1]]
    equation = (v - ((img_size[0] - 1) / 2)) ** 2 + (
        h - ((img_size[1] - 1) / 2)
    ) ** 2
    circle = equation < (equation.max() / 2)

    transformed_data = data.clone()
    transformed_data[:, ~circle] = 0.0
    return transformed_data