import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import itertools
from skimage import transform
import torchvision
import pickle
import os
from PIL import Image
from tqdm import tqdm
from itertools import product 
from scipy.special import jn, yn, jn_zeros, yn_zeros
import random

from transform_datasets.torch import utils


class SmoothedRandom(Dataset):
    def __init__(
        self,
        n_classes=100,
        img_size=65,
        group_type='rotation',
        n_transformations=50,
        blur_parameter=1,
        percent_translations=0.1,
        scale_range=(1,0.5),
        seed=0,
        ravel=True
    ):
        super().__init__()
        np.random.seed(seed)
        self.seed = seed
        self.n_classes = n_classes
        self.group_type = group_type
        self.name = "smoothed_random_{}".format(self.group_type)
        self.img_size = (img_size, img_size)
        self.img_side = img_size
        self.n_transformations = n_transformations
        self.blur_parameter=blur_parameter
        self.percent_translations=percent_translations
        self.scale_range=scale_range
        self.ravel = ravel
        
        dataset, labels = utils.gen_random_data(self.img_size, self.n_classes, self.group_type, self.n_transformations)

        if self.ravel:
            num_elements = dataset.shape[0]
            dataset = dataset.reshape(num_elements,-1)
            self.dim = self.img_side ** 2
        else:
            self.dim = self.img_side

        self.data = dataset
        self.labels = labels

   
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)

    
class DiskHarmonicPatterns(Dataset):
    '''
    Dataset of (rotated) linear combinations of disk harmonics.
    https://www.teachme.codes/scipy-tutorials/Bessel/Bessel/
    '''
    def __init__(self,
                 img_size=32, #ASSUMES SQUARE
                 n_classes=10,
                 n_rotations = 20,
                 n_basis_el=6,
                 mmax=13,
                 ravel=True,
                 seed=0):
        
        super().__init__()
        np.random.seed(seed)
        self.name = 'disk_harmonics'
        self.img_size = (img_size, img_size)
        self.img_side = img_size
        self.ravel = ravel
        if self.ravel:
            self.dim = self.img_side ** 2
        else:
            self.dim = self.img_side
        self.n_basis_el = n_basis_el
        self.mmax = mmax
        self.n_classes = n_classes
        self.n_rotations = n_rotations
        self.seed = seed
        self.gen_dataset()
    
    def disk_harmonic(self,n, m, r, theta, mmax = 12):
        """
        Calculate the displacement of the drum membrane at (r, theta; t=0)
        in the normal mode described by integers n >= 0, 0 < m <= mmax.

        """
        # Pick off the mth zero of Bessel function Jn
        k = jn_zeros(n, mmax+1)[m]
        return np.sin(n*theta) * jn(n, r*k)

    def generate_harmonic_image(self, n, m, rot, im_size=100):
        # Create arrays of cartesian co-ordinates (x, y)
        x = np.linspace(-1, 1, im_size)
        y = np.linspace(-1, 1, im_size)
        xx, yy = np.meshgrid(x, y)

        # Convert to polar coordinates (x,y) -> (r,theta)
        c = xx + 1j * yy
        polar2z = lambda r,theta: r * np.exp( 1j * theta )
        z2polar = lambda z: ( np.abs(z), np.angle(z) )
        r, theta = z2polar(c)

        # Get indices of points within the unit disk
        within_disk = np.abs(r) <=1

        theta = theta - rot
        z = self.disk_harmonic(n, m, r, theta)
        z[~within_disk] = 0

        return z, within_disk

    def generate_linear_combination(self, nm_terms, coefficients, phases, im_size=100):
        eps = 1e-7
        total_image = np.zeros((im_size,im_size))
        for (n,m),coeff,phase in zip(nm_terms,coefficients,phases):
            z, idxs = self.generate_harmonic_image(n,m,phase,im_size=im_size)
            total_image += coeff*z

        # Normalize
        total_image = total_image
        width = np.abs(total_image.max() - total_image.min())
#         import pdb; pdb.set_trace()
        total_image = (total_image - np.mean(total_image))/(width + eps)
        return total_image
    
    def gen_rotated_images(self, nm_terms, coeffs, phases, n_rotations=60, im_size=80):
        rotations = 2*np.pi*np.linspace(0,1,n_rotations)
        images = np.empty((len(rotations),im_size,im_size))
        for i, rot in enumerate(rotations):
            total_image = self.generate_linear_combination(nm_terms,coeffs,phases - rot,im_size=im_size)
            images[i] = total_image
        images = torch.Tensor(images)
        return images
    
    def gen_image_classes_with_rotation(self, n_classes=10, n_rotations=60, im_size=80, n_basis_el=6, mmax=13):
        n_freqs = list(range(1,mmax))
        m_freqs = list(range(0,mmax))
        nm_terms = list(product(n_freqs,m_freqs))

        X = torch.Tensor()
        Y = torch.Tensor()
        for i in range(n_classes):
            nm_terms_sample = random.sample(nm_terms,n_basis_el)
            coeffs = 2*np.random.rand(len(nm_terms_sample)) - 1
            phases = 2*np.pi*np.random.rand(len(nm_terms_sample))

            rot_images = self.gen_rotated_images(nm_terms_sample, coeffs, phases, n_rotations = n_rotations, im_size=im_size)
            labels = torch.Tensor([i]*n_rotations)

            X = torch.cat([X,rot_images],dim = 0)
            Y = torch.cat([Y,labels],dim = 0)

        return X,Y


    def gen_dataset(self):
        self.data, self.labels = self.gen_image_classes_with_rotation(self.n_classes, self.n_rotations, self.img_side, self.n_basis_el, self.mmax)
        if self.ravel:
            n_images = self.data.shape[0]
            self.data = self.data.reshape(n_images,-1)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    

class HarmonicPatternsS1xS1(Dataset):
    
    def __init__(self,
                 img_size=(32, 32),
                 n_classes=10,
                 n_harmonics=5,
                 max_frequency=16,
                 seed=0,
                 ravel=True,
                 real=True):
        
        super().__init__()
        np.random.seed(seed)
        self.name = 'oscillations-s1xs1'
        self.img_size = img_size
        if ravel:
            self.dim = img_size[0] * img_size[1]
        else:
            self.dim = self.img_size
        self.max_frequency = max_frequency
        self.n_classes = n_classes
        self.seed = seed
        self.ravel = ravel
        self.real = real
        self.n_harmonics = n_harmonics
        
        self.coordinates_v = np.linspace(0, np.pi * 2, self.img_size[0], endpoint=False)
        self.coordinates_h = np.linspace(0, np.pi * 2, self.img_size[1], endpoint=False)
        self.grid_h, self.grid_v = np.meshgrid(self.coordinates_h, self.coordinates_v)
        self.gen_dataset()

    def gen_dataset(self):
        data = []
        labels = []
        for c in range(self.n_classes):
            d = self.random_signal()
            if self.ravel:
                d = d.ravel()
            data.append(d)
            labels.append(c)
        self.data = torch.tensor(np.array(data))
        self.labels = torch.tensor(labels)
        
    def random_signal(self):
        d = np.zeros(self.img_size, dtype=np.complex64)
        for i in range(self.n_harmonics):
            omega_h, omega_v = np.random.randint(-self.max_frequency, self.max_frequency + 1), np.random.randint(-self.max_frequency, self.max_frequency + 1)
            phase_h, phase_v = np.random.randint(self.img_size[1]), np.random.randint(self.img_size[0])
            amplitude = np.random.uniform(0, 1)
            coords_h, coords_v = self.translate(self.grid_h, phase_h, phase_v), self.translate(self.grid_v, phase_h, phase_v)
            f = np.cos(coords_h * omega_h + coords_v * omega_v) + 1j * np.sin(coords_h * omega_h + coords_v * omega_v)
            d += f
        if self.real:
            d = d.real
        d -= np.mean(d)
        d /= np.max(abs(d))
        return d

    def translate(self, img, h=0, v=0):
        new_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                oldi = (i - v) % img.shape[0]
                oldj = (j - h) % img.shape[1]
                new_img[i, j] = img[oldi, oldj]
        return new_img
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)

    
class HarmonicPatternsS1xS1Orbit(HarmonicPatternsS1xS1):
    
    def __init__(self,
                 img_size=(32, 32),
                 n_classes=10,
                 n_harmonics=5,
                 max_frequency=16,
                 percent_transformations=1.0,
                 seed=0,
                 ordered=False,
                 ravel=True,
                 real=True,
                 equivariant=False):
        
        self.percent_transformations = percent_transformations
        self.ordered = ordered
        self.n_transformations = int(img_size[0] * img_size[1] * percent_transformations)
        self.equivariant = equivariant
        
        super().__init__(img_size=img_size,
                         n_classes=n_classes,
                         n_harmonics=n_harmonics,
                         max_frequency=max_frequency,
                         seed=seed,
                         ravel=ravel,
                         real=real)
        
        self.name = 'oscillations-s1xs1-orbit'
        
    def gen_dataset(self):
        data = []
        labels = []
        for c in range(self.n_classes):
            signal = self.random_signal()
            orbit = self.gen_orbit(signal)
            if self.ravel:
                orbit = [x.ravel() for x in orbit]
            data += orbit  
            labels += [c] * len(orbit)
        self.data = torch.tensor(np.array(data))
        self.labels = torch.tensor(labels)
        
    def gen_orbit(self, signal):
        orbit = []
        
        all_transformations = list(
            itertools.product(
                np.arange(self.img_size[0]),
                np.arange(self.img_size[1]),
            )
        )
        if not self.ordered:
             np.random.shuffle(all_transformations)
        select_transformations = all_transformations[:self.n_transformations]
        for g1, g2 in select_transformations:
            signal_t = self.translate(signal, g1, g2)
            orbit.append(signal_t) 
        return orbit
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    

class Cyclic2DTranslation(Dataset):
    def __init__(
        self,
        n_classes=100,
        dim=(32, 32),
        noise=0.2,
        seed=0,
        percent_transformations=1.0,
        vectorized=True,
        ordered=False,
    ):
        np.random.seed(seed)
        self.name = "cyclic-2d-translation"
        random_classes = np.random.uniform(-1, 1, size=(n_classes,) + dim)
        random_classes -= np.mean(random_classes, axis=(1, 2), keepdims=True)
        random_classes /= np.std(random_classes, axis=(1, 2), keepdims=True)
        dataset, labels, transformations = [], [], []

        all_transformations = list(
            itertools.product(
                np.arange(dim[0]),
                np.arange(dim[1]),
            )
        )

        n_transformations = int(len(all_transformations) * percent_transformations)

        for i, c in enumerate(random_classes):
            if not ordered:
                np.random.shuffle(all_transformations)
            select_transformations = all_transformations[:n_transformations]
            for h, v in select_transformations:
                datapoint = self.translate(c, h, v)
                n = np.random.uniform(-noise, noise, size=dim)
                datapoint += n

                if vectorized:
                    datapoint = datapoint.ravel()

                dataset.append(datapoint)
                labels.append(i)
                transformations.append((h, v))

        self.data = torch.Tensor(dataset)
        self.labels = torch.Tensor(labels)
        self.transformation = torch.Tensor(transformations)
        self.dim = datapoint.shape[-1]
        self.n_classes = n_classes
        self.noise = noise
        self.seed = seed
        self.percent_transformations = percent_transformations
        self.ordered = ordered
        self.vectorized = vectorized

    def translate(self, img, x=0, y=0):
        """
        Given an image and offset x, y, returns a translated image up in y and right in x
        """
        new_img = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                oldi = (i - x) % img.shape[0]
                oldj = (j - y) % img.shape[1]
                new_img[i, j] = img[oldi, oldj]
        return new_img

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class C42D(Dataset):
    def __init__(
        self,
        n_classes=500,
        dim=(32, 32),
        noise=0.2,
        seed=0,
        vectorized=True,
        ordered=False,
        n_repeats=50,
    ):
        
        super().__init__()

        np.random.seed(seed)
        self.name = "c4-2d"
        random_classes = np.random.uniform(-1, 1, size=(n_classes,) + dim)
        random_classes -= np.mean(random_classes, axis=(1, 2), keepdims=True)
        random_classes /= np.std(random_classes, axis=(1, 2), keepdims=True)
        dataset, labels, transformations = [], [], []

        all_transformations = [0, 1, 2, 3] * n_repeats

        for i, c in enumerate(random_classes):
            if not ordered:
                np.random.shuffle(all_transformations)
            for t in all_transformations:
                datapoint = np.rot90(c, t)
                n = np.random.uniform(-noise, noise, size=dim)
                datapoint += n

                if vectorized:
                    datapoint = datapoint.ravel()

                dataset.append(datapoint)
                labels.append(i)
                transformations.append(t * 90)

        self.data = torch.Tensor(dataset)
        self.labels = torch.Tensor(labels)
        self.transformation = torch.Tensor(transformations)
        self.dim = datapoint.shape[-1]
        self.n_classes = n_classes
        self.noise = noise
        self.seed = seed
        self.ordered = ordered
        self.vectorized = vectorized
        self.n_repeats = n_repeats

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)

    
class MNIST(Dataset):

    def __init__(
        self,
        test=False,
        ordered=False
    ):

        super().__init__()

        self.name = "mnist"
        self.dim = 28 ** 2
        self.img_size = (28, 28)
        self.test = test
        self.ordered = ordered

        if test:
            mnist = np.array(pd.read_csv("/home/sanborn/datasets/mnist/mnist_test.csv"))
        else:
            mnist = np.array(pd.read_csv("/home/sanborn/datasets/mnist/mnist_test.csv"))

        labels = mnist[:, 0]
        mnist = mnist[:, 1:]
        mnist = mnist / 255
        mnist = mnist - mnist.mean(axis=1, keepdims=True)
        mnist = mnist / mnist.std(axis=1, keepdims=True)
#         mnist = mnist.reshape((len(mnist), 28, 28))

        if ordered:
            sort_idx = np.argsort(labels)
            mnist = mnist[sort_idx]
            labels = labels[sort_idx]
            
        self.data = torch.Tensor(mnist)
        self.labels = torch.Tensor(labels)


    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    

class TranslatedMNIST(Dataset):

    # TODO: make unraveled version

    def __init__(
        self,
        exemplars_per_digit,
        digits=np.arange(10),
        percent_transformations=0.5,
        max_translation=7,
        ordered=False,
        noise=0.1,
        seed=0,
    ):

        super().__init__()

        self.name = "translated-mnist"
        np.random.seed(seed)
        self.dim = 28 ** 2

        mnist = np.array(pd.read_csv("~/data/mnist/mnist_test.csv"))

        all_labels = mnist[:, 0]
        label_idxs = {a: np.where(all_labels == a)[0] for a in digits}

        mnist = mnist[:, 1:]
        mnist = mnist / 255
        mnist = mnist - mnist.mean(axis=1, keepdims=True)
        mnist = mnist / mnist.std(axis=1, keepdims=True)
        mnist = mnist.reshape((len(mnist), 28, 28))

        all_translations = list(
            itertools.product(
                np.arange(-max_translation, max_translation),
                np.arange(-max_translation, max_translation),
            )
        )

        n_translations = int(len(all_translations) * percent_transformations)

        data = []
        labels = []
        exemplar_labels = []
        ex_idx = 0

        for number in digits:
            # Select digits
            idxs = np.random.choice(
                label_idxs[number], exemplars_per_digit, replace=False
            )

            # Translate exemplars + Add noise
            for idx in idxs:
                img = mnist[idx]
                l = all_labels[idx]
                # Select translations

                if not ordered:
                    np.random.shuffle(all_translations)
                select_translations = all_translations[:n_translations]

                for i, j in select_translations:
                    t = self.translate(img, i, j).ravel()
                    n = np.random.uniform(-noise, noise, size=self.dim)
                    t = t + n
                    data.append(t)
                    labels.append(l)
                    exemplar_labels.append(ex_idx)

                ex_idx += 1

        data = np.array(data)
        self.data = torch.Tensor(data)
        self.labels = labels
        self.exemplar_labels = exemplar_labels
        self.exemplars_per_digit = exemplars_per_digit

    def translate(self, img, x=0, y=0):
        """
        Given an image and offset x, y, returns a translated image up in y and right in x
        """
        new_img = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                oldi = (i - x) % img.shape[0]
                oldj = (j - y) % img.shape[1]
                new_img[i, j] = img[oldi, oldj]
        return new_img

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class RotatedMNIST(Dataset):
    def __init__(
        self,
        exemplars_per_digit,
        test=False,
        digits=np.arange(10),
        percent_transformations=1.0,
        noise=0.0,
        seed=0,
        n_repeats=50,
        ordered=False,
        ravel=True,
    ):

        super().__init__()

        self.name = "rotated-mnist"
        self.dim = 28 ** 2
        self.img_size = (28, 28)
        self.exemplars_per_digit = exemplars_per_digit
        self.digits = digits
        self.percent_transformations = percent_transformations
        self.noise = noise
        self.seed = seed
        self.ordered = ordered
        self.ravel = ravel
        self.test = test

        np.random.seed(seed)

        if test:
            mnist = np.array(pd.read_csv("/home/sanborn/datasets/mnist/mnist_test.csv"))
        else:
            mnist = np.array(pd.read_csv("/home/sanborn/datasets/mnist/mnist_train.csv"))

        all_labels = mnist[:, 0]
        label_idxs = {a: np.where(all_labels == a)[0] for a in digits}

        mnist = mnist[:, 1:]
        mnist = mnist / 255
        # TODO: VERIFY NORMALIZATION WITH ROTATION BORDERS
        #         mnist = mnist - mnist.mean(axis=1, keepdims=True)
        #         mnist = mnist / mnist.std(axis=1, keepdims=True)
        mnist = mnist.reshape((len(mnist), 28, 28))

        n_rotations = int(359 * percent_transformations)
        all_rotations = np.arange(360)

        data = []
        labels = []
        exemplar_labels = []

        ex_idx = 0

        for number in digits:
            # Select digits
            idxs = np.random.choice(
                label_idxs[number], exemplars_per_digit, replace=False
            )

            # Rotate exemplars + Add noise
            for idx in idxs:
                img = mnist[idx]
                l = all_labels[idx]

                # Select rotations
                if not ordered:
                    np.random.shuffle(all_rotations)
                select_rotations = all_rotations[:n_rotations]

                for angle in select_rotations:
                    t = transform.rotate(img, angle)
                    t -= t.mean(keepdims=True)
                    t /= t.std(keepdims=True)
                    n = np.random.uniform(-noise, noise, size=img.shape)
                    t = t + n
                    if ravel:
                        t = t.ravel()
                    data += [t] * n_repeats
                    labels += [l] * n_repeats
                    exemplar_labels.append(ex_idx)

                ex_idx += 1

        data = np.array(data)
        if not ravel:
            self.channels = 1
            data = data.reshape((data.shape[0], 1, 28, 28))
        self.data = torch.Tensor(data)
        self.labels = labels
        self.exemplar_labels = exemplar_labels
        self.exemplars_per_digit = exemplars_per_digit
        self.n_repeats = n_repeats

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class SinusoidSums2D(Dataset):
    def __init__(
        self,
        img_size=(16, 16),
        n_classes=100,
        n_sinusoids=5,
        max_frequency=5,
        complex_valued=False,
        vectorized=True,
        seed=0,
    ):
        
        super().__init__()

        self.img_size = img_size
        self.n_sinusoids = n_sinusoids
        self.max_frequency = max_frequency
        self.complex_valued = complex_valued
        self.vectorized = vectorized
        self.n_classes = n_classes
        self.seed = seed

        np.random.seed(seed)

        data = []
        labels = []
        for n in range(n_classes):
            img = np.zeros(img_size)
            for i in range(n_sinusoids):
                orientation = np.random.uniform(-np.pi, np.pi)
                frequency = np.random.uniform(0, max_frequency)
                phase = np.random.uniform(-np.pi, np.pi)
                amplitude = np.random.uniform(-1, 1)
                sinusoid = self.get_sinusoid(orientation, frequency, phase)
                img += amplitude * sinusoid
            if self.vectorized:
                img = img.ravel()
            labels.append(n)
            data.append(img)

        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels)

    def get_sinusoid(self, orientation, frequency, phase):

        dh = np.linspace(0, 1, self.img_size[1], endpoint=True)
        dv = np.linspace(0, 1, self.img_size[0], endpoint=True)

        ihs, ivs = np.meshgrid(dh, dv)

        fh = -frequency * np.cos(orientation) * 2 * np.pi
        fv = frequency * np.sin(orientation) * 2 * np.pi

        sinusoid = np.cos(ihs * fh + ivs * fv + phase)

        if self.complex_valued:
            c = np.sin(ihs * fh + ivs * fv + phase)
            sinusoid = sinusoid + 1j * c

        return sinusoid

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class RotatedSinusoidSums2D(SinusoidSums2D):
    def __init__(
        self,
        img_size=(16, 16),
        n_classes=100,
        n_sinusoids=5,
        max_frequency=5,
        complex_valued=False,
        percent_transformations=1.0,
        vectorized=True,
        n_repeats=1,
        ordered=False,
        noise=0.0,
        circle_crop=True,
        seed=0,
        name="rotated-sinusoid-sums-2d",
        equivariant=False,
    ):
        

        super().__init__(
            img_size,
            n_classes=n_classes,
            n_sinusoids=n_sinusoids,
            max_frequency=max_frequency,
            complex_valued=complex_valued,
            vectorized=False,
            seed=seed,
        )

        self.ordered = ordered
        self.percent_transformations = percent_transformations
        self.n_repeats = n_repeats
        self.noise = noise
        self.circle_crop = circle_crop
        self.dim = img_size[0] * img_size[1]
        self.name = name
        self.vectorized = vectorized
        self.equivariant = equivariant

        n_rotations = int(359 * percent_transformations)
        all_rotations = np.arange(360)

        if circle_crop:
            if img_size[0] % 2 == 0 or img_size[1] % 2 == 0:
                raise ValueError(
                    "Image size should be a tuple of odd numbers to ensure centered rotational symmetry."
                )
            v, h = np.mgrid[: img_size[0], : img_size[1]]
            equation = (v - ((img_size[0] - 1) / 2)) ** 2 + (
                h - ((img_size[1] - 1) / 2)
            ) ** 2
            circle = equation < (equation.max() / 2)

        data = []
        labels = []
        s = []
        if equivariant:
            x0 = []

        for i, img in enumerate(self.data):

            # Select rotations
            if not ordered:
                np.random.shuffle(all_rotations)
            select_rotations = all_rotations[:n_rotations]

            for angle in select_rotations:
                for r in range(n_repeats):
                    t = transform.rotate(img, angle)
                    t -= t.mean(keepdims=True)
                    t /= t.std(keepdims=True)
                    n = np.random.uniform(-noise, noise, size=img.shape)
                    t = t + n
                    if circle_crop:
                        t[~circle] = 0.0
                    if vectorized:
                        data.append(t.ravel())
                    else:
                        data.append(t)
                    labels.append(self.labels[i])
                    s.append(angle / 360.0)  # NB: Might want to center at 0
                    if equivariant:
                        if circle_crop:
                            canonical_img = img.clone()
                            canonical_img[~circle] = 0.0       
                        else:
                            canonical_img = img
                        if vectorized:
                            canonical_img = canonical_img.ravel()
                        x0.append(canonical_img)

        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels)
        self.s = torch.Tensor(s)
        if equivariant:
            self.x0 = x0

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.equivariant:
            s = self.s[idx]
            x0 = self.x0[idx]
            return x, y, s, x0
        else:
            return x, y

    def __len__(self):
        return len(self.data)
    

class Omniglot(Dataset):
    def __init__(
        self,
        test=False,
        ravel=True,
    ):

        super().__init__()

        self.name = "omniglot"
        self.ravel = ravel
        self.test = test

        if test:
            omniglot = pd.read_pickle('/home/sanborn/datasets/omniglot/omniglot_30x30_test.p')
        else:
            omniglot = pd.read_pickle('/home/sanborn/datasets/omniglot/omniglot_30x30_train.p')
            
        labels = np.array(list(omniglot.labels))
        alphabet_labels = np.array(list(omniglot.alphabet_labels))
        imgs = np.array(list(omniglot.imgs))

        self.dim = 900
        self.img_size = (30, 30)
        if not ravel:
            self.channels = 1
            imgs = imgs.reshape((imgs.shape[0], 1, 30, 30))
        else:
            imgs = imgs.reshape((imgs.shape[0], -1))
        self.data = torch.Tensor(imgs)
        self.labels = torch.Tensor(labels)
        self.alphabet_labels = torch.Tensor(alphabet_labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class RotatedOmniglot(Dataset):
    def __init__(
        self,
        test=False,
        ravel=True,
        n_exemplars=1,
        n_repeats=10,
        n_transformations=360,
    ):

        super().__init__()

        self.name = "rotated-omniglot"
        self.ravel = ravel
        self.test = test
        self.n_exemplars = n_exemplars
        self.n_repeats = n_repeats
        self.n_transformations = n_transformations

        if test:
            omniglot = pd.read_pickle('/home/sanborn/datasets/omniglot/omniglot_30x30_test.p')
        else:
            omniglot = pd.read_pickle('/home/sanborn/datasets/omniglot/omniglot_30x30_train.p')
            
        all_labels = np.array(list(omniglot.labels))
        all_alphabet_labels = np.array(list(omniglot.alphabet_labels))
        imgs = np.array(list(omniglot.imgs))

        character_idxs = {a: np.where(all_labels == a)[0] for a in all_labels}

        data = []
        labels = []
        alphabet_labels = []
        
        rotations = np.linspace(0, 360, n_transformations)
        
        for l in all_labels:
            exemplar_idxs = np.random.choice(character_idxs[l], n_exemplars)
            character_imgs = imgs[exemplar_idxs]
            for i in exemplar_idxs:
                for angle in rotations:
                    t = transform.rotate(imgs[i], angle)
                    t -= t.mean(keepdims=True)
                    t /= t.std(keepdims=True)
                    if ravel:
                        t = t.ravel()
                    data += [t] * n_repeats
                    labels += [all_labels[i]] * n_repeats
                    alphabet_labels += [all_alphabet_labels[i]] * n_repeats

        data = np.array(data)
        self.dim = 900
        self.img_size = (30, 30)
        
        if not ravel:
            self.channels = 1
            data = data.reshape((data.shape[0], 1, 30, 30))

        self.data = torch.Tensor(data)
        self.labels = labels
        self.alphabet_labels = alphabet_labels


    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class TinyImageNet(Dataset):
    def __init__(self, batch_size=32, train=True):
        super().__init__()

        self.name = "tiny-imagenet"
        if train:
            folder = "~/data/tiny-imagenet-200/train"
        else:
            folder = "~/data/tiny-imagenet-200/val"

        self.data = torchvision.datasets.ImageFolder(
            folder, transform=torchvision.transforms.ToTensor()
        )

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class NORB(Dataset):
    def __init__(
        self,
        camera="left",
        label="category",
        n_exemplars=1,
        categories=range(10),
        test=False,
        normalize=True,
        ravel=True,
        seed=0,
    ):

        super().__init__()

        self.name = "norb"
        np.random.seed(seed)

        if test:
            folder = os.path.expanduser("~/data/NORB/test/")
            label_df = pd.read_pickle(os.path.join(folder, "labels.p"))

            if camera == "left":
                self.data = torch.tensor(np.load(folder + "image1.npy"))

            elif camera == "right":
                self.data = torch.tensor(np.load(folder + "image2.npy"))

            elif camera == "both":
                raise NotImplementedError

            else:
                raise ValueError("camera must be one of [left, right, both]")

        else:
            folder = os.path.expanduser("~/data/NORB/train/")
            label_df = pd.read_pickle(os.path.join(folder, "labels.p"))

            if camera == "left":
                self.data = torch.tensor(np.load(folder + "image1.npy"))

            elif camera == "right":
                self.data = torch.tensor(np.load(folder + "image2.npy"))

            elif camera == "both":
                raise NotImplementedError

            else:
                raise ValueError("camera must be one of [left, right, both]")

        self.data = self.data[:, 10:96, 10:96]
        self.azimuth = torch.tensor(list(label_df.azimuth))
        self.category = torch.tensor(list(label_df.category))
        self.elevation = torch.tensor(list(label_df.elevation))
        self.lighting = torch.tensor(list(label_df.lighting))

        if label == "category":
            self.labels = self.category

        elif label == "azimuth":
            self.labels = self.azimuth

        elif label == "elevation":
            self.labels = self.elevation

        elif label == "lighting":
            self.labels = self.lighting

        if ravel:
            self.data = self.data.reshape((self.data.shape[0], -1))

        # TODO: Make this valid by using the same mean and std over train and test data
        if normalize:
            self.data = self.data - self.data.mean()
            self.data = self.data / self.data.std()

        self.dim = 86 ** 2

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class TranslatedNORB(NORB):

    # TODO: make unraveled version

    def __init__(
        self,
        camera="left",
        label="category",
        n_exemplars=1,
        categories=range(5),
        percent_transformations=0.5,
        max_translation=7,
        test=False,
        normalize=True,
        noise=0.0,
        ravel=True,
        ordered=False,
        seed=0,
    ):

        super().__init__(
            camera=camera,
            label=label,
            test=test,
            normalize=False,
            ravel=False,
            seed=seed,
        )

        self.name = "translated-norb"

        label_idxs = {a: np.where(self.category == a)[0] for a in categories}

        all_translations = list(
            itertools.product(
                np.arange(-max_translation, max_translation),
                np.arange(-max_translation, max_translation),
            )
        )

        n_translations = int(len(all_translations) * percent_transformations)

        data = []
        labels = []
        exemplar_labels = []
        ex_idx = 0
        azimuth = []
        category_labels = []
        elevation = []
        lighting = []
        exemplar_labels = []

        bordered_data = torch.zeros((self.data.shape[0], 100, 100))
        bordered_data = bordered_data + self.data[:, 0, 0].unsqueeze(1).unsqueeze(2)
        bordered_data[:, 7:93, 7:93] = self.data
        del self.data
        self.dim = 100 ** 2

        data = []

        for category in categories:
            idxs = np.random.choice(label_idxs[category], n_exemplars, replace=False)

            # Translate exemplars + Add noise
            for idx in idxs:
                img = bordered_data[idx]
                a = self.azimuth[idx]
                c = self.category[idx]
                e = self.elevation[idx]
                l = self.lighting[idx]

                # Select translations
                if not ordered:
                    np.random.shuffle(all_translations)
                select_translations = all_translations[:n_translations]

                for i, j in select_translations:
                    t = self.translate(img, i, j)
                    n = np.random.uniform(-noise, noise, size=t.shape)
                    t = t + n
                    if ravel:
                        t = t.ravel()
                    data.append(t)
                    azimuth.append(a)
                    category_labels.append(c)
                    elevation.append(e)
                    lighting.append(l)
                    exemplar_labels.append(ex_idx)

                ex_idx += 1

        data = np.array(data)
        self.data = torch.Tensor(data)
        self.azimuth = azimuth
        self.category = category_labels
        self.elevation = elevation
        self.lighting = lighting
        self.exemplar_labels = exemplar_labels

        if normalize:
            self.data = self.data - self.data.mean()
            self.data = self.data / self.data.std()

    def translate(self, img, x=0, y=0):
        """
        Given an image and offset x, y, returns a translated image up in y and right in x
        """
        new_img = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                oldi = (i - x) % img.shape[0]
                oldj = (j - y) % img.shape[1]
                new_img[i, j] = img[oldi, oldj]
        return new_img

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class NaturalCyclicTranslatingPatches(Dataset):
    def __init__(
        self,
        patches_per_image=10,
        patch_size=16,
        n_frames=10,
        images=range(98),
        max_translation=16,
        color=False,
        normalize_imgs=True,
        normalize_vids=False,
        normalize_frames=True,
        min_contrast=0.2,
        seed=0,
        ravel=True,
    ):

        super().__init__()

        self.name = "natural-cyclic-translating-patches"
        np.random.seed(seed)
        img_shape = (512, 512)
        self.dim = patch_size ** 2

        directory = os.path.expanduser("~/data/curated-natural-images/images/")

        data = []
        exemplar_labels = []
        translation = []

        i = 0

        for idx in images:
            n_zeros = 4 - len(str(idx))
            str_idx = "0" * n_zeros + str(idx)
            img = np.asarray(Image.open(directory + "{}.png".format(str_idx)))

            if not color:
                img = img.mean(axis=-1)

            if normalize_imgs:
                img -= img.mean()
                img /= img.std()

            for p in range(patches_per_image):

                low_contrast = True
                while low_contrast:
                    start_x = np.random.randint(0, img_shape[1] - patch_size)
                    start_y = np.random.randint(0, img_shape[0] - patch_size)
                    start_patch = img[
                        start_y : start_y + patch_size, start_x : start_x + patch_size
                    ]
                    if start_patch.std() >= min_contrast:
                        low_contrast = False

                end_x = np.random.randint(-max_translation, max_translation)
                end_y = np.random.randint(-max_translation, max_translation)

                x_trajectory = np.linspace(0, end_x, n_frames)
                y_trajectory = np.linspace(0, end_y, n_frames)

                x_trajectory = [round(a) for a in x_trajectory]
                y_trajectory = [round(a) for a in y_trajectory]

                frames = []
                stds = []

                if normalize_frames:
                    start_patch -= start_patch.mean()
                    start_patch /= start_patch.std()

                frames.append(start_patch)
                stds.append(start_patch.std())

                for j in range(1, n_frames):
                    next_img = self.translate(
                        start_patch, x_trajectory[j], y_trajectory[j]
                    )
                    if normalize_frames:
                        next_img -= next_img.mean()
                        next_img /= next_img.std()
                    frames.append(next_img)
                    stds.append(next_img.std())

                    if np.mean(stds) >= min_contrast:
                        low_contrast = False

                if ravel:
                    frames = [x.ravel() for x in frames]

                if normalize_vids:
                    frames = [x - np.mean(frames) for x in frames]
                    frames = [x / np.std(frames) for x in frames]

                data += frames
                exemplar_labels += [i] * n_frames
                translation += [(start_x, start_y), (end_x, end_y)] * n_frames

                i += 1

        data = np.array(data)
        self.data = torch.Tensor(data)
        self.exemplar_labels = exemplar_labels
        self.patches_per_image = patches_per_image

    def translate(self, img, x=0, y=0):
        """
        Given an image and offset x, y, returns a translated image up in y and right in x
        """
        new_img = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                oldi = (i - x) % img.shape[0]
                oldj = (j - y) % img.shape[1]
                new_img[i, j] = img[oldi, oldj]
        return new_img

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class NaturalTranslatingPatches(Dataset):

    """
    Makes videos of patches linearly sweeping over large images.
    Identical to transform_datasets.torch.video.NaturalTranslatingPatches
    except that each video frame is saved as a separate image with the exemplar label
    indicating the video it belongs to.
    """

    def __init__(
        self,
        patches_per_image=10,
        patch_size=16,
        n_frames=10,
        images=range(98),
        max_translation=10,
        color=False,
        normalize_imgs=True,
        normalize_vids=False,
        normalize_frames=True,
        min_contrast=0.2,
        seed=0,
        ravel=True,
    ):

        super().__init__()

        self.name = "natural-images-translating-patches"
        np.random.seed(seed)
        img_shape = (512, 512)
        self.dim = patch_size ** 2

        directory = os.path.expanduser("~/data/curated-natural-images/images/")

        data = []
        exemplar_labels = []
        translation = []

        i = 0

        for idx in images:
            n_zeros = 4 - len(str(idx))
            str_idx = "0" * n_zeros + str(idx)
            img = np.asarray(Image.open(directory + "{}.png".format(str_idx)))

            if not color:
                img = img.mean(axis=-1)

            if normalize_imgs:
                img -= img.mean()
                img /= img.std()

            for p in range(patches_per_image):

                low_contrast = True
                while low_contrast:
                    start_x = np.random.randint(0, img_shape[1] - patch_size)
                    start_y = np.random.randint(0, img_shape[0] - patch_size)
                    start_patch = img[
                        start_y : start_y + patch_size, start_x : start_x + patch_size
                    ]

                    x_lower_bound = max(0, start_x - max_translation)
                    y_lower_bound = max(0, start_y - max_translation)

                    x_upper_bound = min(
                        start_x + max_translation, img_shape[1] - patch_size
                    )
                    y_upper_bound = min(
                        start_y + max_translation, img_shape[0] - patch_size
                    )

                    end_x = np.random.randint(x_lower_bound, x_upper_bound)
                    end_y = np.random.randint(y_lower_bound, y_upper_bound)

                    x_trajectory = np.linspace(start_x, end_x, n_frames)
                    y_trajectory = np.linspace(start_y, end_y, n_frames)

                    x_trajectory = [round(a) for a in x_trajectory]
                    y_trajectory = [round(a) for a in y_trajectory]

                    frames = []
                    stds = []

                    if ravel:
                        start_patch = start_patch.ravel()

                    if normalize_frames:
                        start_patch -= start_patch.mean()
                        start_patch /= start_patch.std()

                    frames.append(start_patch)
                    stds.append(start_patch.std())

                    for j in range(1, n_frames):
                        next_img = img[
                            y_trajectory[j] : y_trajectory[j] + patch_size,
                            x_trajectory[j] : x_trajectory[j] + patch_size,
                        ]
                        if ravel:
                            next_img = next_img.ravel()
                        if normalize_frames:
                            next_img -= next_img.mean()
                            next_img /= next_img.std()
                        frames.append(next_img)
                        stds.append(next_img.std())

                    if np.mean(stds) >= min_contrast:
                        low_contrast = False

                if normalize_vids:
                    frames = [x - np.mean(frames) for x in frames]
                    frames = [x / np.std(frames) for x in frames]

                data += frames
                exemplar_labels += [i] * n_frames
                translation += [(start_x, start_y), (end_x, end_y)] * n_frames

                i += 1

        data = np.array(data)
        self.data = torch.Tensor(data)
        self.exemplar_labels = exemplar_labels
        self.patches_per_image = patches_per_image

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
