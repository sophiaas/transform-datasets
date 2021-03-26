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

# from harmonics.spectral.wavelets import

# TODO: super init everything


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
        digits=np.arange(10),
        percent_transformations=1.0,
        noise=0.1,
        seed=0,
        n_repeats=50,
        ordered=False,
        ravel=True,
    ):

        self.name = "rotated-mnist"
        self.dim = 28 ** 2
        self.exemplars_per_digit = exemplars_per_digit
        self.digits = digits
        self.percent_transformations = percent_transformations
        self.noise = noise
        self.seed = seed
        self.ordered = ordered
        self.ravel = ravel

        np.random.seed(seed)

        mnist = np.array(pd.read_csv("~/data/mnist/mnist_test.csv"))

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
                    for r in range(n_repeats):
                        t = transform.rotate(img, angle)
                        t -= t.mean(keepdims=True)
                        t /= t.std(keepdims=True)
                        n = np.random.uniform(-noise, noise, size=img.shape)
                        t = t + n
                        if ravel:
                            data.append(t.ravel())
                        else:
                            data.append(t)
                        labels.append(l)
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


class Omniglot(Dataset):
    def __init__(
        self,
        exemplars_per_character,
        characters=np.arange(10),
        alphabets=np.arange(15),
        seed=0,
        ravel=False,
    ):

        self.name = "omniglot"
        np.random.seed(seed)

        omniglot = pd.read_pickle("~/data/omniglot/omniglot_small.p")
        all_labels = np.array(list(omniglot.labels))
        all_alphabet_labels = np.array(list(omniglot.alphabet_labels))
        imgs = np.array(list(omniglot.imgs))

        alphabet_idxs = {a: np.where(all_alphabet_labels == a)[0] for a in alphabets}

        data = []
        labels = []
        alphabet_labels = []

        for alphabet in alphabets:
            cs = imgs[alphabet_idxs[alphabet]]
            ls = all_labels[alphabet_idxs[alphabet]]
            als = all_alphabet_labels[alphabet_idxs[alphabet]]
            start = 0
            for c in characters:
                start = c * 20
                end = start + exemplars_per_character
                exemplars = cs[start:end]
                ex_ls = ls[start:end]
                ex_als = als[start:end]

                data += list(exemplars)
                labels += list(ex_ls)
                alphabet_labels += list(ex_als)
                start += 20

        data = np.array(data)
        self.dim = 900
        if not ravel:
            self.channels = 1
            data = data.reshape((data.shape[0], 1, 30, 30))
        self.data = torch.Tensor(data)
        self.labels = labels
        self.alphabet_labels = alphabet_labels
        self.exemplars_per_character = exemplars_per_character

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
                sinusoid = self.get_sinusoid(orientation, frequency, phase)
                img += sinusoid
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
                        if vectorized:
                            x0.append(img.reshape(-1))
                        else:
                            x0.append(img)

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


class RotatedOmniglot(Dataset):
    def __init__(
        self,
        exemplars_per_character,
        characters=np.arange(10),
        alphabets=np.arange(15),
        seed=0,
        ravel=True,
        ordered=False,
        noise=0.0,
        n_repeats=10,
        percent_transformations=1.0,
    ):

        self.name = "omniglot"
        np.random.seed(seed)

        omniglot = pd.read_pickle("~/data/omniglot/omniglot_small.p")
        all_labels = np.array(list(omniglot.labels))
        all_alphabet_labels = np.array(list(omniglot.alphabet_labels))
        imgs = np.array(list(omniglot.imgs))
        imgs = imgs.reshape((imgs.shape[0], 30, 30))

        alphabet_idxs = {a: np.where(all_alphabet_labels == a)[0] for a in alphabets}

        n_rotations = int(359 * percent_transformations)
        all_rotations = np.arange(360)

        data = []
        labels = []
        alphabet_labels = []

        for alphabet in alphabets:
            cs = imgs[alphabet_idxs[alphabet]]
            ls = all_labels[alphabet_idxs[alphabet]]
            als = all_alphabet_labels[alphabet_idxs[alphabet]]
            start = 0
            for c in characters:
                start = c * 20
                end = start + exemplars_per_character
                exemplars = cs[start:end]
                ex_ls = ls[start:end]
                ex_als = als[start:end]

                for i, img in enumerate(exemplars):
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
                            if ravel:
                                data.append(t.ravel())
                            else:
                                data.append(t)
                            labels.append(ex_ls[i])
                            alphabet_labels.append(ex_als[i])

                start += 20

        data = np.array(data)
        self.dim = 900
        if not ravel:
            self.channels = 1
            data = data.reshape((data.shape[0], 1, 30, 30))
        else:
            data = data.reshape((data.shape[0], 900))
        self.data = torch.Tensor(data)
        self.labels = labels
        self.alphabet_labels = alphabet_labels
        self.exemplars_per_character = exemplars_per_character
        self.characters = characters
        self.alphabets = alphabets
        self.ordered = ordered
        self.ravel = ravel
        self.n_repeats = n_repeats
        self.percent_transformations = percent_transformations
        self.seed = seed
        self.noise = noise

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class TinyImageNet(Dataset):
    def __init__(self, batch_size=32, train=True):
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
