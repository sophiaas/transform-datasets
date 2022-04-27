import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from PIL import Image



class MNIST(Dataset):
    """
    Dataset object for the MNIST dataset.
    Takes the MNIST file path, then loads, standardizes, and saves it internally.
    """

    def __init__(self, path, ordered=False):

        super().__init__()

        self.name = "mnist"
        self.dim = 28 ** 2
        self.img_size = (28, 28)
        self.ordered = ordered

        mnist = np.array(pd.read_csv(path))

        labels = mnist[:, 0]
        mnist = mnist[:, 1:]
        mnist = mnist / 255
        mnist = mnist - mnist.mean(axis=1, keepdims=True)
        mnist = mnist / mnist.std(axis=1, keepdims=True)
        mnist = mnist.reshape((len(mnist), 28, 28))
        if ordered:
            sort_idx = np.argsort(labels)
            mnist = mnist[sort_idx]
            labels = labels[sort_idx]

        self.data = torch.Tensor(mnist)
        self.labels = torch.Tensor(labels).long()

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    
class MNISTExemplars(Dataset):
    """
    Dataset object for the MNIST dataset.
    Takes the MNIST file path, then loads, standardizes, and saves it internally.
    """

    def __init__(self, path, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_exemplars=1):

        super().__init__()

        self.name = "mnist"
        self.dim = 28 ** 2
        self.img_size = (28, 28)
        self.digits = digits
        self.n_exemplars = n_exemplars

        mnist = np.array(pd.read_csv(path))

        labels = mnist[:, 0]
        mnist = mnist[:, 1:]
        mnist = mnist / 255
        mnist = mnist - mnist.mean(axis=1, keepdims=True)
        mnist = mnist / mnist.std(axis=1, keepdims=True)
        mnist = mnist.reshape((len(mnist), 28, 28))
        
        label_idxs = {i: [j for j, x in enumerate(labels) if x == i] for i in range(10)}
        
        exemplar_data = []
        labels = []
        for d in digits:
            idxs = label_idxs[d]
            random_idxs = np.random.choice(idxs, size=self.n_exemplars, replace=False)
            for i in random_idxs:
                exemplar_data.append(mnist[i])
                labels.append(d)
            
        self.data = torch.tensor(exemplar_data)
        self.labels = torch.tensor(labels).long()

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    
class Omniglot(Dataset):
    def __init__(
        self,
        path,
        ravel=True,
    ):

        super().__init__()

        self.name = "omniglot"
        self.ravel = ravel

        omniglot = pd.read_pickle(path)
        labels = np.array(list(omniglot.labels))
        alphabet_labels = np.array(list(omniglot.alphabet_labels))
        imgs = np.array(list(omniglot.imgs))

        self.dim = 900
        self.img_size = (30, 30)
        imgs -= imgs.mean(axis=(1, 2), keepdims=True)
        imgs /= imgs.std(axis=(1, 2), keepdims=True)

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


class NaturalImagePatches(Dataset):
    def __init__(
        self,
        patches_per_image=10,
        patch_size=16,
        images=range(98),
        color=False,
        min_contrast=0.2,
        seed=0,
    ):

        super().__init__()

        self.name = "natural-img-patches"
        np.random.seed(seed)
        img_shape = (512, 512)
        self.dim = patch_size ** 2

        directory = os.path.expanduser("~/data/curated-natural-images/images/")

        data = []
        labels = []
        translation = []

        i = 0

        for idx in images:
            n_zeros = 4 - len(str(idx))
            str_idx = "0" * n_zeros + str(idx)
            img = np.asarray(Image.open(directory + "{}.png".format(str_idx)))

            if not color:
                img = img.mean(axis=-1)

            for p in range(patches_per_image):

                low_contrast = True
                j = 0 
                while low_contrast and j < 100:
                    start_x = np.random.randint(0, img_shape[1] - patch_size)
                    start_y = np.random.randint(0, img_shape[0] - patch_size)
                    patch = img[
                        start_y : start_y + patch_size, start_x : start_x + patch_size
                    ]
                    if patch.std() >= min_contrast:
                        low_contrast = False
                    j += 1
                
                if j == 100 and not low_contrast:
                    print("Couldn't find patch to meet contrast requirement. Skipping.")
                    continue
                    
                data.append(patch)
                labels.append(i)

                i += 1

        self.data = torch.tensor(np.array(data))
        self.labels = torch.tensor(np.array(labels))
        self.patches_per_image = patches_per_image

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)