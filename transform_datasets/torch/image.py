import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import itertools
from skimage import transform
# from harmonics.spectral.wavelets import 


class TranslatedMNIST(Dataset):
    
    #TODO: make unraveled version
    
    def __init__(self,
                 exemplars_per_digit,
                 digits=np.arange(10),
                 percent_transformations=0.5,
                 max_translation=7,
                 noise = 0.1,
                 seed = 0):
       
        self.name = 'translated-mnist'
        np.random.seed(seed)
        self.dim = 28 ** 2
        
        mnist = np.array(pd.read_csv('~/data/mnist/mnist_test.csv'))
        
        all_labels = mnist[:, 0]
        label_idxs = {a: np.where(all_labels==a)[0] for a in digits}

        mnist = mnist[:, 1:]
        mnist = mnist / 255
        mnist = mnist - mnist.mean(axis=1, keepdims=True)
        mnist = mnist / mnist.std(axis=1, keepdims=True)
        mnist = mnist.reshape((len(mnist), 28, 28))
        
        all_translations = list(
                            itertools.product(
                                np.arange(-max_translation, max_translation), 
                                np.arange(-max_translation, max_translation)))
        
        n_translations = int(len(all_translations) * percent_transformations)

        
        data = []
        labels = []
        exemplar_labels = []
        ex_idx = 0
        
        for number in digits:
            # Select digits
            idxs = np.random.choice(label_idxs[number], 
                                    exemplars_per_digit, 
                                    replace=False)
            
            # Translate exemplars + Add noise
            for idx in idxs:
                img = mnist[idx]
                l = all_labels[idx]
                # Select translations

#                 np.random.shuffle(all_translations)
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
                oldi = (i-x)%img.shape[0]
                oldj = (j-y)%img.shape[1]
                new_img[i,j] = img[oldi,oldj]
        return new_img

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    
class RotatedMNIST(Dataset):
    
    def __init__(self,
                 exemplars_per_digit,
                 digits=np.arange(10),
                 percent_transformations=0.3,
                 noise = 0.1,
                 seed = 0,
                 ravel=False):

        self.name = 'rotated-mnist'
        np.random.seed(seed)
        self.dim = 28 ** 2
        
        mnist = np.array(pd.read_csv('~/data/mnist/mnist_test.csv'))
        
        all_labels = mnist[:, 0]
        label_idxs = {a: np.where(all_labels==a)[0] for a in digits}

        mnist = mnist[:, 1:]
        mnist = mnist / 255
        #TODO: VERIFY NORMALIZATION WITH ROTATION BORDERS
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
            idxs = np.random.choice(label_idxs[number], 
                                    exemplars_per_digit, 
                                    replace=False)
            
            # Rotate exemplars + Add noise
            for idx in idxs:
                img = mnist[idx]
                l = all_labels[idx]
                
                # Select rotations
#                 np.random.shuffle(all_rotations)
                select_rotations = all_rotations[:n_rotations]

                for angle in select_rotations:
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

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    

class Omniglot(Dataset):
    
    def __init__(self,
                 exemplars_per_character,
                 characters=np.arange(10),
                 alphabets=np.arange(15),
                 seed=0,
                 ravel=False):
       
        self.name = 'omniglot'
        np.random.seed(seed)
        
        omniglot = pd.read_pickle('~/data/omniglot/omniglot_small.p')
        all_labels = np.array(list(omniglot.labels))
        all_alphabet_labels = np.array(list(omniglot.alphabet_labels))
        imgs = np.array(list(omniglot.imgs))
        
        alphabet_idxs = {a: np.where(all_alphabet_labels==a)[0] for a in alphabets}
        
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
    
    
class TinyImageNet(Dataset):
    
    def __init__(self,
                 batch_size=32):
        self.name = 'tiny-imagenet'
        train_dir = '~/data/tiny-imagenet-200/train'
        val_dir = '~/data/tiny-imagenet-200/val'

        train_dataset = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
        train_loader = data.DataLoader(train_dataset, batch_size=32)