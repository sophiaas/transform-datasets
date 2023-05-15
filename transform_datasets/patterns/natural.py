import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import scipy.io
import os


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
        ravel=False,
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

        if not ravel:
#             self.channels = 1
            imgs = imgs.reshape((imgs.shape[0], 30, 30))
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
    ):

        super().__init__()

        self.name = "natural-img-patches"
        img_shape = (512, 512)
        self.dim = patch_size ** 2

        directory = os.path.expanduser("~/data/curated-natural-images/images/")

        data = []
        labels = []

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
    
    
class VanHaterenWhite(Dataset):
    def __init__(
        self,
        patches_per_image=10,
        patch_size=16,
        images=range(35),
        min_contrast=1.0,
    ):

        super().__init__()

        self.name = "van-hateren-white"
        full_img_shape = (512, 512)
        border = (11, 11)
        img_shape = (full_img_shape[0] - border[0]*2, full_img_shape[1] - border[1]*2)
        
        self.dim = patch_size ** 2

        directory = os.path.expanduser("~/datasets/van-hateren/IMAGES.mat")
        full_images = scipy.io.loadmat(directory)["IMAGES"][11:501, 11:501]
        full_images = np.transpose(full_images, (2, 0, 1))


        data = []
        labels = []

        i = 0
        

        for idx in images:
            
            img = full_images[idx]

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
    
    
class VanHateren(Dataset):
    def __init__(
        self,
        path,
        normalize=True,
        select_img_path="select_imgs.txt",
        patches_per_image=10,
        patch_size=16,
        min_contrast=1.0,
    ):


        super().__init__()
            
        self.name = "van-hateren"
        self.dim = patch_size ** 2
        self.path = path
        self.patches_per_image = patches_per_image
        self.select_img_path = select_img_path
        self.normalize = normalize
        self.patch_size = patch_size
        self.min_contrast = min_contrast
        self.img_shape = (1024, 1536)
        
        full_images = self.load_images()

        self.data, self.labels = self.get_patches(full_images)

        
    def get_patches(self, full_images):
        data = []
        labels = []

        i = 0
        
        for img in full_images:
            for p in range(self.patches_per_image):
                low_contrast = True
                j = 0 
                while low_contrast and j < 100:
                    start_x = np.random.randint(0, self.img_shape[1] - self.patch_size)
                    start_y = np.random.randint(0, self.img_shape[0] - self.patch_size)
                    patch = img[
                        start_y : start_y + self.patch_size, start_x : start_x + self.patch_size
                    ]
                    if patch.std() >= self.min_contrast:
                        low_contrast = False
                        data.append(patch)
                        labels.append(i)
                    j += 1
                
                if j == 100 and not low_contrast:
                    print("Couldn't find patch to meet contrast requirement. Skipping.")
                    continue

                i += 1
        data = torch.tensor(np.array(data))
        labels = torch.tensor(np.array(labels))
        return data, labels
                        
        
    def load_images(self):
        if self.select_img_path is not None:
            with open(self.path + self.select_img_path, "r") as f:
                img_paths = f.read().splitlines()
        else:
            img_paths = os.listdir(path + "images/")

        all_imgs = []

        for i, img_path in enumerate(img_paths):
            try:
                with open(self.path + "images/" + img_path, 'rb') as handle:
                    s = handle.read()
            except:
                print("Can't load image at path {}".format(self.path + img_path))
                continue
            img = np.fromstring(s, dtype='uint16').byteswap()
            if self.normalize:
                # Sets image values to lie between 0 and 1
                img = img.astype(float)
                img -= img.min()
                img /= img.max()
                img -= img.mean()
                img *= 2
            img = img.reshape(self.img_shape)
            all_imgs.append(img)

        all_imgs = np.array(all_imgs)
        return all_imgs

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    
    
class VanHaterenSlices(VanHateren):
    def __init__(
        self,
        path,
        normalize=True,
        select_img_path="select_imgs.txt",
        patches_per_image=10,
        patch_size=16,
        min_contrast=1.0,
    ):


        super().__init__(path=path,
                         normalize=normalize,
                         select_img_path=select_img_path,
                         patches_per_image=patches_per_image,
                         patch_size=patch_size,
                         min_contrast=min_contrast)

        
    def get_patches(self, full_images):
        data = []
        labels = []

        i = 0
        
        for img in full_images:
            for p in range(self.patches_per_image):
                low_contrast = True
                j = 0 
                while low_contrast and j < 100:
                    horizontal = np.random.randint(0, 2)
                    start_x = np.random.randint(0, self.img_shape[1] - self.patch_size)
                    start_y = np.random.randint(0, self.img_shape[0] - self.patch_size)
                    if horizontal == 1:
                        patch = img[
                            start_y, start_x : start_x + self.patch_size
                        ]
                    else:
                        patch = img[
                            start_y : start_y + self.patch_size, start_x
                        ]
                    if patch.std() >= self.min_contrast:
                        low_contrast = False
                        data.append(patch)
                        labels.append(i)
                    j += 1
                
                if j == 100 and not low_contrast:
                    print("Couldn't find patch to meet contrast requirement. Skipping.")
                    continue

                i += 1
        data = torch.tensor(np.array(data))
        labels = torch.tensor(np.array(labels))
        return data, labels
    
    
    
class ModelNet10Voxel(Dataset):
        
    def __init__(
        self,
        path=os.path.expanduser("~/datasets/ModelNet10Voxel/"),
        test=False,
#         grid_size=(20, 20, 20),
    ):
        
        
        super().__init__()
        self.path = path
        self.test = test
        self.grid_size = (20, 20, 20)
        
        self.data, self.labels, self.target_names = self.load_data()
        
    def _train_test_split_paths(self, dp, sub_path):
        path = os.path.join(dp, sub_path)
        file_paths = [os.path.join(path, i)
                      for i in os.listdir(path)]
        binvox_paths = list(filter(lambda x: '.binvox' in x, file_paths))
        return binvox_paths
    
    def _read_file(self, fp):
        from transform_datasets.utils.voxels import read_as_3d_array
        with open(fp, 'rb') as f:
            return read_as_3d_array(f).data

    def load_data(self):        
        dp = self.path
        dims = self.grid_size
        label_dirs = list(os.scandir(dp))
        label_dirs = [i for i in os.scandir(dp) if os.path.isdir(i)]
        target_names = [i.name for i in label_dirs]
        if self.test:
            subpath = "test"
        else:
            subpath = "train"

        paths, labels = [], []
        for i, dir_path in enumerate(label_dirs):
            for path in self._train_test_split_paths(dir_path.path, subpath):
                paths.append(path)
                labels.append(i)

        # converting binvox to numpy and reshape
        data = [self._read_file(i) for i in paths]
        data = np.array(data).reshape(-1, dims[0], dims[1], dims[2], 1)
        labels = np.array(labels)
        data = torch.tensor(data, dtype=torch.float32).squeeze()
        labels = torch.tensor(labels)
        return data, labels, target_names



