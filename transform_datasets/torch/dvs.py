import torch
from torch.utils.data import Dataset
import numpy as np
from transform_datasets.numpy.dvs import collapse_data, get_hot_pixels, erase_hot_pixels
from transform_datasets.numpy.dvs import dvs_to_vid as dvs_to_vid_numpy


def dvs_to_vid(x, y=None, t=None, p=None, frame_rate=120, img_size=(260, 346)):
    
    if y is None:
        y = x[:, 1]
        t = x[:, 2]
        p = x[:, 3]
        x = x[:, 0]
    
    x = x.numpy()
    y = y.numpy()
    t = t.numpy()
    p = p.numpy()
    
    projections = dvs_to_vid_numpy(x, y, t, p, frame_rate, img_size)
    
    return torch.tensor(projections)


class DVSMotion20(Dataset):
    
    def __init__(self,
                 start=0,
                 n_events=None,
                 sequence='classroom',
                 clean=False,
                 n_stds=5
                 ):

        filename = '/home/ssanborn/data/DVSMOTION20/camera-motion-data/{}_sequence/events_clean.npy'.format(sequence)
        of_filename = '/home/ssanborn/data/DVSMOTION20/camera-motion-data/{}_sequence/optic_flow_clean.npy'.format(sequence)
        
        self.img_size = (260, 346)
        
        data = np.load(filename)
        optic_flow = np.load(of_filename)

        if n_events is None:
            n_events = len(data)
            
        x = data[start:start+n_events, 0] 
        y = data[start:start+n_events, 1]
        t = data[start:start+n_events, 2]
        p = data[start:start+n_events, 3]
        optic_flow = optic_flow[start:start+n_events]
        
        if clean:
            x, y, t, p, optic_flow = erase_hot_pixels(x, y, t, p, optic_flow)
        
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.t = torch.tensor(t)
        self.p = torch.tensor(p)
        self.optic_flow = torch.tensor(optic_flow)
        
    def project(self, frame_rate):
        projected_vid = dvs_to_vid(x=self.x, 
                                   y=self.y, 
                                   t=self.t, 
                                   p=self.p, 
                                   frame_rate=frame_rate, 
                                   img_size=self.img_size)
        return projected_vid
        
    @property
    def pos(self):
        return torch.vstack([self.x, self.y, self.t]).T
    
    @property
    def data(self):
        return self.x, self.y, self.t, self.p

    def __getitem__(self, idx, flow=False):
        x = self.x[idx]
        y = self.y[idx]
        t = self.t[idx]
        p = self.p[idx]
        if flow:
            flow = self.optic_flow[idx]
            return x, y, t, p, flow
        else:
            return x, y, t, p
    
    def __len__(self):
        return len(self.x)
    
    
# TODO: Convert to ordinary pytorch dataset
import os
import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import pdb
from data.preprocessing import NormalizeRanges
from tqdm import tqdm


class MVSEC(Dataset):
    def __init__(self, 
                 root,
                 filename,
                 data_phase='train',
                 sample_timesteps=100,
                 sample_patchsize=(20, 20),
                 img_size=(346, 260),
                 pre_transform=NormalizeRanges((1, 20))):

#                  stride = 15,
            
        np.random.seed(0)
        self.root = root
        self.filename = filename
        
        self.sample_timesteps = sample_timesteps
        self.sample_patchsize = sample_patchsize
        self.data_phase = data_phase
        self.pre_transform = pre_transform
#         self.stride = stride
        self.img_size = img_size
        
        super().__init__(self.root, pre_transform=self.pre_transform)
        
    @property
    def raw_file_names(self):
        path = os.path.join(self.root, self.filename)
        return path
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')
        
    @property
    def processed_file_names(self):
#         self.n_datapoints = len(len(os.listdir(self.processed_dir)) - 1)
#         self.n_datapoints = 44891 # (20, 20, 500)
#         self.n_datapoints = 31423 # (20, 20, 1000)
#         self.n_datapoints = 25622 # ? (20, 20, 1000)
        self.n_datapoints = len(os.listdir(self.processed_dir)) - 2
        idxs = np.arange(self.n_datapoints)
        np.random.shuffle(idxs)
        
        n_train = int(self.n_datapoints * 0.7)
        n_validate = int(self.n_datapoints * 0.1)
        n_test = int(self.n_datapoints * 0.2)
        
        self.train_idxs = idxs[:n_train]
        self.validation_idxs = idxs[n_train:n_train+n_validate]
        self.test_idxs = idxs[n_train+n_validate:n_train+n_validate+n_test]
        
        if self.data_phase == "train":
            return ['data_{}.pt'.format(x) for x in self.train_idxs]
        elif self.data_phase == "validation":
            return ['data_{}.pt'.format(x) for x in self.validation_idxs]
        elif self.data_phase == "test":
            return ['data_{}.pt'.format(x) for x in self.test_idxs]
    
    def process(self):
        f = h5py.File(os.path.join(self.root, self.filename), 'r')
        
        events = f['davis']['left']['events'][1:]
        
        x = events[:, 0]
        y = events[:, 1]
        
        t = events[:, 2]
        t -= t.min()
        t *= 10 ** 4
        
        p = events[:, 3]
        p = p.astype(np.int32)
        
        i = 0
        
        max_timestep = t[-1]

        pos = np.vstack([x, y, t]).astype(np.float32).T

        for j in tqdm(range(int(max_timestep) // self.sample_timesteps)):
            if j * self.sample_timesteps < 50000:
                continue
            else:
                a = pos[:, 2] > j * self.sample_timesteps
                b = pos[:, 2] <= (j + 1) * self.sample_timesteps
                idx = a * b
                pos_crop = pos[idx]
                pol_crop = p[idx]
                if len(pos_crop) > 0:
                    pos_crop[:, 2] -= pos_crop[0, 2]

                    for x in range(self.img_size[0] // self.sample_patchsize[0]):
                        for y in range(self.img_size[1] // self.sample_patchsize[1]):
                            c = pos_crop[:, 0] > x * self.sample_patchsize[0]
                            d = pos_crop[:, 0] <= (x + 1) * self.sample_patchsize[0]
                            e = pos_crop[:, 1] > y * self.sample_patchsize[1]
                            f = pos_crop[:, 1] <= (y + 1) * self.sample_patchsize[1]

                            idx = c * d * e * f

                            pos_patch = pos_crop[idx]
                            pol_patch = pol_crop[idx]
                            
                            if sum(abs(pol_patch)) < 20:
                                continue
                                
                            else:
                                pos_patch[:, 0] -= x * self.sample_patchsize[0]
                                pos_patch[:, 1] -= y * self.sample_patchsize[1]

                                data = Data(x=torch.tensor(pol_patch, dtype=torch.float32),
                                         pos=torch.tensor(pos_patch, dtype=torch.float32))
                                
                                if self.pre_transform is not None:
                                    data = self.pre_transform(data)

                                torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
                                i += 1
                                
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
    
