import torch
from torch.utils.data import Dataset
import numpy as np
from transform_datasets.numpy.dvs import collapse_data, get_hot_pixels, erase_hot_pixels


class DVSMotion20(Dataset):
    
    def __init__(self,
                 start=0,
                 n_events=None,
                 sequence='classroom',
                 clean=False,
                 n_stds=5
                 ):

        filename = '/home/ssanborn/data/DVSMOTION20/camera-motion-data/{}_sequence/events.npy'.format(sequence)
        of_filename = '/home/ssanborn/data/DVSMOTION20/camera-motion-data/{}_sequence/optic_flow.npy'.format(sequence)
        
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
        
    @property
    def pos(self):
        return np.vstack([self.x, self.y, self.t]).T
    
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