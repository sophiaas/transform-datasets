import torch
from torch.utils.data import Dataset
import numpy as np
from transform_datasets.numpy.dvs import collapse_data, get_hot_pixels, erase_hot_pixels
from transform_datasets.numpy.dvs import dvs_to_vid as dvs_to_vid_numpy
import os


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


class DVS(Dataset):
    
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
    

class DVSMotion20(Dataset):
    
    def __init__(self,
                 start=0,
                 n_events=None,
                 sequence='classroom',
                 clean=False,
                 n_stds=5
                 ):

        filename = os.path.expanduser('~/data/DVSMOTION20/camera-motion-data/{}_sequence/events_clean.npy').format(sequence)
        of_filename = os.path.expanduser('~/data/DVSMOTION20/camera-motion-data/{}_sequence/optic_flow_clean.npy').format(sequence)
        
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
        

class MVSEC(DVS):
    def __init__(self, 
                 start=0,
                 n_events=None,
                 sequence='indoor_flying',
                 sequence_number=3,
                 clean=False):

        filename = os.path.expanduser('~/data/MVSEC/{}/{}{}_data.hdf5').format(sequence, sequence, sequence_number)
        of_filename = os.path.expanduser('~/data/MVSEC/{}/{}{}_gt_flow_dist.npz').format(sequence, sequence, sequence_number)
        
        self.img_size = (260, 346)
        
        with h5py.File(filename, 'r') as f:
            # TODO: Include right also?
            events = f['davis']['left']['events'][1:]
        
            x = events[:, 0]
            y = events[:, 1]
            t = events[:, 2] # seconds
            p = events[:, 3]        

        if n_events is None:
            n_events = len(x)
            
        x = x[start:start+n_events]
        y = y[start:start+n_events]
        t = t[start:start+n_events]
        p = p[start:start+n_events]
        
        t = t - t[0] # convert to reasonable range
        
        #TODO: extract only slice of flow if n_events is not None
        f = np.load(of_filename)
        flow_t = f['timestamps'] #seconds
        flow_x = f['x_flow_dist']
        flow_y = f['y_flow_dist']
        
        flow_t = flow_t - flow_t[0]
                    
        if clean:
            x, y, t, p = erase_hot_pixels(x, y, t, p)
                        
        self.x = x
        self.y = y
        self.t = t
        self.p = p
        
        self.flow_x = flow_x
        self.flow_y = flow_y
        self.flow_t = flow_t 
