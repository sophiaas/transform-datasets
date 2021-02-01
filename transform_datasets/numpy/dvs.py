import re
import numpy as np
import pandas as pd

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def read_event_txt(filename, savename=None):
    events = pd.read_csv(filename, header=None)
    events = list(events[0])
    events_parsed = []
    for e in events:
        t, x, y, p = e.split(' ')
        p = int(p)
        if p == 0:
            p = -1
        events_parsed.append([int(x), int(y), float(t), p])

    events = np.array(events_parsed)
    if savename is not None:
        np.save(savename, events)
    return events

import torch
from torch.utils.data import Dataset
import numpy as np


def collapse_data(x, y=None, t=None, p=None, img_size=(260, 346)):
    
    if y is None:
        y = x[:, 1]
        t = x[:, 2]
        p = x[:, 3]
        x = x[:, 0]
        
    img = np.zeros(img_size)
    for i in range(len(x)):
        img[int(y[i]), int(x[i])] += p[i]
        
    return img
      
    
def get_hot_pixels(x, y=None, t=None, p=None, img_size=(260, 346), n_stds=5):
    
    if y is None:
        y = x[:, 1]
        t = x[:, 2]
        p = x[:, 3]
        x = x[:, 0]
        
    img = collapse_data(x, y, t, p, img_size)
    img = abs(img).ravel()
    sorted_idxs = np.argsort(img[::-1])
    hot_pixels = []
    for idx in sorted_idxs:
        if img[idx] > img.mean() + n_stds * img.std():
            hot_pixels.append(np.unravel_index(idx, img_size))
    return hot_pixels


def erase_hot_pixels(x, y=None, t=None, p=None, flow=None, img_size=(260, 346), n_stds=5):
        
    if y is None:
        y = x[:, 1]
        t = x[:, 2]
        p = x[:, 3]
        x = x[:, 0]
        
    hot_pixels = get_hot_pixels(x, y, t, p, img_size, n_stds)
    print(len(hot_pixels))
    x_, y_, t_, p_ = [], [], [], []
    
    if flow is not None:
        flow_ = []
        
    for i in range(len(x)):
        
        exclude = False
        
        for idx in hot_pixels:
            y_idx, x_idx = idx
            if x[i] == x_idx and y[i] == y_idx: 
                exclude = True
                break
                
        if exclude:
            continue
            
        else:
            x_.append(x[i])
            y_.append(y[i])
            t_.append(t[i])
            p_.append(p[i])
            if flow is not None:
                flow_.append(flow[i])
                
    x_ = np.array(x_)
    y_ = np.array(y_)
    t_ = np.array(t_)
    p_ = np.array(p_)

    if flow is not None:
        flow_ = np.array(flow_)
        return x_, y_, t_, p_, flow_

    else:
        return x_, y_, t_, p_
            
        
def dvs_to_vid(x, y=None, t=None, p=None, frame_rate=120, img_size=(260, 346), normalize=True):
    if y is None:
        y = x[:, 1]
        t = x[:, 2]
        p = x[:, 3]
        x = x[:, 0]
        
    projections = []
    start = t[0]
    end = t[-1]
    max_idx = len(t)
    idx = 0
    for s in np.arange(start, end, 1/frame_rate):
        n_events = 0
        if t[idx] >= s and t[idx] < s + 1/frame_rate:
            in_timestep = True
            img = np.zeros(img_size)
            while in_timestep:
                img[int(y[idx]), int(x[idx])] += p[idx]
                idx += 1
                n_events += 1
                if idx == max_idx:
                    break
                if t[idx] >= s + 1/frame_rate:
                    in_timestep = False
        if normalize:
            img /= n_events
        if idx == max_idx:
            break
        projections.append(img)
    return np.array(projections)
    

class DVSMotion20(object):
    
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
            
        self.x = data[start:start+n_events, 0] 
        self.y = data[start:start+n_events, 1]
        self.t = data[start:start+n_events, 2]
        self.p = data[start:start+n_events, 3]
        self.optic_flow = optic_flow[start:start+n_events]
        
        if clean:
            self.x, self.y, self.t, self.p, self.optic_flow = erase_hot_pixels(self.x, self.y, self.t, self.p, self.optic_flow)
            
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