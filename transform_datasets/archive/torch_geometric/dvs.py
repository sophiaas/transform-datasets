import os
import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import pdb
from tqdm import tqdm


class NormalizeRanges(object):
    def __init__(self, pos_range=(-30, 30)):
        self.pos_range = pos_range
        
    def __call__(self, data):
        if data.pos.shape[0] == 0:
            return data
        
        pos_min, pos_max = self.pos_range
        
        if data.pos.shape[0] == 1:
            min_timestep = data.pos[:, 2][0]
        else:
            min_timestep = data.pos[:, 2].min()
        
        min_vals = torch.Tensor([pos_min, pos_min, min_timestep])
        new_pos = data.pos - min_vals

        max_timestep = new_pos[:, 2].max()
        new_pos[:, :2] /= pos_max - pos_min
        
        if data.pos.shape[0] != 1:
            new_pos[:, 2] /= max_timestep
            new_pos = new_pos * 2 - 1
        else:
            new_pos[:, :2] = new_pos[:, :2] * 2 - 1

        data.pos = new_pos
        return data

    
class NormalizeByDim(object):
    """
    No good
    """
    def __init__(self, frame_size=(33, 33)):
        self.frame_size = frame_size
        
    def __call__(self, data):
        if data.pos.shape[0] == 0:
            return data
        
        else:
            x_max, y_max = self.frame_size

            min_timestep = data.pos[:, 2].min()
            min_vals = torch.Tensor([0, 0, min_timestep])
            new_pos = data.pos - min_vals

            max_timestep = new_pos[:, 2].max()
            max_vals = torch.Tensor([x_max, y_max, max_timestep])
            new_pos = new_pos / max_vals

            new_pos = new_pos * 2 - 1
            data.pos = new_pos
            return data
        

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
    

class Shapes(Dataset):
    def __init__(self, 
                 root,
                 filename,
                 data_phase='train',
                 sample_time=0.5, #in milliseconds
                 sample_patchsize=(20, 20),
                 img_size=(346, 260),
                 pre_transform=NormalizeRanges((0, 19))):
            
        np.random.seed(0)
        self.root = root
        self.filename = filename
        
        self.sample_time = sample_time
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
        processed_files = os.listdir(self.processed_dir)
        self.n_datapoints = len([x for x in processed_files if x[:4] == 'data'])
        self.n_test = len([x for x in processed_files if x[:4] == 'test'])

        idxs = np.arange(self.n_datapoints)
        np.random.shuffle(idxs)
        
        n_train = int(self.n_datapoints * 0.9)
        n_validate = int(self.n_datapoints * 0.1)
        
        train_idxs = idxs[:n_train]
        validation_idxs = idxs[n_train:]
        
        if self.data_phase == "train":
            self.idxs = train_idxs
            return ['data_{}.pt'.format(x) for x in train_idxs]
        elif self.data_phase == "validation":
            self.idxs = validation_idxs
            return ['data_{}.pt'.format(x) for x in validation_idxs]
        elif self.data_phase == "test":
            self.idxs = list(range(self.n_test))
            return ['test_{}.pt'.format(x) for x in range(self.n_test)]
    
    def process(self):
        df = pd.read_csv(os.path.join(self.root, self.filename), sep=" ", header=None)
        
        x = np.array(df[1])
        y = np.array(df[2])
        t = np.array(df[0])

        t -= t.min()
        
        p = np.array(df[3]) * 2 - 1
        
        i = 0
        test_idx = 0
        
        max_time = t[-1]

        pos = np.vstack([x, y, t]).astype(np.float32).T
        
        random_test_idx = np.random.randint(max_time // self.sample_time)

        for j in tqdm(range(int(max_time // self.sample_time))):
            a = pos[:, 2] > j * self.sample_time
            b = pos[:, 2] <= (j + 1) * self.sample_time
            idx = a * b
            pos_crop = pos[idx]
            pol_crop = p[idx]
            if len(pos_crop) > 0:
                pos_crop[:, 2] -= pos_crop[0, 2]

                for x in range(self.img_size[0] // self.sample_patchsize[0]):
                    for y in range(self.img_size[1] // self.sample_patchsize[1]):
                        c = pos_crop[:, 0] >= x * self.sample_patchsize[0]
                        d = pos_crop[:, 0] < (x + 1) * self.sample_patchsize[0]
                        e = pos_crop[:, 1] >= y * self.sample_patchsize[1]
                        f = pos_crop[:, 1] < (y + 1) * self.sample_patchsize[1]

                        idx = c * d * e * f

                        pos_patch = pos_crop[idx]
                        pol_patch = pol_crop[idx]
                        
                        if j == random_test_idx:
                            pos_patch[:, 0] -= x * self.sample_patchsize[0]
                            pos_patch[:, 1] -= y * self.sample_patchsize[1]

                            data = Data(x=torch.tensor(pol_patch, dtype=torch.float32),
                                     pos=torch.tensor(pos_patch, dtype=torch.float32))

                            if self.pre_transform is not None:
                                data = self.pre_transform(data)

                            torch.save(data, os.path.join(self.processed_dir, 'test_{}.pt'.format(test_idx)))
                            test_idx += 1
                          
                        else:
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
        i = self.idxs[idx]
        if self.data_phase == 'test':
            return torch.load(os.path.join(self.processed_dir, 'test_{}.pt'.format(i)))
        else:
            return torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))

        
class NMNIST(Dataset):
    def __init__(self, 
                 root, 
                 data_phase='train', 
                 transform=None, 
                 pre_transform=NormalizeByDim(), 
                 saccade_length=100,
                 dataset_type='categorization',
                 window_size=10,
                 stride=5):
        """
        dataset_type may be either 'categorization', 'frame_prediction', or 'flow'
        window_size and stride are parameters for a frame prediction dataset
        """
#         if train:
#             self.root = os.path.join(root, 'train')
#         else:
#             self.root = os.path.join(root, 'test')

        self.root = os.path.join(root, 'test')
        self.data_phase = data_phase
            
        np.random.seed(0)
        idxs = np.arange(10000)
        np.random.shuffle(idxs)
        self.train_idxs = idxs[:1000]
        self.validation_idxs = idxs[1000:1100]
        self.test_idxs = idxs[1100:2100]
        self.sample_length = 350
        self.dataset_type = dataset_type
        self.saccade_length = saccade_length
        self.window_size = window_size
        self.stride = stride
        self.label_path = os.path.join(self.root, 'labels.txt')

        print(self.root)

        super().__init__(self.root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        path = os.path.join(self.root, 'raw')
        return os.listdir(path)
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dataset_type)
        
    @property
    def processed_file_names(self):
        if self.data_phase == 'train':
            return ['data_{}.pt'.format(x) for x in self.train_idxs]
        elif self.data_phase == 'validation':
            return ['data_{}.pt'.format(x) for x in self.validation_idxs]
        elif self.data_phase == 'test':
            return ['data_{}.pt'.format(x) for x in self.test_idxs]
        else:
            raise NotImplementedError('{} is not yet set up'.format(self.data_phase)) 
            
    def process(self):
        self.labels = np.loadtxt(self.label_path).astype(int)
        
        i = 0
        for raw_path in tqdm(self.raw_paths):
            # Read data from `raw_path`.
            index = int(raw_path.split('/')[-1].split('.')[0]) - 1
            with open(raw_path, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.uint8).astype('uint')
                
            label = self.labels[index]

            x = raw_data[0::5]
            y = raw_data[1::5]
            t = ((raw_data[2::5] << 16) | (raw_data[3::5] << 8) | (raw_data[4::5]) ) & 0x7FFFFF
            t = t // 1000

            pos = np.vstack([x, y, t]).astype(np.float32).T

            polarity = raw_data[2::5] >> 7
            polarity = polarity.astype(np.float32)
            polarity *= 2
            polarity -= 1
            
            for j in range(self.sample_length // self.saccade_length):
                a = pos[:, 2] > j * self.saccade_length
                b = pos[:, 2] <= (j + 1) * self.saccade_length
                idx = a * b
                pos_crop = pos[idx]
                pol_crop = polarity[idx]
                pos_crop[:, 2] -= pos_crop[0, 2]
                
                if self.dataset_type == 'frame_prediction':
                    current_pos_slice = None
                    current_pol_slice = None

                    d = []

                    # Init
                    start_idx = 0
                    next_start_idx = -1
                    end_idx = -1
                    current_val = pos[0, 2]

                    for k in np.arange(self.saccade_length // self.stride):
                        start = k * self.stride
                        end_val = start + self.window_size
                        if end_val <= total_length:
                            while current_val < end_val and end_idx < len(pos_crop) - 1:
                                if current_val <= (k + 1) * self.stride:
                                    next_start_idx = end_idx
                                end_idx += 1
                                current_val = pos_crop[end_idx, 2]

                            next_pos_slice = pos_crop[start_idx:end_idx]
                            next_pol_slice = pol_crop[start_idx:end_idx]

                            start_idx = next_start_idx
                            end_idx = start_idx

                        if current_pos_slice is not None and len(next_pos_slice > 0 ):
                            next_frame =  Data(x=torch.tensor(next_pol_slice, dtype=torch.float32),
                                     pos=torch.tensor(next_pos_slice, dtype=torch.float32))

                            if self.pre_transform is not None:
                                next_frame = self.pre_transform(next_frame)

                            data = Data(x=torch.tensor(current_pol_slice, dtype=torch.float32),
                                     pos=torch.tensor(current_pos_slice, dtype=torch.float32),
                                     y=next_frame, label=torch.tensor([label], dtype=torch.long))

                            if self.pre_transform is not None:
                                data = self.pre_transform(data)

                            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))

                            i += 1

                        current_pol_slice = next_pol_slice
                        current_pos_slice = next_pos_slice
                        
                elif self.dataset_type == 'flow':
                    transformation = np.random.uniform(-0.15, 0.15, size=(2))
                    

                    coeffs = self.saccade_length - pos_crop[:, 2]
                    
                    t = np.array([c * transformation for c in coeffs])

                    transformed_pos = pos_crop[:, :2] + t

                    transformed_pos = np.hstack([transformed_pos, np.expand_dims(pos_crop[:, 2], 1)])
                    
                    
                    data = Data(x=torch.tensor(pol_crop, dtype=torch.float32), 
                                pos=torch.tensor(transformed_pos, dtype=torch.float32), 
                                y=torch.tensor(transformation, dtype=torch.float32),
                                label=torch.tensor([label], dtype=torch.long),
                                original=torch.tensor(pos_crop, dtype=torch.float32))
                    
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
                    i += 1


                else:
                    data = Data(x=torch.tensor(pol_crop, dtype=torch.float32),
                             pos=torch.tensor(pos_crop, dtype=torch.float32),
                             y=torch.tensor([label-1], dtype=torch.long),
                             saccade=torch.tensor([j], dtype=torch.long))

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
                    i += 1
            
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
    
    
import bisect
import struct
import h5py
import glob
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.transforms import Center, NormalizeScale, RandomTranslate, RandomScale
import os

"""
- For each datapoint, plot original 3D gesture, then gestures transformed by learned 3x3 matrix
- dvs mnist, split samples into 100 ms intervals (1st, 2nd, 3rd saccade), show them separately to the network. see if it can learn which saccade with the transform matrix
- Tracking problems, high speed low latency, seeing bullets
- Prediction at different length scales
"""

def one_hot(idx, max_idx):
    out = np.zeros((len(idx), max_idx), dtype=np.int8)
    for row, i in enumerate(idx):
        out[row, i] = 1
    return out


class DataAugmentation(object):
    def __init__(self):
        self.translate = RandomTranslate(0.1)
        self.scale = RandomScale((0.9, 1.1))
        
    def __call__(self, data):
        translated = self.translate(data)
        scaled = self.scale(translated)
        return scaled

    
class CenterNormalizeByDim(object):
    def __init__(self):
        self.center = Center()
        
    def __call__(self, data):
        data = self.center(data)
        max_val, max_idx =  data.pos.abs().max(axis=0)
        scale = (1 / max_val) * 0.999999
        data.pos = data.pos * scale
        return data
    
    
class DVSGestures(InMemoryDataset):
    
    def __init__(self, 
                 root, 
                 train=True, 
                 transform=DataAugmentation(), 
                 pre_transform=CenterNormalizeByDim()):
        
        self.n_classes = 11
        self.load_train = train
        
        super().__init__(root, transform, pre_transform)
        
        self.root = root
        self.data = torch.load(self.processed_paths[0])
#         self.data, self.slices = torch.load(self.processed_paths[0])
#         self.train = torch.load(self.processed_paths[0])
#         self.test = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ['dvs_gestures_events.hdf5']

    @property
    def processed_file_names(self):
        if self.load_train:
            return ['train.pt']
        else:
            return ['test.pt']
    
    def process(self):
        f = h5py.File(os.path.join(self.root, 'raw', self.raw_file_names[0]), 'r', swmr=True, libver="latest")
        
        if self.load_train:
            data = self.build_dataset(f, "train")
        else:
            data = self.build_dataset(f, "test")

        if self.pre_transform is not None:
            data = [self.pre_transform(d) for d in data]
            
        torch.save(data, self.processed_paths[0])
        
    def build_dataset(self,
                     f,
                     group_name="train", 
                     n_classes = 11, 
                     n_events=1500,
                     offset = 0):
        
        hdf5_group = f[group_name]
        
        subjects = {k:range(len(v['labels'][()])) for k, v in hdf5_group.items()}
        dataset = []
    
        for subject, data in tqdm(hdf5_group.items()):
            class_partitions = data['labels'][()]
            times = data['time'][()]
            events = data['data']

            for partition in class_partitions:

                label = partition[0]
                start_time = partition[1]
                end_time = partition[2]
    
                slice_start = self.find_first(times, start_time)
                slice_end = slice_start + n_events
#                 slice_end = self.find_first(times[slice_start:], end_time) + slice_start

                event_slice = events[slice_start:slice_end]
                time_slice = times[slice_start:slice_end]
                time_slice = (time_slice - time_slice[0])
                
                pos = np.vstack([event_slice[:, 0], event_slice[:, 1], time_slice]).astype(np.int32).T

                polarity = one_hot(event_slice[:, 2], 2)
#                 polarity = event_slice[:, 2].astype(np.int32) * 2 - 1
                
#                 l = one_hot([label-1], self.n_classes)
                
                d = Data(x=torch.tensor(polarity, dtype=torch.float32),
                         pos=torch.tensor(pos, dtype=torch.float32),
                         y=torch.tensor([label-1], dtype=torch.long))

                dataset.append(d)
                
        return dataset
    
    def find_first(self, a, tgt):
        return bisect.bisect_left(a, tgt)
        

    def get(self, idx):
        return self.data[idx]
#         data = torch.load(os.path.join(self.processed_dir, '{}.pt'.format(name)))
#         return data
    
    def len(self):
        return len(self.data)
    