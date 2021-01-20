from torch.utils.data import Dataset
import pandas as pd
from skimage import transform
from skimage.transform import EuclideanTransform


class TranslationMotionMNIST(Dataset):
    
    def __init__(self,
                 exemplars_per_digit,
                 n_frames=10,
                 digits=np.arange(10),
                 max_translation=20,
                 seed = 0):
       
        np.random.seed(seed)

        self.dim = 64
        
        mnist = np.array(pd.read_csv('/home/sanborn/projects/torch-polynomial-nets/datasets/mnist_test.csv'))
        all_labels = mnist[:, 0]
        label_idxs = {a: np.where(all_labels==a)[0] for a in digits}
        
        mnist = mnist[:, 1:]
        mnist = mnist.reshape((len(mnist), 28, 28))
        empty = np.zeros((len(mnist), self.dim, self.dim))
        empty[:, 18:18+28, 18:18+28] = mnist
        mnist = empty
        mnist = mnist / 255
        mnist = mnist - mnist.mean(axis=(1,2), keepdims=True)
        mnist = mnist / mnist.std(axis=(1,2), keepdims=True)
        

        
        start_x = np.random.randint(-max_translation, max_translation, size=len(digits)*exemplars_per_digit)
        start_y = np.random.randint(-max_translation, max_translation, size=len(digits)*exemplars_per_digit)
        end_x = np.random.randint(-max_translation, max_translation, size=len(digits)*exemplars_per_digit)
        end_y = np.random.randint(-max_translation, max_translation, size=len(digits)*exemplars_per_digit)

        data = []
        labels = []
        exemplar_labels = []
        translation = []
        
        i = 0
        
        for number in digits:
            # Select digits
            idxs = np.random.choice(label_idxs[number], 
                                    exemplars_per_digit, 
                                    replace=False)
            
            # Translate exemplars + Add noise
            for idx in idxs:
                img = mnist[idx]
                l = all_labels[idx]

                frames = []
                
                x_trajectory = np.linspace(start_x[i], end_x[i], n_frames)
                y_trajectory = np.linspace(start_y[i], end_y[i], n_frames)
                x_trajectory = [round(a) for a in x_trajectory]
                y_trajectory = [round(a) for a in y_trajectory]
                
                start_img = self.translate(img, x_trajectory[0], y_trajectory[0])
                frames.append(start_img)
                
                for j in range(1, n_frames):

                    next_img = self.translate(img, x_trajectory[j], y_trajectory[j])
                    frames.append(next_img)
                    
                vid = np.array(frames)
                data.append(vid)
                translation.append([(start_x[i], start_y[i]), (end_x[i], end_y[i])])
                labels.append(l)
                exemplar_labels.append(i)
                    
                i += 1
                    
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
    
    
class RotationMotionMNIST(Dataset):
    
    def __init__(self,
                 exemplars_per_digit,
                 n_frames=10,
                 digits=np.arange(10),
                 max_translation=20,
                 seed = 0):
       
        np.random.seed(seed)

        self.dim = 64
        
        mnist = np.array(pd.read_csv('/home/sanborn/projects/torch-polynomial-nets/datasets/mnist_test.csv'))
        all_labels = mnist[:, 0]
        label_idxs = {a: np.where(all_labels==a)[0] for a in digits}
        
        mnist = mnist[:, 1:]
        mnist = mnist / 255
#         mnist = mnist - mnist.mean(axis=1, keepdims=True)
#         mnist = mnist / mnist.std(axis=1, keepdims=True)
        mnist = mnist.reshape((len(mnist), 28, 28))
        
        start_angle = np.random.randint(0, 359, size=len(digits)*exemplars_per_digit)
        end_angle = np.random.randint(0, 359, size=len(digits)*exemplars_per_digit)

        data = []
        labels = []
        exemplar_labels = []
        rotation = []
        
        i = 0
        
        for number in digits:
            # Select digits
            idxs = np.random.choice(label_idxs[number], 
                                    exemplars_per_digit, 
                                    replace=False)
            
            # Translate exemplars + Add noise
            for idx in idxs:
                img = mnist[idx]
                l = all_labels[idx]

                frames = []
                
                rotation_trajectory = np.linspace(start_angle[i], end_angle[i], n_frames)
                rotation_trajectory = [round(a) for a in rotation_trajectory]
                
                start_img = transform.rotate(img, rotation_trajectory[0])   
                start_img -= start_img.mean(keepdims=True)
                start_img /= start_img.std(keepdims=True)
                frames.append(start_img)
                
                for j in range(1, n_frames):
                    next_img = transform.rotate(img, rotation_trajectory[j])    
                    next_img -= next_img.mean(keepdims=True)
                    next_img /= next_img.std(keepdims=True)
                    frames.append(next_img)

                    
                vid = np.array(frames)
                data.append(vid)
                rotation.append([(rotation_trajectory[0], rotation_trajectory[-1])])
                labels.append(l)
                exemplar_labels.append(i)
                    
                i += 1
                    
        data = np.array(data)
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