import os, sys
import utils.kernels
import numpy as np
from utils.kernels import *
from sklearn.model_selection import train_test_split


class SwissRoll():
    def __init__(self, n_samples=1000, noise=0.0, test_size=0.8, random_state=0, hole=True):
        b = 0.1
        generator = np.random.RandomState(0)
        
        # extra_l1 = 2 + 1.5 * generator.uniform(size=8)
        # extra_l2 = 12.5 + 1.5 * generator.uniform(size=8)
        # extra_z = 2 + 2 * generator.uniform(size=16)

        l = -2 + 12 * generator.uniform(size=2 * n_samples)
        z = 6 * generator.uniform(size=2 * n_samples)
        # l = np.concatenate((l, extra_l1, extra_l2))
        # z = np.concatenate((z, extra_z))

        idx = np.arange(len(l))
        generator.shuffle(idx)
        l = l[idx]
        z = z[idx]
        t = (np.log(b * l / np.sqrt(1 + b ** 2) + 1)) / b
        x = np.exp(b * t) * np.cos(t)
        y = np.exp(b * t) * np.sin(t)
        
        self.X = np.array([x, y, z]).T
        self.latent = np.array([l, z]).T
        self.color = t

        if hole:
            hole_range = [2.5, 3.5, 1.5, 6.5]
            non_hole_idx = ((z > hole_range[0]) & 
                            (z < hole_range[1]) & 
                            (l > hole_range[2]) & 
                            (l < hole_range[3]))
            self.X, self.color, self.latent = self.X[~non_hole_idx], self.color[~non_hole_idx], self.latent[~non_hole_idx]
            self.X, self.color, self.latent = self.X[:n_samples], self.color[:n_samples], self.latent[:n_samples]

        if noise > 0:
            self.X += noise * generator.randn(n_samples, 3)
        self.X_train, self.X_test, self.color_train, self.color_test, self.latent_train, self.latent_test = train_test_split(
            self.X, self.color, self.latent, test_size=test_size, random_state=random_state)

    def get_data(self, type='train'):
        if type == 'train':
            return self.X_train, self.color_train
        elif type == 'test':
            return self.X_test, self.color_test
        elif type == 'all':
            return self.X, self.color


class dSprites:
    def __init__(self, N=10000, test_size=0.5, shape='all', 
                 fix_scale=False, fix_orientation=False, fix_x=True, fix_y=True, XY_hole=False, **kwargs):
        verbose = kwargs.get('verbose', True)
        
        dataset_path = 'data/dSprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        try:
            dataset = np.load(dataset_path)
        except FileNotFoundError:
            print(f'Fail to load the data file: {dataset_path}')
        
        imgs = dataset['imgs']
        latents_values = dataset['latents_values']
        
        ''' 
        imgs : numpy.array, (737280, 64, 64)
        
        latent values
            0: Color, White(1) only
            1: Shape, Square(1), Ellipse(2), Heart(3)       # becomes 0-th axis when no color
            2: Scale, np.linspace(0.5, 1, 6)                # becomes 1-th axis when no color
            3: Orientation, np.linspace(0, 2*np.pi, 40)     # becomes 2-th axis when no color
            4: PosX, np.linspace(0, 1, 32)                  # becomes 3-th axis when no color
            5: PosY, np.linspace(0, 1, 32)                  # becomes 4-th axis when no color
        '''
        
        latents_sizes = np.array([1, 3, 6, 40, 32, 32])
        
        if shape == 'all':
            pass
        else:
            latents_sizes[1] = 1
            if shape == 'square':
                imgs = imgs[latents_values[:, 1] == 1]
                latents_values = latents_values[latents_values[:, 1] == 1]
            elif shape == 'ellipse':
                imgs = imgs[latents_values[:, 1] == 2]
                latents_values = latents_values[latents_values[:, 1] == 2]
            elif shape == 'heart':
                imgs = imgs[latents_values[:, 1] == 3]
                latents_values = latents_values[latents_values[:, 1] == 3]
            else:
                print("shape must be one of ['all', 'square', 'ellipse', 'heart']")
                raise NotImplementedError
        
        if fix_scale:
            latents_sizes[2] = 1
            imgs = imgs[latents_values[:, 2] == 1]
            latents_values = latents_values[latents_values[:, 2] == 1]
        
        if fix_orientation:
            latents_sizes[3] = 1
            imgs = imgs[latents_values[:, 3] == 0]
            latents_values = latents_values[latents_values[:, 3] == 0]
        else:
            latents_sizes[3] = 39
            imgs = imgs[latents_values[:, 3] != 0]
            latents_values = latents_values[latents_values[:, 3] != 0]
        
        if fix_x:
            latents_sizes[4] = 1
            imgs = imgs[latents_values[:, 4] == 0]
            latents_values = latents_values[latents_values[:, 4] == 16 / 31]
        
        if fix_y:
            latents_sizes[5] = 1
            imgs = imgs[latents_values[:, 5] == 0]
            latents_values = latents_values[latents_values[:, 5] == 16 / 31]
        
        if N > len(imgs):
            if verbose:
                print(f"Warning: N ({N}) is larger than the number of images ({len(imgs)}). N is set to {len(imgs)}")
            N = len(imgs)
            sampled_indices = np.random.permutation(N)
        else:
            latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))
            samples = np.zeros((N, latents_sizes.size))
            for lat_i, lat_size in enumerate(latents_sizes):
                samples[:, lat_i] = np.random.randint(lat_size, size=N)
            sampled_indices = np.dot(samples, latents_bases).astype(int)
            
        data = imgs[sampled_indices].reshape(-1, 1, 64, 64)
        targets = latents_values[sampled_indices, 1:]
        
        if XY_hole:
            hole_range = [0.2, 0.8, 0.4, 0.6]
            hole_idx = ((targets[:, 3] > hole_range[0]) & (targets[:, 3] < hole_range[1]) & \
                        (targets[:, 4] > hole_range[2]) & (targets[:, 4] < hole_range[3])) | \
                       ((targets[:, 4] > hole_range[0]) & (targets[:, 4] < hole_range[1]) & \
                        (targets[:, 3] > hole_range[2]) & (targets[:, 3] < hole_range[3]))
            non_hole_idx = ~hole_idx
            data = data[non_hole_idx]
            targets = targets[non_hole_idx]
            N = len(data)
        
        split_train_val_test = (1 - test_size, test_size)
        
        num_train_data = int(N * split_train_val_test[0])
        
        idx = np.random.permutation(N)
        train_idx = idx[:num_train_data]
        test_idx = idx[num_train_data:]
        
        self.data_train, self.targets_train = data[train_idx], targets[train_idx]
        self.data_test, self.targets_test = data[test_idx], targets[test_idx]
        self.data, self.targets = data, targets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y, idx
    
    def get_data(self, type='all'):
        if type == 'train':
            return self.data_train, self.targets_train
        elif type == 'test':
            return self.data_test, self.targets_test
        elif type == 'all':
            return self.data, self.targets
        else:
            raise ValueError("type must be one of ['train', 'test', 'all']")