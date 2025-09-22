from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np
import torch
import h5py
torch.set_default_dtype(torch.float32)

class CylinderDynamicsDataset(Dataset):
    def __init__(self,
                data_path: str,
                seq_length: int = 12,
                mean=None,
                std=None,
                ):

        self.data_path = data_path

        self.seq_length = seq_length
        
        self.data = np.load(self.data_path).astype(np.float32)
        print(f"Loaded Cylinder data with shape: {self.data.shape}")
        
        self.n_samples, self.frames, self.channel, self.H, self.W = self.data.shape
        
        if mean is None:
            mean_value = np.mean(self.data, axis=(0, 1, 3, 4))  # shape: [channel]
            self.mean = torch.from_numpy(mean_value.astype(np.float32))
        else:
            self.mean = mean

        if std is None:
            std_value = np.std(self.data, axis=(0, 1, 3, 4))  # shape: [channel]
            std_value[std_value < 1e-6] = 1.0  # avoid divide-by-zero
            self.std = torch.from_numpy(std_value.astype(np.float32))
        else:
            self.std = std
        
        self.num_per_sample = self.frames - self.seq_length

        self.total_sample = self.num_per_sample * self.n_samples

        self.create_data_set(self.data)
    
    def create_data_set(self, data):
        pool = []
        for i in range(self.n_samples):
            for j in range(self.num_per_sample):
                pool.append(data[i, j:j+self.seq_length+1])
        print('dataset total samples:', len(pool))
        self.pool = pool
    
    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        data = self.pool[idx]
        pre_seq = torch.tensor(data[:self.seq_length], dtype=torch.float32)
        post_seq = torch.tensor(data[1:self.seq_length+1], dtype=torch.float32)
        return self.normalize(pre_seq), self.normalize(post_seq)
    
    def normalize(self, x):
        mean = self.mean.reshape(1, -1, 1, 1)
        std = self.std.reshape(1, -1, 1, 1)
        return (x - mean) / std
    
    def denormalizer(self):
        def denormalize(x: torch.Tensor) -> torch.Tensor:
            mean = self.mean.reshape(1, -1, 1, 1)
            std = self.std.reshape(1, -1, 1, 1)
            try:
                return x * std + mean
            except:
                std_np = std.numpy()
                mean_np = mean.numpy()
                return x * std_np + mean_np
        return denormalize


class DamDynamicsDataset(Dataset):
    def __init__(self,
                data_path: str,
                seq_length: int = 12,
                mean=None,
                std=None,
                ):

        self.data_path = data_path

        self.seq_length = seq_length
        
        self.data = np.load(self.data_path).astype(np.float32)
        print(f"Loaded Cylinder data with shape: {self.data.shape}")
        
        self.n_samples, self.frames, self.channel, self.H, self.W = self.data.shape
        
        if mean is None:
            mean_value = np.mean(self.data, axis=(0, 1, 3, 4))  # shape: [channel]
            self.mean = torch.from_numpy(mean_value.astype(np.float32))
        else:
            self.mean = mean

        if std is None:
            std_value = np.std(self.data, axis=(0, 1, 3, 4))  # shape: [channel]
            std_value[std_value < 1e-6] = 1.0  # avoid divide-by-zero
            self.std = torch.from_numpy(std_value.astype(np.float32))
        else:
            self.std = std
        
        self.num_per_sample = self.frames - self.seq_length

        self.total_sample = self.num_per_sample * self.n_samples

        self.create_data_set(self.data)
    
    def create_data_set(self, data):
        pool = []
        for i in range(self.n_samples):
            for j in range(self.num_per_sample):
                pool.append(data[i, j:j+self.seq_length+1])
        print('dataset total samples:', len(pool))
        self.pool = pool
    
    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        data = self.pool[idx]
        pre_seq = torch.tensor(data[:self.seq_length], dtype=torch.float32)
        post_seq = torch.tensor(data[1:self.seq_length+1], dtype=torch.float32)
        return self.normalize(pre_seq), self.normalize(post_seq)
    
    def normalize(self, x):
        mean = self.mean.reshape(1, -1, 1, 1)
        std = self.std.reshape(1, -1, 1, 1)
        return (x - mean) / std
    
    def denormalizer(self):
        def denormalize(x: torch.Tensor) -> torch.Tensor:
            mean = self.mean.reshape(1, -1, 1, 1)
            std = self.std.reshape(1, -1, 1, 1)
            try:
                return x * std + mean
            except:
                std_np = std.numpy()
                mean_np = mean.numpy()
                return x * std_np + mean_np
        return denormalize


class ERA5Dataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 seq_length: int = 12,
                 min_path: str = None,
                 max_path: str = None):
        self.data_path = data_path
        self.seq_length = seq_length

        with h5py.File(self.data_path, 'r') as f:
            self.data = f['data'][:]  # shape: [N, H, W, C]
        print(f"Loaded ERA5 data with shape: {self.data.shape}")
        self.n_frames, self.H, self.W, self.C = self.data.shape

        if min_path is not None and max_path is not None:
            self.min = torch.from_numpy(np.load(min_path).astype(np.float32))  # shape: [C]
            self.max = torch.from_numpy(np.load(max_path).astype(np.float32))
        else:
            raise ValueError("min_path and max_path must be provided")

        self.num_per_sample = self.n_frames - seq_length
        self.create_data_set(self.data)

    def create_data_set(self, data):
        self.pool = []
        for i in range(self.num_per_sample):
            self.pool.append(data[i:i+self.seq_length+1])
        self.total_sample = len(self.pool)
        print('Dataset total samples:', len(self.pool))

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, idx):
        data = self.pool[idx]
        data = torch.from_numpy(data).permute(0, 3, 1, 2).float()

        pre_seq = data[:self.seq_length]
        post_seq = data[1:self.seq_length+1]

        return self.normalize(pre_seq), self.normalize(post_seq)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        min_v = self.min.reshape(1, -1, 1, 1)
        max_v = self.max.reshape(1, -1, 1, 1)
        return (x - min_v) / (max_v - min_v + 1e-6)

    def denormalizer(self):
        def denormalize(x: torch.Tensor) -> torch.Tensor:
            min_v = self.min.reshape(1, -1, 1, 1)
            max_v = self.max.reshape(1, -1, 1, 1)
            return x * (max_v - min_v + 1e-6) + min_v
        return denormalize


if __name__ == '__main__':

    Cylinder_data = CylinderDynamicsDataset(data_path="cylinder_train_data.npy",
                seq_length = 12,
                mean=None,
                std=None)
    
    print(Cylinder_data.mean)
    print(Cylinder_data.std)

    inx = 10
    print(Cylinder_data[inx][0].dtype)
    print(Cylinder_data[inx][1].dtype)

    print(Cylinder_data[inx][0].shape)
    print(Cylinder_data[inx][1].shape)

    print(Cylinder_data[inx][0].min())
    print(Cylinder_data[inx][1].max())

    print(Cylinder_data.total_sample)

    val_Cylinder_data = CylinderDynamicsDataset(data_path="cylinder_val_data.npy",
                seq_length = 12,
                mean=Cylinder_data.mean,
                std=Cylinder_data.std)
    
    print(val_Cylinder_data.mean)
    print(val_Cylinder_data.std)

    print(val_Cylinder_data[inx][0].shape)
    print(val_Cylinder_data[inx][1].shape)

    print(val_Cylinder_data[inx][0].min())
    print(val_Cylinder_data[inx][1].max())


    Dam_data = DamDynamicsDataset(data_path="dam_train_data.npy",
                seq_length = 12,
                mean=None,
                std=None)
    
    print(Dam_data.mean)
    print(Dam_data.std)

    inx = 10
    print(Dam_data[inx][0].dtype)
    print(Dam_data[inx][1].dtype)

    print(Dam_data[inx][0].shape)
    print(Dam_data[inx][1].shape)

    print(Dam_data[inx][0].min())
    print(Dam_data[inx][1].max())

    print(Dam_data.total_sample)

    val_Dam_data = DamDynamicsDataset(data_path="dam_val_data.npy",
                seq_length = 12,
                mean=Dam_data.mean,
                std=Dam_data.std)
    
    print(val_Dam_data.mean)
    print(val_Dam_data.std)

    print(val_Dam_data[inx][0].shape)
    print(val_Dam_data[inx][1].shape)

    print(val_Dam_data[inx][0].min())
    print(val_Dam_data[inx][1].max())
