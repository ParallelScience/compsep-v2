# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
BASE = '/home/node/data/compsep_data/cut_maps'
class FLAMINGODataset(Dataset):
    def __init__(self, split, splits_file='data/splits.npz', stats_file=None, augment=False):
        self.split = split
        self.augment = augment
        splits = np.load(splits_file)
        self.indices = splits[split]
        self.frequencies = [90, 150, 217, 353, 545, 857]
        self.factors = {353: 1e6, 545: 1e6 * utils.jysr2uk(545), 857: 1e6 * utils.jysr2uk(857)}
        self.signals = {freq: np.load(BASE + '/stacked_' + str(freq) + '.npy', mmap_mode='r') for freq in self.frequencies}
        self.so_noise = {freq: np.load(BASE + '/so_noise/' + str(freq) + '.npy', mmap_mode='r') for freq in [90, 150, 217]}
        self.tsz = np.load(BASE + '/tsz.npy', mmap_mode='r')
        self.stats = None
        if stats_file and os.path.exists(stats_file):
            self.stats = np.load(stats_file)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        patch_idx = self.indices[idx]
        i_so = torch.randint(0, 3000, (1,)).item()
        i_planck = torch.randint(0, 100, (1,)).item()
        x = np.zeros((len(self.frequencies), 256, 256), dtype=np.float32)
        for c, freq in enumerate(self.frequencies):
            sig = np.array(self.signals[freq][patch_idx])
            if freq <= 217:
                noise = np.array(self.so_noise[freq][i_so])
            else:
                raw_noise = np.array(np.load(BASE + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy', mmap_mode='r')[patch_idx])
                noise = raw_noise * self.factors[freq]
            x[c] = sig + noise
        y = np.array(self.tsz[patch_idx], dtype=np.float32)
        y = np.expand_dims(y, axis=0)
        if self.stats is not None:
            x = (x - self.stats['x_mean'][:, None, None]) / self.stats['x_std'][:, None, None]
            y = (y - self.stats['y_mean']) / self.stats['y_std']
        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x = np.rot90(x, k, axes=(1, 2)).copy()
                y = np.rot90(y, k, axes=(1, 2)).copy()
            if torch.rand(1).item() > 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=1).copy()
            if torch.rand(1).item() > 0.5:
                x = np.flip(x, axis=2).copy()
                y = np.flip(y, axis=2).copy()
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
def compute_statistics(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    x_sum = np.zeros((6,), dtype=np.float64)
    x_sq_sum = np.zeros((6,), dtype=np.float64)
    y_sum = 0.0
    y_sq_sum = 0.0
    num_pixels = 0
    for i, (x, y) in enumerate(loader):
        B = x.shape[0]
        pixels_in_batch = B * 256 * 256
        num_pixels += pixels_in_batch
        x_sum += x.sum(dim=(0, 2, 3)).numpy().astype(np.float64)
        x_sq_sum += (x ** 2).sum(dim=(0, 2, 3)).numpy().astype(np.float64)
        y_sum += y.sum().item()
        y_sq_sum += (y ** 2).sum().item()
    x_mean = x_sum / num_pixels
    x_var = np.maximum((x_sq_sum / num_pixels) - (x_mean ** 2), 1e-12)
    x_std = np.sqrt(x_var)
    y_mean = y_sum / num_pixels
    y_var = np.maximum((y_sq_sum / num_pixels) - (y_mean ** 2), 1e-12)
    y_std = np.sqrt(y_var)
    return x_mean, x_std, np.array([y_mean]), np.array([y_std])
if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    n_patch = 1523
    rng = np.random.default_rng(seed=42)
    indices = np.arange(n_patch)
    rng.shuffle(indices)
    n_train = int(0.8 * n_patch)
    n_val = int(0.1 * n_patch)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    splits_file = 'data/splits.npz'
    np.savez(splits_file, train=train_indices, val=val_indices, test=test_indices)
    stats_file = 'data/normalization_stats.npz'
    train_dataset = FLAMINGODataset('train', splits_file=splits_file, stats_file=None, augment=False)
    x_mean, x_std, y_mean, y_std = compute_statistics(train_dataset)
    np.savez(stats_file, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)