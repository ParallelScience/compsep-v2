# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data/')
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
os.environ['OMP_NUM_THREADS'] = '8'
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
BASE_DIR = '/home/node/data/compsep_data/cut_maps'
DATA_DIR = 'data'
STATS_PATH = os.path.join(DATA_DIR, 'channel_stats.npz')
def compute_and_cache_stats(base_dir, stats_path):
    if os.path.exists(stats_path):
        print('Stats already exist. Skipping computation.')
        return
    print('Computing global statistics (without noise)...')
    frequencies = [90, 150, 217, 353, 545, 857]
    mean_x = np.zeros(6, dtype=np.float32)
    std_x = np.zeros(6, dtype=np.float32)
    for c, freq in enumerate(frequencies):
        signal = np.load(base_dir + '/stacked_' + str(freq) + '.npy')
        mean_x[c] = np.mean(signal)
        std_x[c] = np.std(signal)
        print('Freq ' + str(freq) + ' GHz: mean=' + str(mean_x[c]) + ', std=' + str(std_x[c]))
    tsz = np.load(base_dir + '/tsz.npy')
    mean_y = np.mean(tsz)
    std_y = np.std(tsz)
    print('tSZ: mean=' + str(mean_y) + ', std=' + str(std_y))
    np.savez(stats_path, mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y)
    print('Statistics saved to ' + stats_path)
class CompSepDataset(Dataset):
    def __init__(self, base_dir, stats_path, transform=True):
        self.base_dir = base_dir
        self.transform = transform
        self.frequencies = [90, 150, 217, 353, 545, 857]
        self.n_patch = 1523
        self.n_planck = 100
        self.n_so = 3000
        stats = np.load(stats_path)
        self.mean_x = stats['mean_x']
        self.std_x = stats['std_x']
        self.mean_y = stats['mean_y']
        self.std_y = stats['std_y']
        self.signals = {}
        for freq in self.frequencies:
            self.signals[freq] = np.load(base_dir + '/stacked_' + str(freq) + '.npy', mmap_mode='r')
        self.tsz = np.load(base_dir + '/tsz.npy', mmap_mode='r')
        self.so_noise = {}
        for freq in [90, 150, 217]:
            self.so_noise[freq] = np.load(base_dir + '/so_noise/' + str(freq) + '.npy', mmap_mode='r')
    def __len__(self):
        return self.n_patch
    def __getitem__(self, idx):
        i_so = np.random.randint(self.n_so)
        i_planck = np.random.randint(self.n_planck)
        x = np.zeros((6, 256, 256), dtype=np.float32)
        for c, freq in enumerate(self.frequencies):
            signal = self.signals[freq][idx]
            if freq <= 217:
                noise = self.so_noise[freq][i_so]
            else:
                noise_path = self.base_dir + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy'
                raw_noise = np.load(noise_path, mmap_mode='r')[idx]
                if freq == 353:
                    noise = raw_noise * 1e6
                else:
                    noise = raw_noise * 1e6 * utils.jysr2uk(freq)
            if self.transform:
                noise_scale = np.random.uniform(0.9, 1.1)
                noise = noise * noise_scale
            x[c] = signal + noise
        y = self.tsz[idx].astype(np.float32)
        if self.transform:
            k = np.random.randint(4)
            if k > 0:
                x = np.rot90(x, k, axes=(1, 2))
                y = np.rot90(y, k, axes=(0, 1))
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=0)
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=1)
        x = (x - self.mean_x[:, None, None]) / self.std_x[:, None, None]
        y = (y - self.mean_y) / self.std_y
        y = np.expand_dims(y, axis=0)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
if __name__ == '__main__':
    compute_and_cache_stats(BASE_DIR, STATS_PATH)
    dataset = CompSepDataset(BASE_DIR, STATS_PATH, transform=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    print('Fetching a sample batch...')
    for batch_x, batch_y in dataloader:
        print('Batch X shape: ' + str(batch_x.shape))
        print('Batch Y shape: ' + str(batch_y.shape))
        print('Batch X stats (mean, std):')
        for c in range(6):
            mean_val = batch_x[:, c, :, :].mean().item()
            std_val = batch_x[:, c, :, :].std().item()
            print('  Channel ' + str(c) + ': mean=' + str(mean_val) + ', std=' + str(std_val))
        mean_y_val = batch_y.mean().item()
        std_y_val = batch_y.std().item()
        print('Batch Y stats: mean=' + str(mean_y_val) + ', std=' + str(std_y_val))
        break