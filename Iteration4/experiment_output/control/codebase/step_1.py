# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import torch
from torch.utils.data import Dataset
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
class CompSepDataset(Dataset):
    def __init__(self, patch_indices, split='train', seed=42):
        self.patch_indices = patch_indices
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.frequencies = [90, 150, 217, 353, 545, 857]
        self.base_dir = '/home/node/data/compsep_data/cut_maps'
        with open('data/norm_stats.json', 'r') as f:
            self.norm_stats = json.load(f)
        self.signals = {f: np.load(self.base_dir + '/stacked_' + str(f) + '.npy', mmap_mode='r') for f in self.frequencies}
        self.tsz = np.load(self.base_dir + '/tsz.npy', mmap_mode='r')
        self.so_noise = {f: np.load(self.base_dir + '/so_noise/' + str(f) + '.npy', mmap_mode='r') for f in [90, 150, 217]}
    def __len__(self):
        return len(self.patch_indices)
    def __getitem__(self, idx):
        p_idx = self.patch_indices[idx]
        i_so = self.rng.integers(0, 3000)
        i_planck = self.rng.integers(0, 100)
        x = np.zeros((len(self.frequencies), 256, 256), dtype=np.float32)
        noise_vars = np.zeros(len(self.frequencies), dtype=np.float32)
        for i, freq in enumerate(self.frequencies):
            signal = self.signals[freq][p_idx]
            if freq <= 217:
                noise = self.so_noise[freq][i_so]
            else:
                raw = np.load(self.base_dir + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy', mmap_mode='r')[p_idx]
                if freq == 353:
                    noise = raw * 1e6
                else:
                    noise = raw * 1e6 * utils.jysr2uk(freq)
            noisy_signal = signal + noise
            noise_vars[i] = np.var(noise)
            median = self.norm_stats[str(freq)]['median']
            iqr = self.norm_stats[str(freq)]['iqr']
            x[i] = (noisy_signal - median) / iqr
        y_tsz = self.tsz[p_idx].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(noise_vars), torch.from_numpy(y_tsz)
if __name__ == '__main__':
    BASE = '/home/node/data/compsep_data/cut_maps'
    DATA_DIR = 'data'
    tsz_path = BASE + '/tsz.npy'
    tsz = np.load(tsz_path, mmap_mode='r')
    tsz_intensity = np.mean(np.abs(tsz), axis=(1, 2))
    n_patches = len(tsz_intensity)
    indices = np.argsort(tsz_intensity)
    train_idx = []
    val_idx = []
    test_idx = []
    for i in range(n_patches):
        if i % 10 < 8:
            train_idx.append(int(indices[i]))
        elif i % 10 == 8:
            val_idx.append(int(indices[i]))
        else:
            test_idx.append(int(indices[i]))
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    rng = np.random.default_rng(42)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    splits = {'train': train_idx.tolist(), 'val': val_idx.tolist(), 'test': test_idx.tolist()}
    with open(DATA_DIR + '/splits.json', 'w') as f:
        json.dump(splits, f)
    print('Splits saved to ' + DATA_DIR + '/splits.json')
    frequencies = [90, 150, 217, 353, 545, 857]
    signals = {}
    for freq in frequencies:
        signals[freq] = np.load(BASE + '/stacked_' + str(freq) + '.npy', mmap_mode='r')
    norm_stats = {}
    for freq in frequencies:
        train_signals = signals[freq][train_idx]
        if freq <= 217:
            noise_path = BASE + '/so_noise/' + str(freq) + '.npy'
            noise_mmap = np.load(noise_path, mmap_mode='r')
            noise_idx = rng.integers(0, 3000, size=len(train_idx))
            noise = noise_mmap[noise_idx]
        else:
            noise = np.zeros_like(train_signals)
            for i, p_idx in enumerate(train_idx):
                n_idx = rng.integers(0, 100)
                raw = np.load(BASE + '/planck_noise/planck_noise_' + str(freq) + '_' + str(n_idx) + '.npy', mmap_mode='r')[p_idx]
                if freq == 353:
                    noise[i] = raw * 1e6
                else:
                    noise[i] = raw * 1e6 * utils.jysr2uk(freq)
        train_data = train_signals + noise
        median = float(np.median(train_data))
        q75, q25 = np.percentile(train_data, [75, 25])
        iqr = float(q75 - q25)
        if iqr == 0:
            iqr = 1.0
        norm_stats[str(freq)] = {'median': median, 'iqr': iqr}
        print('Freq ' + str(freq) + ' GHz - Median: ' + str(median) + ', IQR: ' + str(iqr))
    with open(DATA_DIR + '/norm_stats.json', 'w') as f:
        json.dump(norm_stats, f)
    print('Normalization stats saved to ' + DATA_DIR + '/norm_stats.json')
    dataset = CompSepDataset(train_idx, split='train')
    x, noise_vars, y = dataset[0]
    print('Dataset test - x shape: ' + str(x.shape) + ', noise_vars shape: ' + str(noise_vars.shape) + ', y shape: ' + str(y.shape))