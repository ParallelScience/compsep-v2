# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['OMP_NUM_THREADS'] = '16'
import numpy as np
import torch
from torch.utils.data import Dataset
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
class TSZDataset(Dataset):
    def __init__(self, X, Y, M, augment=True):
        if isinstance(X, str):
            self.X = np.load(X)
            self.Y = np.load(Y)
            self.M = np.load(M)
        else:
            self.X = X
            self.Y = Y
            self.M = M
        self.augment = augment
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        m = self.M[idx]
        if self.augment:
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=-1)
                y = np.flip(y, axis=-1)
                m = np.flip(m, axis=-1)
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=-2)
                y = np.flip(y, axis=-2)
                m = np.flip(m, axis=-2)
            k = np.random.randint(0, 4)
            if k > 0:
                x = np.rot90(x, k, axes=(-2, -1))
                y = np.rot90(y, k, axes=(-2, -1))
                m = np.rot90(m, k, axes=(-2, -1))
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), torch.from_numpy(m.copy())
def prepare_data():
    BASE = '/home/node/data/compsep_data/cut_maps'
    data_dir = 'data/'
    n_patch = 1523
    rng = np.random.default_rng(seed=42)
    so_indices = rng.integers(0, 3000, size=n_patch)
    planck_indices = rng.integers(0, 100, size=n_patch)
    stacked = {f: np.load(BASE + '/stacked_' + str(f) + '.npy') for f in [90, 150, 217, 353, 545, 857]}
    so_noise = {f: np.load(BASE + '/so_noise/' + str(f) + '.npy') for f in [90, 150, 217]}
    maps = {}
    for f in [90, 150, 217]:
        maps[f] = stacked[f] + so_noise[f][so_indices]
    for f in [353, 545, 857]:
        maps[f] = np.zeros((n_patch, 256, 256), dtype=np.float32)
    for i in range(100):
        patches_in_i = np.where(planck_indices == i)[0]
        if len(patches_in_i) > 0:
            noise_353 = np.load(BASE + '/planck_noise/planck_noise_353_' + str(i) + '.npy', mmap_mode='r')
            noise_545 = np.load(BASE + '/planck_noise/planck_noise_545_' + str(i) + '.npy', mmap_mode='r')
            noise_857 = np.load(BASE + '/planck_noise/planck_noise_857_' + str(i) + '.npy', mmap_mode='r')
            for p in patches_in_i:
                maps[353][p] = stacked[353][p] + noise_353[p] * 1e6
                maps[545][p] = stacked[545][p] + noise_545[p] * 1e6 * utils.jysr2uk(545)
                maps[857][p] = stacked[857][p] + noise_857[p] * 1e6 * utils.jysr2uk(857)
    diff_150_90 = maps[150] - maps[90]
    diff_217_150 = maps[217] - maps[150]
    noise_diff_150_90 = so_noise[150] - so_noise[90]
    std_noise_150_90 = np.std(noise_diff_150_90)
    diff_150_90 /= std_noise_150_90
    noise_diff_217_150 = so_noise[217] - so_noise[150]
    std_noise_217_150 = np.std(noise_diff_217_150)
    diff_217_150 /= std_noise_217_150
    for f in [353, 545, 857]:
        median = np.median(maps[f])
        q75, q25 = np.percentile(maps[f], [75, 25])
        iqr = q75 - q25
        maps[f] = (maps[f] - median) / iqr
    tsz = np.load(BASE + '/tsz.npy')
    tsz_mean = np.mean(tsz)
    tsz_std = np.std(tsz)
    tsz_norm = (tsz - tsz_mean) / tsz_std
    mask = (tsz > 1e-7).astype(np.float32)
    X = np.stack([diff_150_90, diff_217_150, maps[353], maps[545], maps[857]], axis=1).astype(np.float32)
    Y = tsz_norm[:, np.newaxis, :, :].astype(np.float32)
    M = mask[:, np.newaxis, :, :].astype(np.float32)
    np.save(os.path.join(data_dir, 'X_features.npy'), X)
    np.save(os.path.join(data_dir, 'Y_target.npy'), Y)
    np.save(os.path.join(data_dir, 'M_mask.npy'), M)
    stats = {'tsz_mean': tsz_mean, 'tsz_std': tsz_std, 'std_noise_150_90': std_noise_150_90, 'std_noise_217_150': std_noise_217_150}
    np.save(os.path.join(data_dir, 'norm_stats.npy'), stats)
if __name__ == '__main__':
    prepare_data()