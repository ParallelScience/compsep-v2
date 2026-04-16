# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
BASE = '/home/node/data/compsep_data/cut_maps'
class FlamingoDataset(Dataset):
    def __init__(self, split='train', seed=42):
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.n_patch = 1523
        self.n_planck = 100
        self.n_so = 3000
        all_indices = np.arange(self.n_patch)
        self.rng.shuffle(all_indices)
        if split == 'train':
            self.indices = all_indices[:-300]
        elif split == 'val':
            self.indices = all_indices[-300:-150]
        elif split == 'test':
            self.indices = all_indices[-150:]
        else:
            self.indices = all_indices
        self.frequencies = [90, 150, 217, 353, 545, 857]
        self.so_freqs = [90, 150, 217]
        self.planck_freqs = [353, 545, 857]
        self.stacked_signals = {}
        for freq in self.frequencies:
            self.stacked_signals[freq] = np.load(BASE + '/stacked_' + str(freq) + '.npy', mmap_mode='r')
        self.so_noise = {}
        self.so_noise_var = {}
        for freq in self.so_freqs:
            self.so_noise[freq] = np.load(BASE + '/so_noise/' + str(freq) + '.npy', mmap_mode='r')
            self.so_noise_var[freq] = np.var(self.so_noise[freq], axis=0)
        self.planck_noise = {freq: [] for freq in self.planck_freqs}
        for freq in self.planck_freqs:
            for i in range(self.n_planck):
                self.planck_noise[freq].append(np.load(BASE + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i) + '.npy', mmap_mode='r'))
        self.tsz = np.load(BASE + '/tsz.npy', mmap_mode='r')
        self.ksz = np.load(BASE + '/ksz.npy', mmap_mode='r')
        self.cmb = np.load(BASE + '/lensed_cmb.npy', mmap_mode='r')
        self.cib = {}
        for freq in self.frequencies:
            self.cib[freq] = np.load(BASE + '/cib_' + str(freq) + '.npy', mmap_mode='r')
        sample_idx = all_indices[:50]
        tsz_sample = np.stack([self.tsz[i] for i in sample_idx])
        self.tsz_scale = np.std(tsz_sample)
        if self.tsz_scale == 0: self.tsz_scale = 1.0
        self.cib_obs_scales = {}
        for freq in self.planck_freqs:
            cib_obs_sample = np.stack([self.stacked_signals[freq][i] for i in sample_idx])
            self.cib_obs_scales[freq] = np.std(cib_obs_sample)
            if self.cib_obs_scales[freq] == 0: self.cib_obs_scales[freq] = 1.0
        self.cib_gt_scales = {}
        for freq in self.frequencies:
            cib_gt_sample = np.stack([self.cib[freq][i] for i in sample_idx])
            self.cib_gt_scales[freq] = np.std(cib_gt_sample)
            if self.cib_gt_scales[freq] == 0: self.cib_gt_scales[freq] = 1.0
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        i_patch = self.indices[idx]
        i_so = torch.randint(0, self.n_so, (1,)).item()
        i_planck = torch.randint(0, self.n_planck, (1,)).item()
        so_inputs = []
        so_vars = []
        for freq in self.so_freqs:
            sig = self.stacked_signals[freq][i_patch]
            noise = self.so_noise[freq][i_so]
            obs = sig + noise
            so_inputs.append(obs)
            so_vars.append(self.so_noise_var[freq])
        planck_inputs = []
        for freq in self.planck_freqs:
            sig = self.stacked_signals[freq][i_patch]
            raw_noise = self.planck_noise[freq][i_planck][i_patch]
            if freq == 353:
                noise = raw_noise * 1e6
            else:
                noise = raw_noise * 1e6 * utils.jysr2uk(freq)
            obs = sig + noise
            obs_trans = np.arcsinh(obs / self.cib_obs_scales[freq])
            planck_inputs.append(obs_trans)
        tsz_gt = self.tsz[i_patch]
        tsz_gt_trans = np.arcsinh(tsz_gt / self.tsz_scale)
        cib_gt_trans = []
        for freq in self.frequencies:
            cib_val = self.cib[freq][i_patch]
            cib_gt_trans.append(np.arcsinh(cib_val / self.cib_gt_scales[freq]))
        so_tensor = torch.tensor(np.stack(so_inputs), dtype=torch.float32)
        so_var_tensor = torch.tensor(np.stack(so_vars), dtype=torch.float32)
        cib_tensor = torch.tensor(np.stack(planck_inputs), dtype=torch.float32)
        tsz_tensor = torch.tensor(tsz_gt_trans, dtype=torch.float32).unsqueeze(0)
        cib_gt_tensor = torch.tensor(np.stack(cib_gt_trans), dtype=torch.float32)
        ksz_tensor = torch.tensor(self.ksz[i_patch], dtype=torch.float32).unsqueeze(0)
        cmb_tensor = torch.tensor(self.cmb[i_patch], dtype=torch.float32).unsqueeze(0)
        return {'so_inputs': so_tensor, 'so_vars': so_var_tensor, 'cib_inputs': cib_tensor, 'tsz_gt': tsz_tensor, 'cib_gt': cib_gt_tensor, 'ksz_gt': ksz_tensor, 'cmb_gt': cmb_tensor, 'patch_idx': i_patch}
if __name__ == '__main__':
    start_time = time.time()
    train_dataset = FlamingoDataset(split='train')
    val_dataset = FlamingoDataset(split='val')
    print('Datasets initialized in ' + str(time.time() - start_time) + ' seconds.')
    print('Train patches: ' + str(len(train_dataset)) + ', Val patches: ' + str(len(val_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    print('\nFetching one batch from DataLoader...')
    start_time = time.time()
    for batch in train_loader:
        so_inputs = batch['so_inputs']
        so_vars = batch['so_vars']
        cib_inputs = batch['cib_inputs']
        tsz_gt = batch['tsz_gt']
        cib_gt = batch['cib_gt']
        patch_idx = batch['patch_idx']
        break
    print('Batch fetched in ' + str(time.time() - start_time) + ' seconds.')
    print('\n--- Batch Statistics ---')
    print('SO Inputs Shape: ' + str(so_inputs.shape))
    print('SO Vars Shape: ' + str(so_vars.shape))
    print('CIB Inputs Shape: ' + str(cib_inputs.shape))
    print('tSZ GT Shape: ' + str(tsz_gt.shape))
    print('CIB GT Shape: ' + str(cib_gt.shape))
    print('\nSO Inputs (90, 150, 217 GHz) [uK_CMB]:')
    for i, freq in enumerate([90, 150, 217]):
        print('  ' + str(freq) + ' GHz - Mean: ' + str(so_inputs[:, i].mean().item()) + ', Std: ' + str(so_inputs[:, i].std().item()) + ', Min: ' + str(so_inputs[:, i].min().item()) + ', Max: ' + str(so_inputs[:, i].max().item()))
    print('\nSO Noise Variance Maps [uK_CMB^2]:')
    for i, freq in enumerate([90, 150, 217]):
        print('  ' + str(freq) + ' GHz - Mean: ' + str(so_vars[:, i].mean().item()) + ', Std: ' + str(so_vars[:, i].std().item()))
    print('\nCIB Inputs (353, 545, 857 GHz) [Arcsinh Transformed]:')
    for i, freq in enumerate([353, 545, 857]):
        print('  ' + str(freq) + ' GHz - Mean: ' + str(cib_inputs[:, i].mean().item()) + ', Std: ' + str(cib_inputs[:, i].std().item()) + ', Min: ' + str(cib_inputs[:, i].min().item()) + ', Max: ' + str(cib_inputs[:, i].max().item()))
    print('\ntSZ Ground Truth [Arcsinh Transformed]:')
    print('  Mean: ' + str(tsz_gt.mean().item()) + ', Std: ' + str(tsz_gt.std().item()) + ', Min: ' + str(tsz_gt.min().item()) + ', Max: ' + str(tsz_gt.max().item()))
    print('\ntSZ Scaling Factor used: ' + str(train_dataset.tsz_scale))
    print('CIB Observed Scaling Factors used:')
    for freq, scale in train_dataset.cib_obs_scales.items():
        print('  ' + str(freq) + ' GHz: ' + str(scale))
    print('\nData Preprocessing and Dataset Class Implementation completed successfully.')