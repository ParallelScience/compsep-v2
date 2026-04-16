# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
from torch.utils.data import Dataset
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
class CompSepDataset(Dataset):
    def __init__(self, patch_indices, base_dir, scaling_params=None, tsz_clip_threshold=None, split='train'):
        self.patch_indices = patch_indices
        self.base_dir = base_dir
        self.split = split
        self.scaling_params = scaling_params
        self.tsz_clip_threshold = tsz_clip_threshold
        self.frequencies = [90, 150, 217, 353, 545, 857]
        self.stacked_signals = None
        self.so_noise = None
        self.tsz = None
        self.planck_noise_mmaps = None
    def _init_mmaps(self):
        if self.stacked_signals is not None:
            return
        self.stacked_signals = {freq: np.load(os.path.join(self.base_dir, 'stacked_' + str(freq) + '.npy'), mmap_mode='r') for freq in self.frequencies}
        self.so_noise = {freq: np.load(os.path.join(self.base_dir, 'so_noise', str(freq) + '.npy'), mmap_mode='r') for freq in [90, 150, 217]}
        self.tsz = np.load(os.path.join(self.base_dir, 'tsz.npy'), mmap_mode='r')
        self.planck_noise_mmaps = {}
        for freq in [353, 545, 857]:
            for i in range(100):
                filepath = os.path.join(self.base_dir, 'planck_noise', 'planck_noise_' + str(freq) + '_' + str(i) + '.npy')
                self.planck_noise_mmaps[(freq, i)] = np.load(filepath, mmap_mode='r')
    def __len__(self):
        return len(self.patch_indices)
    def __getitem__(self, idx):
        self._init_mmaps()
        i_patch = self.patch_indices[idx]
        if self.split == 'train':
            i_planck = np.random.randint(0, 100)
            i_so = np.random.randint(0, 3000)
        else:
            rng = np.random.default_rng(seed=i_patch)
            i_planck = rng.integers(0, 100)
            i_so = rng.integers(0, 3000)
        inputs = []
        for freq in self.frequencies:
            signal = self.stacked_signals[freq][i_patch]
            if freq <= 217:
                noise = self.so_noise[freq][i_so]
            else:
                raw_noise = self.planck_noise_mmaps[(freq, i_planck)][i_patch]
                if freq == 353:
                    noise = raw_noise * 1e6
                else:
                    noise = raw_noise * 1e6 * utils.jysr2uk(freq)
            channel_data = signal + noise
            if self.scaling_params is not None:
                params = self.scaling_params
                if isinstance(params, np.ndarray):
                    params = params.item()
                median = params[freq]['median']
                iqr = params[freq]['iqr']
                channel_data = (channel_data - median) / (iqr + 1e-8)
            inputs.append(channel_data)
        inputs = np.stack(inputs, axis=0).astype(np.float32)
        tsz_data = self.tsz[i_patch].astype(np.float32)
        if self.tsz_clip_threshold is not None:
            tsz_data = self.tsz_clip_threshold * np.tanh(tsz_data / self.tsz_clip_threshold)
        return torch.from_numpy(inputs), torch.from_numpy(tsz_data)
if __name__ == '__main__':
    print('Starting Data Pipeline and Preprocessing...')
    base_dir = '/home/node/data/compsep_data/cut_maps'
    tsz_maps = np.load(os.path.join(base_dir, 'tsz.npy'), mmap_mode='r')
    n_patch = len(tsz_maps)
    peak_amplitudes = np.array([np.max(tsz_maps[i]) for i in range(n_patch)])
    sorted_indices = np.argsort(peak_amplitudes)
    val_indices = []
    train_indices = []
    n_val = 150
    bin_size = n_patch / float(n_val)
    for i in range(n_val):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size) if i < n_val - 1 else n_patch
        bin_indices = sorted_indices[start_idx:end_idx]
        mid = len(bin_indices) // 2
        val_indices.append(bin_indices[mid])
        train_indices.extend(np.delete(bin_indices, mid))
    val_indices = np.array(val_indices)
    train_indices = np.array(train_indices)
    print('Total patches: ' + str(n_patch))
    print('Training patches: ' + str(len(train_indices)))
    print('Validation patches: ' + str(len(val_indices)))
    print('\nComputing robust scaling parameters (median and IQR) on training set...')
    np.random.seed(42)
    frequencies = [90, 150, 217, 353, 545, 857]
    scaling_params = {}
    for freq in frequencies:
        print('  Processing ' + str(freq) + ' GHz channel...')
        channel_data = np.zeros((len(train_indices), 256, 256), dtype=np.float32)
        stacked = np.load(os.path.join(base_dir, 'stacked_' + str(freq) + '.npy'), mmap_mode='r')
        if freq <= 217:
            so_noise = np.load(os.path.join(base_dir, 'so_noise', str(freq) + '.npy'), mmap_mode='r')
        for idx, i_patch in enumerate(train_indices):
            signal = stacked[i_patch]
            if freq <= 217:
                i_so = np.random.randint(0, 3000)
                noise = so_noise[i_so]
            else:
                i_planck = np.random.randint(0, 100)
                raw_noise = np.load(os.path.join(base_dir, 'planck_noise', 'planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy'), mmap_mode='r')[i_patch]
                if freq == 353:
                    noise = raw_noise * 1e6
                else:
                    noise = raw_noise * 1e6 * utils.jysr2uk(freq)
            channel_data[idx] = signal + noise
        median = np.median(channel_data)
        q75, q25 = np.percentile(channel_data, [75, 25])
        iqr = q75 - q25
        scaling_params[freq] = {'median': float(median), 'iqr': float(iqr)}
        print('    Median: ' + ('%.4e' % median) + ', IQR: ' + ('%.4e' % iqr))
    print('\nComputing tSZ soft-clipping threshold from training set 99.9th percentile...')
    tsz_train = np.zeros((len(train_indices), 256, 256), dtype=np.float32)
    for idx, i_patch in enumerate(train_indices):
        tsz_train[idx] = tsz_maps[i_patch]
    tsz_clip_threshold = float(np.percentile(tsz_train, 99.9))
    print('  tSZ 99.9th percentile threshold: ' + ('%.4e' % tsz_clip_threshold))
    save_path = os.path.join('data', 'scaling_params.npz')
    np.savez(save_path, scaling_params=scaling_params, tsz_clip_threshold=tsz_clip_threshold, train_indices=train_indices, val_indices=val_indices)
    print('\nScaling parameters and indices saved to ' + save_path)
    print('\nVerifying scaling on a subset of training data...')
    dataset = CompSepDataset(train_indices, base_dir, scaling_params, tsz_clip_threshold, split='train')
    subset_indices = np.random.choice(len(train_indices), 100, replace=False)
    scaled_inputs = []
    clipped_tsz = []
    for idx in subset_indices:
        x, y = dataset[idx]
        scaled_inputs.append(x.numpy())
        clipped_tsz.append(y.numpy())
    scaled_inputs = np.stack(scaled_inputs)
    clipped_tsz = np.stack(clipped_tsz)
    print('-' * 90)
    header = 'Channel'.ljust(10) + ' | ' + 'Min'.ljust(15) + ' | ' + 'Max'.ljust(15) + ' | ' + 'Mean'.ljust(15) + ' | ' + 'Std'.ljust(15)
    print(header)
    print('-' * 90)
    for i, freq in enumerate(frequencies):
        data = scaled_inputs[:, i, :, :]
        line = (str(freq) + ' GHz').ljust(10) + ' | ' + ('%.4e' % np.min(data)).ljust(15) + ' | ' + ('%.4e' % np.max(data)).ljust(15) + ' | ' + ('%.4e' % np.mean(data)).ljust(15) + ' | ' + ('%.4e' % np.std(data)).ljust(15)
        print(line)
    print('-' * 90)
    line = 'tSZ (clip)'.ljust(10) + ' | ' + ('%.4e' % np.min(clipped_tsz)).ljust(15) + ' | ' + ('%.4e' % np.max(clipped_tsz)).ljust(15) + ' | ' + ('%.4e' % np.mean(clipped_tsz)).ljust(15) + ' | ' + ('%.4e' % np.std(clipped_tsz)).ljust(15)
    print(line)
    print('-' * 90)
    print('\nData Pipeline and Preprocessing completed successfully.')