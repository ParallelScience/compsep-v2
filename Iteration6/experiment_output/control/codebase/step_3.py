# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
import numpy as np
import torch
from step_1 import FLAMINGODataset

def get_ell_2d(npix=256, pixel_size_arcmin=1.171875):
    dx = pixel_size_arcmin * np.pi / (180 * 60)
    kx = np.fft.fftfreq(npix, d=dx)
    ky = np.fft.fftfreq(npix, d=dx)
    k = np.sqrt(kx[None, :]**2 + ky[:, None]**2)
    ell = 2 * np.pi * k
    return ell

def apply_hilc(x, beam_ratio, C, e, ell, bin_edges, patch_idx):
    fft_x = np.fft.fft2(x)
    fft_x_deconv = fft_x * beam_ratio
    y_fft = np.zeros((256, 256), dtype=np.complex128)
    for b in range(len(bin_edges) - 1):
        mask = (ell >= bin_edges[b]) & (ell < bin_edges[b+1])
        if not np.any(mask):
            continue
        X_bin = fft_x_deconv[:, mask]
        N = X_bin.shape[1]
        R = np.real(X_bin @ X_bin.conj().T) / N
        R += np.eye(6) * 1e-6 * np.trace(R)
        invR = np.linalg.inv(R)
        left = invR @ C
        mid = np.linalg.inv(C.T @ invR @ C)
        W = left @ mid @ e
        if patch_idx == 0 and b == 0:
            print("cILC weights for ell " + str(bin_edges[b]) + "-" + str(bin_edges[b+1]) + ": " + str(W))
        y_fft[mask] = W.T @ X_bin
    y_pred = np.real(np.fft.ifft2(y_fft))
    return y_pred

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    print("Computing CIB spectral signature from training set...")
    BASE = '/home/node/data/compsep_data/cut_maps'
    frequencies = [90, 150, 217, 353, 545, 857]
    splits = np.load('data/splits.npz')
    train_indices = splits['train'][:20]
    a_cib = np.zeros(6)
    for i, freq in enumerate(frequencies):
        cib_map = np.load(os.path.join(BASE, 'cib_' + str(freq) + '.npy'), mmap_mode='r')[train_indices]
        cib_uk = cib_map * utils.jysr2uk(freq)
        a_cib[i] = np.mean(np.std(cib_uk, axis=(1, 2)))
    a_cib /= a_cib[5]
    a_tsz = np.array([utils.tsz(freq) for freq in frequencies])
    C = np.column_stack([a_tsz, a_cib])
    e = np.array([1, 0])
    fwhms = np.array([2.2, 1.4, 1.0, 4.5, 4.72, 4.42])
    target_fwhm = 1.0
    sigma_in = fwhms * np.pi / (180 * 60) / np.sqrt(8 * np.log(2))
    sigma_out = target_fwhm * np.pi / (180 * 60) / np.sqrt(8 * np.log(2))
    sigma2_diff = sigma_in**2 - sigma_out**2
    ell = get_ell_2d(256)
    beam_ratio = np.exp(ell[None, :, :]**2 * sigma2_diff[:, None, None] / 2)
    beam_ratio = np.clip(beam_ratio, 0, 1e4)
    bin_edges = [0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000]
    test_dataset = FLAMINGODataset('test', splits_file='data/splits.npz', stats_file=None, augment=False)
    torch.manual_seed(42)
    np.random.seed(42)
    mse_list = []
    bias_list = []
    y_preds = []
    y_trues = []
    for i in range(len(test_dataset)):
        x, y = test_dataset[i]
        x = x.numpy()
        y = y.numpy()[0]
        y_pred = apply_hilc(x, beam_ratio, C, e, ell, bin_edges, patch_idx=i)
        mse = np.mean((y_pred - y)**2)
        bias = np.mean(y_pred - y)
        mse_list.append(mse)
        bias_list.append(bias)
        y_preds.append(y_pred)
        y_trues.append(y)
        if (i + 1) % 20 == 0:
            print("Processed " + str(i + 1) + "/" + str(len(test_dataset)) + " patches...")
    mean_mse = np.mean(mse_list)
    mean_bias = np.mean(bias_list)
    std_mse = np.std(mse_list)
    std_bias = np.std(bias_list)
    print("\n--- cILC Baseline Results ---")
    print("Mean Squared Error (MSE): " + str(mean_mse) + " ± " + str(std_mse))
    print("Bias (Mean Error):        " + str(mean_bias) + " ± " + str(std_bias))
    output_file = 'data/cilc_results.npz'
    np.savez(output_file, y_pred=np.array(y_preds), y_true=np.array(y_trues), mse=mean_mse, bias=mean_bias)
    print("Saved cILC predictions and metrics to " + output_file)