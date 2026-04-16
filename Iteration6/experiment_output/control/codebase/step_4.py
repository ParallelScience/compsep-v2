# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data/')
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
import utils
from step_1 import FLAMINGODataset
from step_2 import SRDAE

plt.rcParams['text.usetex'] = False

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    print('Loading normalization statistics...')
    stats = np.load('data/normalization_stats.npz')
    y_mean = stats['y_mean'][0]
    y_std = stats['y_std'][0]
    print('Loading test dataset...')
    test_dataset = FLAMINGODataset('test', splits_file='data/splits.npz', stats_file='data/normalization_stats.npz', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print('Loading SR-DAE model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRDAE().to(device)
    model.load_state_dict(torch.load('data/best_srdae.pth', map_location=device))
    model.eval()
    srdae_preds = []
    srdae_trues = []
    print('Performing inference with SR-DAE...')
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_pred = model(x).cpu().numpy()
            y = y.numpy()
            y_pred = y_pred * y_std + y_mean
            y = y * y_std + y_mean
            srdae_preds.append(y_pred)
            srdae_trues.append(y)
    srdae_preds = np.concatenate(srdae_preds, axis=0).squeeze(1)
    srdae_trues = np.concatenate(srdae_trues, axis=0).squeeze(1)
    print('Loading cILC results...')
    cilc_data = np.load('data/cilc_results.npz')
    cilc_preds_raw = cilc_data['y_pred']
    cilc_trues = cilc_data['y_true']
    scale_factor = np.sum(cilc_preds_raw * cilc_trues) / np.sum(cilc_preds_raw**2)
    print('Applying scaling factor to cILC predictions: ' + str(scale_factor))
    cilc_preds = cilc_preds_raw * scale_factor
    print('Loading CIB maps for test set...')
    BASE = '/home/node/data/compsep_data/cut_maps'
    splits = np.load('data/splits.npz')
    test_indices = splits['test']
    cib_freqs = [353, 545, 857]
    cib_maps = {}
    for freq in cib_freqs:
        cib_maps[freq] = np.load(BASE + '/cib_' + str(freq) + '.npy', mmap_mode='r')[test_indices]
    def compute_cross_corr(R, C):
        R_mean = R.mean(axis=(1, 2), keepdims=True)
        C_mean = C.mean(axis=(1, 2), keepdims=True)
        R_centered = R - R_mean
        C_centered = C - C_mean
        cov = np.sum(R_centered * C_centered, axis=(1, 2))
        var_R = np.sum(R_centered**2, axis=(1, 2))
        var_C = np.sum(C_centered**2, axis=(1, 2))
        corr = cov / np.sqrt(var_R * var_C + 1e-12)
        return corr.mean(), corr.std()
    print('Computing cross-correlations...')
    corr_srdae = {}
    corr_cilc = {}
    for freq in cib_freqs:
        C = cib_maps[freq]
        corr_srdae[freq] = compute_cross_corr(srdae_trues - srdae_preds, C)
        corr_cilc[freq] = compute_cross_corr(cilc_trues - cilc_preds, C)
    print('Computing power spectra and transfer functions...')
    ell_bins = None
    P_auto_true = []
    P_cross_srdae = []
    P_auto_srdae = []
    P_cross_cilc = []
    P_auto_cilc = []
    pixel_size_arcmin = 1.171875
    for i in range(len(srdae_trues)):
        true_map = srdae_trues[i]
        srdae_map = srdae_preds[i]
        cilc_map = cilc_preds[i]
        res = utils.powers(true_map, true_map, ps=pixel_size_arcmin)
        if np.mean(np.abs(res[0])) < np.mean(np.abs(res[1])):
            power_idx, ell_idx = 0, 1
        else:
            power_idx, ell_idx = 1, 0
        p_auto_t = res[power_idx]
        ell = res[ell_idx]
        p_cross_s = utils.powers(srdae_map, true_map, ps=pixel_size_arcmin)[power_idx]
        p_auto_s = utils.powers(srdae_map, srdae_map, ps=pixel_size_arcmin)[power_idx]
        p_cross_c = utils.powers(cilc_map, true_map, ps=pixel_size_arcmin)[power_idx]
        p_auto_c = utils.powers(cilc_map, cilc_map, ps=pixel_size_arcmin)[power_idx]
        if ell_bins is None:
            ell_bins = ell
        P_auto_true.append(p_auto_t)
        P_cross_srdae.append(p_cross_s)
        P_auto_srdae.append(p_auto_s)
        P_cross_cilc.append(p_cross_c)
        P_auto_cilc.append(p_auto_c)
    P_auto_true = np.array(P_auto_true)
    P_cross_srdae = np.array(P_cross_srdae)
    P_auto_srdae = np.array(P_auto_srdae)
    P_cross_cilc = np.array(P_cross_cilc)
    P_auto_cilc = np.array(P_auto_cilc)
    threshold = 1e-20
    P_auto_true_safe = np.where(P_auto_true < threshold, threshold, P_auto_true)
    T_srdae_patches = P_cross_srdae / P_auto_true_safe
    T_cilc_patches = P_cross_cilc / P_auto_true_safe
    std_T_srdae = np.std(T_srdae_patches, axis=0)
    std_T_cilc = np.std(T_cilc_patches, axis=0)
    print('Generating plots...')
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(ell_bins, np.mean(T_srdae_patches, axis=0), label='SR-DAE', color='blue')
    axs[0, 0].fill_between(ell_bins, np.mean(T_srdae_patches, axis=0) - std_T_srdae, np.mean(T_srdae_patches, axis=0) + std_T_srdae, color='blue', alpha=0.3)
    axs[0, 0].plot(ell_bins, np.mean(T_cilc_patches, axis=0), label='cILC', color='orange')
    axs[0, 0].fill_between(ell_bins, np.mean(T_cilc_patches, axis=0) - std_T_cilc, np.mean(T_cilc_patches, axis=0) + std_T_cilc, color='orange', alpha=0.3)
    axs[0, 0].axhline(1.0, color='k', linestyle='--')
    axs[0, 0].set_title('tSZ Reconstruction Transfer Function')
    axs[0, 0].set_xlabel('Multipole l')
    axs[0, 0].set_ylabel('T(l)')
    axs[0, 0].set_ylim(0, 2)
    axs[0, 0].legend()
    axs[0, 1].loglog(ell_bins, np.mean(P_auto_true, axis=0), label='True', color='black')
    axs[0, 1].loglog(ell_bins, np.mean(P_auto_srdae, axis=0), label='SR-DAE', color='blue')
    axs[0, 1].loglog(ell_bins, np.mean(P_auto_cilc, axis=0), label='cILC', color='orange')
    axs[0, 1].set_title('tSZ Power Spectrum')
    axs[0, 1].set_xlabel('Multipole l')
    axs[0, 1].set_ylabel('Cl')
    axs[0, 1].legend()
    im = axs[1, 0].imshow(srdae_trues[0] - srdae_preds[0], cmap='RdBu_r')
    axs[1, 0].set_title('SR-DAE Residual Map (Patch 0)')
    cbar = fig.colorbar(im, ax=axs[1, 0])
    cbar.set_label('Residual Compton-y')
    axs[1, 1].bar(np.arange(len(cib_freqs)) - 0.2, [corr_srdae[f][0] for f in cib_freqs], width=0.4, label='SR-DAE', color='blue')
    axs[1, 1].bar(np.arange(len(cib_freqs)) + 0.2, [corr_cilc[f][0] for f in cib_freqs], width=0.4, label='cILC', color='orange')
    axs[1, 1].set_xticks(np.arange(len(cib_freqs)))
    axs[1, 1].set_xticklabels([str(f) + ' GHz' for f in cib_freqs])
    axs[1, 1].set_title('Residual-CIB Cross-Correlation')
    axs[1, 1].set_ylabel('Pearson Correlation Coefficient')
    axs[1, 1].legend()
    plt.tight_layout()
    plt.savefig('data/evaluation_summary.png')
    print('Saved to data/evaluation_summary.png')
    print('\n--- Quantitative Metrics ---')
    print('SR-DAE MSE: ' + str(np.mean((srdae_trues - srdae_preds)**2)))
    print('cILC MSE: ' + str(np.mean((cilc_trues - cilc_preds)**2)))
    print('SR-DAE Bias: ' + str(np.mean(srdae_preds - srdae_trues)))
    print('cILC Bias: ' + str(np.mean(cilc_preds - cilc_trues)))
    print('\nResidual-CIB Cross-Correlation (Mean ± Std):')
    for freq in cib_freqs:
        print('  ' + str(freq) + ' GHz - SR-DAE: ' + str(corr_srdae[freq][0]) + ' ± ' + str(corr_srdae[freq][1]))
        print('  ' + str(freq) + ' GHz - cILC:   ' + str(corr_cilc[freq][0]) + ' ± ' + str(corr_cilc[freq][1]))
    print('\nTransfer Function T(l) at specific multipoles (SR-DAE):')
    target_ells = [1000, 3000, 5000]
    for target_ell in target_ells:
        idx = np.argmin(np.abs(ell_bins - target_ell))
        print('  l ~ ' + str(int(ell_bins[idx])) + ': ' + str(np.mean(T_srdae_patches, axis=0)[idx]) + ' ± ' + str(std_T_srdae[idx]))