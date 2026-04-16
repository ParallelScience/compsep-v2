# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = False
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
DATA_DIR = 'data'
BASE_DIR = '/home/node/data/compsep_data/cut_maps'
def generate_plots():
    print('Loading data subsets...')
    preds_std = np.load(os.path.join(DATA_DIR, 'preds_std_subset.npy'))
    preds_null = np.load(os.path.join(DATA_DIR, 'preds_null_subset.npy'))
    targets = np.load(os.path.join(DATA_DIR, 'targets_subset.npy'))
    stacked_150 = np.load(os.path.join(BASE_DIR, 'stacked_150.npy'), mmap_mode='r')[:50]
    so_noise_150 = np.load(os.path.join(BASE_DIR, 'so_noise', '150.npy'), mmap_mode='r')[0]
    raw_150_y = (stacked_150 + so_noise_150) / utils.tsz(150)
    print('Generating Map Comparison Figure...')
    patch_sums = np.sum(targets[:, 0, :, :], axis=(1, 2))
    top_indices = np.argsort(patch_sums)[-3:][::-1]
    global_vmin = min(targets[top_indices, 0].min(), preds_std[top_indices, 0].min())
    global_vmax = max(targets[top_indices, 0].max(), preds_std[top_indices, 0].max())
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, idx in enumerate(top_indices):
        gt = targets[idx, 0]
        raw = raw_150_y[idx]
        pred = preds_std[idx, 0]
        im0 = axes[i, 0].imshow(gt, vmin=global_vmin, vmax=global_vmax, cmap='viridis', origin='lower')
        axes[i, 0].set_title('Ground Truth (Patch ' + str(idx) + ')')
        axes[i, 0].axis('off')
        im1 = axes[i, 1].imshow(raw, vmin=global_vmin, vmax=global_vmax, cmap='viridis', origin='lower')
        axes[i, 1].set_title('Raw SO-LAT 150GHz (Patch ' + str(idx) + ')')
        axes[i, 1].axis('off')
        im2 = axes[i, 2].imshow(pred, vmin=global_vmin, vmax=global_vmax, cmap='viridis', origin='lower')
        axes[i, 2].set_title('SR-DAE Recon (Patch ' + str(idx) + ')')
        axes[i, 2].axis('off')
    fig.subplots_adjust(right=0.9, wspace=0.1, hspace=0.2)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Compton-y (dimensionless)')
    timestamp = str(int(time.time()))
    fig1_path = os.path.join(DATA_DIR, 'map_comparison_1_' + timestamp + '.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print('Map comparison plot saved to ' + fig1_path)
    print('Computing Power Spectra and Cross-correlations...')
    cl_gt_gt = []
    cl_pred_pred = []
    cl_pred_gt = []
    cl_null_null = []
    cl_null_gt = []
    ell_ps = None
    for i in range(50):
        gt = targets[i, 0]
        pred = preds_std[i, 0]
        null = preds_null[i, 0]
        try:
            res_gt_gt = utils.powers(gt, gt, ps=5, ell_n=199, window_alpha=None)
            res_pred_pred = utils.powers(pred, pred, ps=5, ell_n=199, window_alpha=None)
            res_pred_gt = utils.powers(pred, gt, ps=5, ell_n=199, window_alpha=None)
            res_null_null = utils.powers(null, null, ps=5, ell_n=199, window_alpha=None)
            res_null_gt = utils.powers(null, gt, ps=5, ell_n=199, window_alpha=None)
            if np.mean(res_gt_gt[0]) < np.mean(res_gt_gt[1]):
                cl_idx, ell_idx = 0, 1
            else:
                cl_idx, ell_idx = 1, 0
            cl_gt_gt.append(res_gt_gt[cl_idx])
            cl_pred_pred.append(res_pred_pred[cl_idx])
            cl_pred_gt.append(res_pred_gt[cl_idx])
            cl_null_null.append(res_null_null[cl_idx])
            cl_null_gt.append(res_null_gt[cl_idx])
            if ell_ps is None:
                ell_ps = res_gt_gt[ell_idx]
        except Exception:
            pass
    cl_gt_gt_mean = np.mean(cl_gt_gt, axis=0)
    cl_pred_pred_mean = np.mean(cl_pred_pred, axis=0)
    cl_pred_gt_mean = np.mean(cl_pred_gt, axis=0)
    cl_null_null_mean = np.mean(cl_null_null, axis=0)
    cl_null_gt_mean = np.mean(cl_null_gt, axis=0)
    denom_std = np.sqrt(np.maximum(cl_pred_pred_mean * cl_gt_gt_mean, 1e-30))
    r_ell_std = cl_pred_gt_mean / denom_std
    denom_null = np.sqrt(np.maximum(cl_null_null_mean * cl_gt_gt_mean, 1e-30))
    r_ell_null = cl_null_gt_mean / denom_null
    print('Computing Y_SZ - Mass Relation with Cluster Masking...')
    mass_proxy_50 = []
    y_sz_pred_50 = []
    y_sz_raw_50 = []
    y_sz_null_50 = []
    for i in range(50):
        mask = targets[i, 0] > 5e-7
        if np.sum(mask) == 0:
            mask = targets[i, 0] > np.max(targets[i, 0]) * 0.1
        mass_proxy_50.append(np.sum(targets[i, 0][mask]))
        y_sz_pred_50.append(np.sum(preds_std[i, 0][mask]))
        y_sz_raw_50.append(np.sum(raw_150_y[i][mask]))
        y_sz_null_50.append(np.sum(preds_null[i, 0][mask]))
    mass_proxy_50 = np.array(mass_proxy_50)
    y_sz_pred_50 = np.array(y_sz_pred_50)
    y_sz_raw_50 = np.array(y_sz_raw_50)
    y_sz_null_50 = np.array(y_sz_null_50)
    num_bins = 5
    bins = np.percentile(mass_proxy_50, np.linspace(0, 100, num_bins + 1))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    scatter_std = []
    scatter_null = []
    scatter_raw = []
    for i in range(num_bins):
        mask = (mass_proxy_50 >= bins[i]) & (mass_proxy_50 <= bins[i+1])
        if np.sum(mask) == 0:
            scatter_std.append(0)
            scatter_null.append(0)
            scatter_raw.append(0)
            continue
        scatter_std.append(np.std(y_sz_pred_50[mask] - mass_proxy_50[mask]))
        scatter_null.append(np.std(y_sz_null_50[mask] - mass_proxy_50[mask]))
        scatter_raw.append(np.std(y_sz_raw_50[mask] - mass_proxy_50[mask]))
    print('Generating Performance Assessment Figure...')
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes2[0, 0]
    valid_ps = ell_ps > 0
    ax.plot(ell_ps[valid_ps], cl_gt_gt_mean[valid_ps], label='Ground Truth', color='black', linewidth=2)
    ax.plot(ell_ps[valid_ps], cl_pred_pred_mean[valid_ps], label='SR-DAE Recon', color='blue', linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Multipole l')
    ax.set_ylabel('Power Spectrum C_l (dimensionless)')
    ax.set_title('Radial Power Spectra')
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax = axes2[0, 1]
    ax.plot(ell_ps[valid_ps], r_ell_std[valid_ps], label='Standard Recon', color='blue')
    ax.plot(ell_ps[valid_ps], r_ell_null[valid_ps], label='Null Test (CIB Shuffled)', color='red', linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('Multipole l')
    ax.set_ylabel('Cross-correlation r_l (dimensionless)')
    ax.set_title('Cross-correlation Coefficient')
    ax.set_ylim(-0.2, 1.1)
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax = axes2[1, 0]
    ax.scatter(mass_proxy_50, y_sz_raw_50, alpha=0.5, label='Raw SO-LAT', color='gray', s=20)
    ax.scatter(mass_proxy_50, y_sz_pred_50, alpha=0.8, label='SR-DAE Recon', color='blue', s=20)
    min_line = mass_proxy_50.min()
    max_line = mass_proxy_50.max()
    ax.plot([min_line, max_line], [min_line, max_line], 'k--', label='Ideal (y=x)')
    ax.set_xlabel('Mass Proxy (Integrated GT tSZ, dimensionless)')
    ax.set_ylabel('Reconstructed Integrated Y_SZ (dimensionless)')
    ax.set_title('Y_SZ - Mass Relation')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax = axes2[1, 1]
    ax.plot(bin_centers, scatter_raw, marker='o', label='Raw SO-LAT', color='gray')
    ax.plot(bin_centers, scatter_null, marker='s', label='Null Test', color='red', linestyle='--')
    ax.plot(bin_centers, scatter_std, marker='^', label='SR-DAE Recon', color='blue')
    ax.set_yscale('log')
    ax.set_xlabel('Mass Proxy (Integrated GT tSZ, dimensionless)')
    ax.set_ylabel('Residual Scatter (Std Dev, dimensionless)')
    ax.set_title('Residual Scatter vs Mass Proxy')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig2_path = os.path.join(DATA_DIR, 'performance_assessment_2_' + timestamp + '.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print('Performance assessment plot saved to ' + fig2_path)
    print('\n--- Summary Statistics for Researcher ---')
    print('Mean Power Spectrum (Ground Truth) at low l: ' + str(cl_gt_gt_mean[1]))
    print('Mean Power Spectrum (Recon) at low l: ' + str(cl_pred_pred_mean[1]))
    print('Mean Power Spectrum (Ground Truth) at high l: ' + str(cl_gt_gt_mean[-1]))
    print('Mean Power Spectrum (Recon) at high l: ' + str(cl_pred_pred_mean[-1]))
    print('\nCross-correlation r_l (Standard vs Null):')
    step = max(1, len(ell_ps)//5)
    for i in range(0, len(ell_ps), step):
        print('  l=' + str(round(ell_ps[i], 1)) + ': r_l_std=' + str(round(r_ell_std[i], 4)) + ', r_l_null=' + str(round(r_ell_null[i], 4)))
    print('\nResidual Scatter vs Mass Proxy (Standard vs Null vs Raw):')
    for i in range(len(bin_centers)):
        print('  Mass Bin ' + str(i+1) + ' (center=' + str(round(bin_centers[i], 6)) + '): Scatter STD=' + str(round(scatter_std[i], 6)) + ', NULL=' + str(round(scatter_null[i], 6)) + ', RAW=' + str(round(scatter_raw[i], 6)))
if __name__ == '__main__':
    generate_plots()