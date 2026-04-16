# filename: codebase/step_8.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ['OMP_NUM_THREADS'] = '16'

sys.path.insert(0, os.path.abspath('codebase'))
sys.path.insert(0, '/home/node/data/compsep_data/')
import utils
from step_4 import SR_DAE

plt.rcParams['text.usetex'] = False

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading data and models...')
    test_features = np.load('data/test_features.npy')
    test_targets = np.load('data/test_targets.npy')
    stats = np.load('data/tsz_norm_stats.npz')
    train_mean = stats['mean']
    train_std = stats['std']
    true_unnorm = test_targets * train_std + train_mean
    full_model = SR_DAE(in_channels=6, out_channels=1, init_features=32).to(device)
    full_model.load_state_dict(torch.load('data/finetuned_model.pth', map_location=device))
    full_model.eval()
    ablated_model = SR_DAE(in_channels=3, out_channels=1, init_features=32).to(device)
    ablated_model.load_state_dict(torch.load('data/finetuned_ablated_model.pth', map_location=device))
    ablated_model.eval()
    print('Computing predictions on test set (Full Model)...')
    preds_full = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(test_features), batch_size):
            batch = torch.from_numpy(test_features[i:i+batch_size]).float().to(device)
            out = full_model(batch).cpu().numpy()[:, 0]
            preds_full.append(out)
    preds_full = np.concatenate(preds_full, axis=0)
    preds_full_unnorm = preds_full * train_std + train_mean
    print('Computing predictions on test set (Ablated Model)...')
    preds_ablated = []
    with torch.no_grad():
        for i in range(0, len(test_features), batch_size):
            batch = torch.from_numpy(test_features[i:i+batch_size, :3]).float().to(device)
            out = ablated_model(batch).cpu().numpy()[:, 0]
            preds_ablated.append(out)
    preds_ablated = np.concatenate(preds_ablated, axis=0)
    preds_ablated_unnorm = preds_ablated * train_std + train_mean
    max_y_per_patch = np.max(true_unnorm, axis=(1, 2))
    best_patch_idx = np.argmax(max_y_per_patch)
    gt_map = true_unnorm[best_patch_idx]
    recon_map = preds_full_unnorm[best_patch_idx]
    print('Computing power spectra and r_ell...')
    cl_true_list = []
    cl_pred_full_list = []
    cl_pred_ablated_list = []
    cl_cross_full_list = []
    cl_cross_ablated_list = []
    for i in range(len(true_unnorm)):
        cl_t, ell = utils.powers(true_unnorm[i], true_unnorm[i], ps=5.0)
        cl_p_full, _ = utils.powers(preds_full_unnorm[i], preds_full_unnorm[i], ps=5.0)
        cl_p_ablated, _ = utils.powers(preds_ablated_unnorm[i], preds_ablated_unnorm[i], ps=5.0)
        cl_cross_full, _ = utils.powers(preds_full_unnorm[i], true_unnorm[i], ps=5.0)
        cl_cross_ablated, _ = utils.powers(preds_ablated_unnorm[i], true_unnorm[i], ps=5.0)
        cl_true_list.append(cl_t)
        cl_pred_full_list.append(cl_p_full)
        cl_pred_ablated_list.append(cl_p_ablated)
        cl_cross_full_list.append(cl_cross_full)
        cl_cross_ablated_list.append(cl_cross_ablated)
    cl_true_mean = np.real(np.mean(cl_true_list, axis=0))
    cl_pred_full_mean = np.real(np.mean(cl_pred_full_list, axis=0))
    cl_pred_ablated_mean = np.real(np.mean(cl_pred_ablated_list, axis=0))
    cl_cross_full_mean = np.real(np.mean(cl_cross_full_list, axis=0))
    cl_cross_ablated_mean = np.real(np.mean(cl_cross_ablated_list, axis=0))
    denom_full = np.sqrt(np.clip(cl_pred_full_mean * cl_true_mean, 0.0, None))
    r_ell_full = np.divide(cl_cross_full_mean, denom_full, out=np.zeros_like(cl_cross_full_mean), where=denom_full!=0)
    denom_ablated = np.sqrt(np.clip(cl_pred_ablated_mean * cl_true_mean, 0.0, None))
    r_ell_ablated = np.divide(cl_cross_ablated_mean, denom_ablated, out=np.zeros_like(cl_cross_ablated_mean), where=denom_ablated!=0)
    print('Computing residual bias...')
    true_flat = true_unnorm.flatten()
    pred_flat = preds_full_unnorm.flatten()
    mask = true_flat > 1e-7
    true_sig = true_flat[mask]
    pred_sig = pred_flat[mask]
    bins = np.logspace(-7, -4, 15)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    bias_abs = []
    bias_std = []
    for i in range(len(bins)-1):
        b_mask = (true_sig >= bins[i]) & (true_sig < bins[i+1])
        n_pixels = np.sum(b_mask)
        if n_pixels > 0:
            diff = pred_sig[b_mask] - true_sig[b_mask]
            bias_abs.append(np.mean(diff))
            bias_std.append(np.std(diff) / np.sqrt(n_pixels))
        else:
            bias_abs.append(np.nan)
            bias_std.append(np.nan)
    print('Generating plots...')
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 6, figure=fig, wspace=0.6, hspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0:3])
    im1 = ax1.imshow(gt_map, cmap='magma', origin='lower')
    ax1.set_title('(a) Ground Truth tSZ Map')
    ax1.set_xlabel('Pixel X')
    ax1.set_ylabel('Pixel Y')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Compton-y')
    ax2 = fig.add_subplot(gs[0, 3:6])
    im2 = ax2.imshow(recon_map, cmap='magma', origin='lower', vmin=gt_map.min(), vmax=gt_map.max())
    ax2.set_title('Reconstructed tSZ Map (Full Model)')
    ax2.set_xlabel('Pixel X')
    ax2.set_ylabel('Pixel Y')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Compton-y')
    valid_idx = ell > 0
    ell_valid = ell[valid_idx]
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.plot(ell_valid, r_ell_full[valid_idx], label='Full Model (with CIB)', linewidth=2)
    ax3.plot(ell_valid, r_ell_ablated[valid_idx], label='Ablated Model (no CIB)', linewidth=2, linestyle='--')
    ax3.set_title('(b) Cross-correlation Coefficient')
    ax3.set_xlabel('Multipole ell')
    ax3.set_ylabel('r_ell')
    ax3.set_ylim(0, 1.05)
    ax3.set_xlim(left=100)
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True)
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax4.plot(ell_valid, cl_true_mean[valid_idx], label='Ground Truth', linewidth=2)
    ax4.plot(ell_valid, cl_pred_full_mean[valid_idx], label='Reconstructed (Full)', linewidth=2, linestyle='--')
    ax4.plot(ell_valid, cl_pred_ablated_mean[valid_idx], label='Reconstructed (Ablated)', linewidth=2, linestyle=':')
    ax4.set_title('(d) Power Spectra')
    ax4.set_xlabel('Multipole ell')
    ax4.set_ylabel('C_ell')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlim(left=100)
    ax4.legend()
    ax4.grid(True)
    ax5 = fig.add_subplot(gs[1, 4:6])
    ax5.axhline(0, color='k', linestyle='--')
    ax5.errorbar(bin_centers, bias_abs, yerr=bias_std, fmt='o-', color='red', capsize=3, linewidth=2)
    ax5.set_xscale('log')
    ax5.set_title('(c) Absolute Residual Bias vs tSZ Intensity')
    ax5.set_xlabel('True tSZ Intensity (Compton-y)')
    ax5.set_ylabel('Bias (Pred - True) [Compton-y]')
    ax5.grid(True)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = 'data/performance_assessment_3_' + str(timestamp) + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)
    np.savez('data/summary_statistics.npz', ell=ell, r_ell_full=r_ell_full, r_ell_ablated=r_ell_ablated, cl_true_mean=cl_true_mean, cl_pred_full_mean=cl_pred_full_mean, cl_pred_ablated_mean=cl_pred_ablated_mean, bin_centers=bin_centers, bias_abs=bias_abs, bias_std=bias_std)
    print('Summary statistics saved to data/summary_statistics.npz')
    print('\n--- SUMMARY STATISTICS ---')
    print('\nCross-correlation coefficient (r_ell) at selected multipoles:')
    print('ell | Full Model | Ablated Model')
    for target_ell in [500, 1000, 2000, 3000, 4000]:
        idx = np.argmin(np.abs(ell - target_ell))
        print(str(round(ell[idx], 1)) + ' | ' + str(round(r_ell_full[idx], 4)) + ' | ' + str(round(r_ell_ablated[idx], 4)))
    print('\nPower Spectra (C_ell) at selected multipoles:')
    print('ell | Ground Truth | Recon (Full) | Ratio (Full/True)')
    for target_ell in [500, 1000, 2000, 3000, 4000]:
        idx = np.argmin(np.abs(ell - target_ell))
        ratio = cl_pred_full_mean[idx] / cl_true_mean[idx] if cl_true_mean[idx] > 0 else 0
        print(str(round(ell[idx], 1)) + ' | ' + str(cl_true_mean[idx]) + ' | ' + str(cl_pred_full_mean[idx]) + ' | ' + str(round(ratio, 4)))
    print('\nResidual Bias vs True tSZ Intensity:')
    print('Bin Center (y) | Absolute Bias | Std Error')
    for i in range(len(bin_centers)):
        print(str(bin_centers[i]) + ' | ' + str(bias_abs[i]) + ' | ' + str(bias_std[i]))
    cluster_results = np.load('data/cluster_injection_results.npy', allow_pickle=True)
    print('\nCluster Injection Results (from Step 7):')
    for res in cluster_results:
        print('log M_500: ' + str(res['M_log']) + ', Injected Peak: ' + str(res['injected_peak']) + ', Recon Peak: ' + str(res['recon_peak']) + ', Recovery Frac: ' + str(res['recovery_frac']))

if __name__ == '__main__':
    main()