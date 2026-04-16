# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data/')
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import utils
from step_1 import FlamingoDataset
from step_2 import DualBranchUNet
def safe_powers(a, b, ps=5.0):
    out1, out2 = utils.powers(a, b, ps=ps)
    if np.mean(out1) > np.mean(out2):
        ell = out1
        cl = out2
    else:
        ell = out2
        cl = out1
    return ell, cl
def get_null_batch(dataset, batch_indices):
    so_inputs = []
    so_vars = []
    cib_inputs = []
    for idx in batch_indices:
        i_patch = dataset.indices[idx]
        i_so = torch.randint(0, dataset.n_so, (1,)).item()
        i_planck = torch.randint(0, dataset.n_planck, (1,)).item()
        so_patch = []
        so_var_patch = []
        for freq in dataset.so_freqs:
            noise = dataset.so_noise[freq][i_so]
            so_patch.append(noise)
            so_var_patch.append(dataset.so_noise_var[freq])
        planck_patch = []
        for freq in dataset.planck_freqs:
            raw_noise = dataset.planck_noise[freq][i_planck][i_patch]
            if freq == 353:
                noise = raw_noise * 1e6
            else:
                noise = raw_noise * 1e6 * utils.jysr2uk(freq)
            obs_trans = np.arcsinh(noise / dataset.cib_obs_scales[freq])
            planck_patch.append(obs_trans)
        so_inputs.append(np.stack(so_patch))
        so_vars.append(np.stack(so_var_patch))
        cib_inputs.append(np.stack(planck_patch))
    return (torch.tensor(np.stack(so_inputs), dtype=torch.float32), torch.tensor(np.stack(so_vars), dtype=torch.float32), torch.tensor(np.stack(cib_inputs), dtype=torch.float32))
if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    print('Loading test dataset...')
    start_time = time.time()
    test_dataset = FlamingoDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print('Test dataset loaded in ' + str(round(time.time() - start_time, 2)) + ' seconds.')
    print('Loading model...')
    model = DualBranchUNet().to(device)
    model.load_state_dict(torch.load('data/best_model.pth', map_location=device))
    model.eval()
    tsz_scale = test_dataset.tsz_scale
    all_rmse = []
    all_ssim = []
    all_cl_pred = []
    all_cl_gt = []
    all_cl_cross = []
    all_cl_res = []
    ell_curr = None
    print('Evaluating on test set...')
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            so_inputs = batch['so_inputs'].to(device)
            so_vars = batch['so_vars'].to(device)
            cib_inputs = batch['cib_inputs'].to(device)
            tsz_gt_trans = batch['tsz_gt'].cpu().numpy()
            pred_trans = model(so_inputs, so_vars, cib_inputs).cpu().numpy()
            pred_phys = np.sinh(pred_trans) * tsz_scale
            gt_phys = np.sinh(tsz_gt_trans) * tsz_scale
            for i in range(pred_phys.shape[0]):
                p = np.nan_to_num(pred_phys[i, 0])
                g = np.nan_to_num(gt_phys[i, 0])
                p = p - np.mean(p)
                g = g - np.mean(g)
                rmse = np.sqrt(np.mean((p - g)**2))
                all_rmse.append(rmse)
                data_range = g.max() - g.min()
                if data_range == 0: data_range = 1e-9
                s = ssim(g, p, data_range=data_range)
                all_ssim.append(s)
                ell_curr, cl_p = safe_powers(p, p, ps=5.0)
                _, cl_g = safe_powers(g, g, ps=5.0)
                _, cl_cross = safe_powers(p, g, ps=5.0)
                _, cl_r = safe_powers(p - g, p - g, ps=5.0)
                all_cl_pred.append(cl_p)
                all_cl_gt.append(cl_g)
                all_cl_cross.append(cl_cross)
                all_cl_res.append(cl_r)
    mean_cl_pred = np.mean(all_cl_pred, axis=0)
    mean_cl_gt = np.mean(all_cl_gt, axis=0)
    mean_cl_cross = np.mean(all_cl_cross, axis=0)
    mean_cl_res = np.mean(all_cl_res, axis=0)
    r_ell = mean_cl_cross / np.sqrt(np.abs(mean_cl_pred * mean_cl_gt) + 1e-20)
    print('Running null test...')
    null_preds = []
    gt_rms_list = []
    with torch.no_grad():
        for batch_idx in range(0, len(test_dataset), 16):
            batch_indices = list(range(batch_idx, min(batch_idx + 16, len(test_dataset))))
            so_in, so_v, cib_in = get_null_batch(test_dataset, batch_indices)
            so_in = so_in.to(device)
            so_v = so_v.to(device)
            cib_in = cib_in.to(device)
            pred_trans = model(so_in, so_v, cib_in).cpu().numpy()
            pred_phys = np.sinh(pred_trans) * tsz_scale
            for i, idx in enumerate(batch_indices):
                p = np.nan_to_num(pred_phys[i, 0])
                p = p - np.mean(p)
                null_preds.append(np.sqrt(np.mean(p**2)))
                i_patch = test_dataset.indices[idx]
                g = np.nan_to_num(test_dataset.tsz[i_patch])
                g = g - np.mean(g)
                gt_rms_list.append(np.sqrt(np.mean(g**2)))
    null_rms = np.mean(null_preds)
    gt_rms = np.mean(gt_rms_list)
    hallucination_fraction = null_rms / gt_rms
    mean_rmse = np.mean(all_rmse)
    std_rmse = np.std(all_rmse)
    mean_ssim = np.mean(all_ssim)
    std_ssim = np.std(all_ssim)
    print('\n--- Evaluation Metrics on Test Set ---')
    print('Mean RMSE: ' + str(mean_rmse) + ' +- ' + str(std_rmse))
    print('Mean SSIM: ' + str(mean_ssim) + ' +- ' + str(std_ssim))
    print('\n--- Cross-Correlation Coefficient r_ell ---')
    step = max(1, len(ell_curr) // 10)
    for i in range(0, len(ell_curr), step):
        print('ell = ' + str(round(ell_curr[i], 1)) + ': r_ell = ' + str(round(r_ell[i], 4)))
    print('\n--- Null Test Results ---')
    print('Null Test Hallucination RMS: ' + str(null_rms))
    print('Ground Truth tSZ RMS: ' + str(gt_rms))
    print('Hallucination Fraction: ' + str(round(hallucination_fraction * 100, 2)) + '%')
    print('\nGenerating multi-panel figure...')
    batch = next(iter(test_loader))
    so_inputs = batch['so_inputs'].to(device)
    so_vars = batch['so_vars'].to(device)
    cib_inputs = batch['cib_inputs'].to(device)
    tsz_gt_trans = batch['tsz_gt'].cpu().numpy()
    with torch.no_grad():
        pred_trans = model(so_inputs, so_vars, cib_inputs).cpu().numpy()
    pred_phys = np.sinh(pred_trans) * tsz_scale
    gt_phys = np.sinh(tsz_gt_trans) * tsz_scale
    p0 = np.nan_to_num(pred_phys[0, 0])
    g0 = np.nan_to_num(gt_phys[0, 0])
    p0 = p0 - np.mean(p0)
    g0 = g0 - np.mean(g0)
    res0 = p0 - g0
    ell_plot, cl_p0 = safe_powers(p0, p0, ps=5.0)
    _, cl_g0 = safe_powers(g0, g0, ps=5.0)
    _, cl_r0 = safe_powers(res0, res0, ps=5.0)
    cl_p0 = np.maximum(cl_p0, 1e-20)
    cl_g0 = np.maximum(cl_g0, 1e-20)
    cl_r0 = np.maximum(cl_r0, 1e-20)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    vmax_tsz = max(np.percentile(np.abs(g0), 99), np.percentile(np.abs(p0), 99))
    if vmax_tsz == 0: vmax_tsz = 1e-9
    im0 = axs[0, 0].imshow(g0, cmap='RdBu_r', vmin=-vmax_tsz, vmax=vmax_tsz)
    axs[0, 0].set_title('Ground Truth tSZ (Mean Subtracted)')
    fig.colorbar(im0, ax=axs[0, 0], label='Compton-y')
    im1 = axs[0, 1].imshow(p0, cmap='RdBu_r', vmin=-vmax_tsz, vmax=vmax_tsz)
    axs[0, 1].set_title('Reconstructed tSZ (Mean Subtracted)')
    fig.colorbar(im1, ax=axs[0, 1], label='Compton-y')
    vmax_res = np.percentile(np.abs(res0), 99)
    if vmax_res == 0: vmax_res = 1e-9
    im2 = axs[1, 0].imshow(res0, cmap='RdBu_r', vmin=-vmax_res, vmax=vmax_res)
    axs[1, 0].set_title('Residual (Pred - GT)')
    fig.colorbar(im2, ax=axs[1, 0], label='Compton-y')
    axs[1, 1].plot(ell_plot, cl_g0, label='Ground Truth')
    axs[1, 1].plot(ell_plot, cl_p0, label='Reconstruction')
    axs[1, 1].plot(ell_plot, cl_r0, label='Residual')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlabel('Multipole ell')
    axs[1, 1].set_ylabel('C_ell')
    axs[1, 1].set_title('Angular Power Spectra')
    axs[1, 1].legend()
    metrics_text = ('Mean RMSE: ' + str(round(mean_rmse, 8)) + ' | ' + 'Mean SSIM: ' + str(round(mean_ssim, 4)) + ' | ' + 'Hallucination Frac: ' + str(round(hallucination_fraction * 100, 2)) + '%')
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'data/evaluation_plot_1_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)