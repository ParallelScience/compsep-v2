# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = False

sys.path.insert(0, os.path.abspath('codebase'))
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
from step_2 import CompSepDataset
from step_3 import SR_DAE

def get_power(a, b, ps=5.0):
    res = utils.powers(a, b, ps=ps)
    if np.mean(res[0]) > np.mean(res[1]):
        return res[0], res[1]
    else:
        return res[1], res[0]

def integrated_gradients(model, input_tensor, baseline_tensor, steps=50):
    model.eval()
    alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1).to(input_tensor.device)
    path = baseline_tensor + alphas * (input_tensor - baseline_tensor)
    path.requires_grad = True
    preds = model(path)
    score = preds.sum()
    score.backward()
    grads = path.grad
    avg_grads = grads.mean(dim=0, keepdim=True)
    ig = (input_tensor - baseline_tensor) * avg_grads
    return ig

if __name__ == '__main__':
    print('Starting Performance Assessment and Visualization...')
    data_dir = 'data/'
    base_dir = '/home/node/data/compsep_data/cut_maps'
    scaling_data = np.load(os.path.join(data_dir, 'scaling_params.npz'), allow_pickle=True)
    scaling_params = scaling_data['scaling_params'].item()
    tsz_clip_threshold = scaling_data['tsz_clip_threshold'].item()
    val_indices = scaling_data['val_indices']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SR_DAE(main_in_channels=3, aux_in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(os.path.join(data_dir, 'best_model.pth')))
    model.eval()
    val_dataset = CompSepDataset(val_indices, base_dir, scaling_params, tsz_clip_threshold, split='val')
    all_pred_tsz = []
    all_true_tsz = []
    print('Evaluating model on validation set...')
    with torch.no_grad():
        for i in range(len(val_dataset)):
            x, y = val_dataset[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x)
            pred_np = pred.squeeze().cpu().numpy()
            pred_clipped = np.clip(pred_np, -0.999 * tsz_clip_threshold, 0.999 * tsz_clip_threshold)
            pred_physical = tsz_clip_threshold * np.arctanh(pred_clipped / tsz_clip_threshold)
            true_np = y.numpy()
            true_clipped = np.clip(true_np, -0.999 * tsz_clip_threshold, 0.999 * tsz_clip_threshold)
            true_physical = tsz_clip_threshold * np.arctanh(true_clipped / tsz_clip_threshold)
            all_pred_tsz.append(pred_physical)
            all_true_tsz.append(true_physical)
    all_pred_tsz = np.array(all_pred_tsz)
    all_true_tsz = np.array(all_true_tsz)
    print('Computing cross-correlation coefficient r_ell...')
    ell_list = []
    cl_pt_list = []
    cl_pp_list = []
    cl_tt_list = []
    for pred, true in zip(all_pred_tsz, all_true_tsz):
        ell, cl_pt = get_power(pred, true, ps=5.0)
        _, cl_pp = get_power(pred, pred, ps=5.0)
        _, cl_tt = get_power(true, true, ps=5.0)
        ell_list.append(ell)
        cl_pt_list.append(cl_pt)
        cl_pp_list.append(cl_pp)
        cl_tt_list.append(cl_tt)
    ell = ell_list[0]
    mean_cl_pt = np.mean(cl_pt_list, axis=0)
    mean_cl_pp = np.mean(cl_pp_list, axis=0)
    mean_cl_tt = np.mean(cl_tt_list, axis=0)
    r_ell = mean_cl_pt / np.sqrt(np.maximum(mean_cl_pp * mean_cl_tt, 1e-20))
    print('r_ell computed successfully.')
    for e, r in zip(ell[::10], r_ell[::10]):
        print('  ell=' + ('%.1f' % e) + ': r_ell=' + ('%.4f' % r))
    print('\nPerforming Bias-Variance check...')
    true_flat = all_true_tsz.flatten()
    pred_flat = all_pred_tsz.flatten()
    residuals = pred_flat - true_flat
    bins = np.linspace(true_flat.min(), true_flat.max(), 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bias_mean = []
    bias_std = []
    print('Bin Range'.rjust(25) + ' | ' + 'Mean Residual'.rjust(15) + ' | ' + 'Std Residual'.rjust(15) + ' | ' + 'Count'.rjust(10))
    print('-' * 75)
    for i in range(10):
        mask = (true_flat >= bins[i]) & (true_flat < bins[i+1])
        if i == 9:
            mask = (true_flat >= bins[i]) & (true_flat <= bins[i+1])
        count = np.sum(mask)
        if count > 0:
            b_mean = np.mean(residuals[mask])
            b_std = np.std(residuals[mask])
        else:
            b_mean = np.nan
            b_std = np.nan
        bias_mean.append(b_mean)
        bias_std.append(b_std)
        if i == 9:
            bin_range = '[' + ('%.2e' % bins[i]) + ', ' + ('%.2e' % bins[i+1]) + ']'
        else:
            bin_range = '[' + ('%.2e' % bins[i]) + ', ' + ('%.2e' % bins[i+1]) + ')'
        print(bin_range.rjust(25) + ' | ' + ('%.4e' % b_mean).rjust(15) + ' | ' + ('%.4e' % b_std).rjust(15) + ' | ' + str(count).rjust(10))
    print('\nGenerating Integrated Gradients saliency maps...')
    peak_tsz = np.max(all_true_tsz, axis=(1, 2))
    high_mass_idx = np.argmax(peak_tsz)
    p25 = np.percentile(peak_tsz, 25)
    low_mass_idx = np.argmin(np.abs(peak_tsz - p25))
    print('Selected High-Mass Patch Index: ' + str(high_mass_idx) + ' (Peak tSZ: ' + ('%.4e' % peak_tsz[high_mass_idx]) + ')')
    print('Selected Low-Mass Patch Index: ' + str(low_mass_idx) + ' (Peak tSZ: ' + ('%.4e' % peak_tsz[low_mass_idx]) + ')')
    def compute_and_plot_saliency(idx, name):
        x, y = val_dataset[idx]
        x_tensor = x.unsqueeze(0).to(device)
        baseline = torch.zeros_like(x_tensor)
        ig = integrated_gradients(model, x_tensor, baseline, steps=50)
        ig_np = ig.squeeze().cpu().detach().numpy()
        input_var = np.var(x.numpy(), axis=(1, 2), keepdims=True)
        ig_normalized = ig_np / (input_var + 1e-8)
        ig_main = np.mean(np.abs(ig_normalized[:3]), axis=0)
        ig_aux = np.mean(np.abs(ig_normalized[3:]), axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axes[0].imshow(all_true_tsz[idx], cmap='magma')
        axes[0].set_title('True tSZ (' + name + ')\n[dimensionless]')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(ig_main, cmap='viridis')
        axes[1].set_title('Saliency: SO Channels\n(Mean Abs, Normalized)')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(ig_aux, cmap='viridis')
        axes[2].set_title('Saliency: Planck CIB Channels\n(Mean Abs, Normalized)')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        timestamp = str(int(time.time()))
        plot_filename = os.path.join(data_dir, 'saliency_' + name.replace(' ', '_').lower() + '_' + timestamp + '.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print('Saliency plot saved to ' + plot_filename)
        plt.close()
    compute_and_plot_saliency(high_mass_idx, 'High Mass Cluster')
    compute_and_plot_saliency(low_mass_idx, 'Low Mass Filament')
    print('\nSaving all numerical results to a structured file...')
    inj_results_path = os.path.join(data_dir, 'signal_injection_results.npz')
    if os.path.exists(inj_results_path):
        inj_data = np.load(inj_results_path, allow_pickle=True)
        integrated_Y_recovery = inj_data['results']
    else:
        integrated_Y_recovery = np.array([])
        print('Warning: signal_injection_results.npz not found.')
    save_path = os.path.join(data_dir, 'performance_results.npz')
    np.savez(save_path, ell=ell, r_ell=r_ell, bias_bin_centers=bin_centers, bias_mean=np.array(bias_mean), bias_std=np.array(bias_std), integrated_Y_recovery=integrated_Y_recovery)
    print('Numerical results saved to ' + save_path)
    print('Performance Assessment and Visualization completed successfully.')