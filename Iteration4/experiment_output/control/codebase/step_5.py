# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from datetime import datetime
from step_1 import CompSepDataset
from step_2 import UNet

mpl.rcParams['text.usetex'] = False

sys.path.insert(0, '/home/node/data/compsep_data/')
import utils

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))

    with open('data/splits.json', 'r') as f:
        splits = json.load(f)
    test_idx = splits['test']
    print('Test samples: ' + str(len(test_idx)))

    test_dataset = CompSepDataset(test_idx, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = UNet(in_channels=6, out_channels=1, cond_dim=6, features=[64, 128, 256, 512]).to(device)
    model.load_state_dict(torch.load('data/sr_dae_model.pth', map_location=device))
    model.eval()

    ps_R_list = []
    ps_T_list = []
    ells = None

    print('Computing residuals and power spectra...')
    with torch.no_grad():
        for x, noise_vars, y in test_loader:
            x = x.to(device)
            noise_vars = torch.log10(noise_vars + 1e-8).to(device)
            pred = model(x, noise_vars)
            
            pred_np = pred.cpu().numpy()[:, 0, :, :] / 1e6
            y_np = y.cpu().numpy()
            
            R_np = y_np - pred_np
            
            for i in range(R_np.shape[0]):
                r = R_np[i]
                t = y_np[i]
                
                out1_r, out2_r = utils.powers(r, r, ps=1.17, window_alpha=0.5)
                if np.max(out1_r) > np.max(out2_r):
                    ell = out1_r
                    ps_r = out2_r
                else:
                    ell = out2_r
                    ps_r = out1_r
                    
                out1_t, out2_t = utils.powers(t, t, ps=1.17, window_alpha=0.5)
                if np.max(out1_t) > np.max(out2_t):
                    ps_t = out2_t
                else:
                    ps_t = out1_t
                    
                if ells is None:
                    ells = np.array(ell)
                    
                ps_R_list.append(ps_r)
                ps_T_list.append(ps_t)

    ps_R_arr = np.array(ps_R_list)
    ps_T_arr = np.array(ps_T_list)
    
    mean_ps_R = np.mean(ps_R_arr, axis=0)
    mean_ps_T = np.mean(ps_T_arr, axis=0)
    
    B_ell = mean_ps_R / (mean_ps_T + 1e-20)
    
    target_ells = [1000, 3000, 5000, 8000]
    print('Scale-dependent bias B(ell) = mean(R_ell) / mean(truth_ell):')
    for target in target_ells:
        idx = np.argmin(np.abs(ells - target))
        actual_ell = ells[idx]
        bias_val = B_ell[idx]
        print('  ell approx ' + str(int(actual_ell)) + ' (target ' + str(target) + '): ' + str(round(bias_val, 6)))
        
    np.savez('data/residual_statistics.npz', ells=ells, mean_ps_R=mean_ps_R, mean_ps_T=mean_ps_T, B_ell=B_ell)
    print('Residual statistics saved to data/residual_statistics.npz')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    mask_valid = ells > 0
    ells_plot = ells[mask_valid]
    mean_ps_R_plot = np.clip(mean_ps_R[mask_valid], 1e-20, None)
    mean_ps_T_plot = np.clip(mean_ps_T[mask_valid], 1e-20, None)
    B_ell_plot = B_ell[mask_valid]
    
    axes[0].plot(ells_plot, mean_ps_T_plot, label='Truth', color='black')
    axes[0].plot(ells_plot, mean_ps_R_plot, label='Residual', color='red', linestyle='--')
    axes[0].set_xlabel('Multipole ell')
    axes[0].set_ylabel('Power Spectrum')
    axes[0].set_title('Power Spectra vs ell')
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True, which='both', ls='-', alpha=0.2)
    
    axes[1].plot(ells_plot, B_ell_plot, label='Bias B(ell)', color='blue')
    axes[1].axhline(1.0, color='k', linestyle=':', alpha=0.5)
    axes[1].axhline(0.0, color='k', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('Multipole ell')
    axes[1].set_ylabel('Bias B(ell) = R_ell / Truth_ell')
    axes[1].set_title('Scale-dependent Bias')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True, which='both', ls='-', alpha=0.2)
    
    sc = axes[2].scatter(mean_ps_T_plot, mean_ps_R_plot, c=ells_plot, cmap='viridis', norm=mpl.colors.LogNorm())
    min_val = min(np.min(mean_ps_T_plot), np.min(mean_ps_R_plot))
    max_val = max(np.max(mean_ps_T_plot), np.max(mean_ps_R_plot))
    axes[2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    axes[2].set_xlabel('Truth Power Spectrum')
    axes[2].set_ylabel('Residual Power Spectrum')
    axes[2].set_title('Residual vs Truth Power Spectrum')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, which='both', ls='-', alpha=0.2)
    cbar = plt.colorbar(sc, ax=axes[2])
    cbar.set_label('Multipole ell')
    
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = 'data/residual_analysis_' + str(timestamp) + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)

if __name__ == '__main__':
    main()