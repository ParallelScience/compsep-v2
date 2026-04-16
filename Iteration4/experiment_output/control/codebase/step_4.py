# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import time

mpl.rcParams['text.usetex'] = False

from step_1 import CompSepDataset
from step_2 import UNet
import utils

class CIBOnlyDataset(CompSepDataset):
    def __getitem__(self, idx):
        p_idx = self.patch_indices[idx]
        i_so = self.rng.integers(0, 3000)
        i_planck = self.rng.integers(0, 100)
        x = np.zeros((len(self.frequencies), 256, 256), dtype=np.float32)
        noise_vars = np.zeros(len(self.frequencies), dtype=np.float32)
        for i, freq in enumerate(self.frequencies):
            if freq <= 217:
                signal = 0.0
                noise = self.so_noise[freq][i_so]
            else:
                signal = self.signals[freq][p_idx]
                raw = np.load(os.path.join(self.base_dir, 'planck_noise', 'planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy'), mmap_mode='r')[p_idx]
                if freq == 353:
                    noise = raw * 1e6
                else:
                    noise = raw * 1e6 * utils.jysr2uk(freq)
            noisy_signal = signal + noise
            noise_vars[i] = np.var(noise)
            median = self.norm_stats[str(freq)]['median']
            iqr = self.norm_stats[str(freq)]['iqr']
            x[i] = (noisy_signal - median) / iqr
        y_tsz = self.tsz[p_idx].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(noise_vars), torch.from_numpy(y_tsz)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    with open('data/splits.json', 'r') as f:
        splits = json.load(f)
    val_idx = splits['val']
    val_dataset = CIBOnlyDataset(val_idx, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    model = UNet(in_channels=6, out_channels=1, cond_dim=6, features=[64, 128, 256, 512]).to(device)
    model.load_state_dict(torch.load('data/sr_dae_model.pth', map_location=device))
    model.eval()
    pixel_corrs = []
    cross_spectra = []
    auto_pred = []
    auto_true = []
    ells = None
    saved_maps = []
    with torch.no_grad():
        for x, noise_vars, y in val_loader:
            x = x.to(device)
            noise_vars = torch.log10(noise_vars + 1e-8).to(device)
            pred = model(x, noise_vars)
            pred_np = pred.cpu().numpy() / 1e6
            y_np = y.cpu().numpy()
            for i in range(pred_np.shape[0]):
                p = pred_np[i, 0]
                t = y_np[i]
                if len(saved_maps) < 3:
                    saved_maps.append((t, p))
                p_flat = p.flatten()
                t_flat = t.flatten()
                if np.std(p_flat) == 0 or np.std(t_flat) == 0:
                    corr = 0.0
                else:
                    corr = np.corrcoef(p_flat, t_flat)[0, 1]
                pixel_corrs.append(corr)
                out1, out2 = utils.powers(p, t, ps=1.17, window_alpha=0.5)
                if np.max(out1) > np.max(out2):
                    ell = out1
                    cross = out2
                else:
                    ell = out2
                    cross = out1
                out1_p, out2_p = utils.powers(p, p, ps=1.17, window_alpha=0.5)
                if np.max(out1_p) > np.max(out2_p): auto_p = out2_p
                else: auto_p = out1_p
                out1_t, out2_t = utils.powers(t, t, ps=1.17, window_alpha=0.5)
                if np.max(out1_t) > np.max(out2_t): auto_t = out2_t
                else: auto_t = out1_t
                if ells is None:
                    ells = np.array(ell)
                cross_spectra.append(cross)
                auto_pred.append(auto_p)
                auto_true.append(auto_t)
    pixel_corrs = np.array(pixel_corrs)
    cross_spectra = np.array(cross_spectra)
    auto_pred = np.array(auto_pred)
    auto_true = np.array(auto_true)
    mean_pixel_corr = np.mean(pixel_corrs)
    std_pixel_corr = np.std(pixel_corrs)
    mean_cross = np.mean(cross_spectra, axis=0)
    mean_auto_p = np.mean(auto_pred, axis=0)
    mean_auto_t = np.mean(auto_true, axis=0)
    r_ell = mean_cross / np.sqrt(mean_auto_p * mean_auto_t + 1e-20)
    np.savez('data/cib_only_results.npz', pixel_corrs=pixel_corrs, ells=ells, mean_cross=mean_cross, mean_auto_p=mean_auto_p, mean_auto_t=mean_auto_t, r_ell=r_ell)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.figure(figsize=(8, 6))
    mask_valid = ells > 0
    plt.plot(ells[mask_valid], r_ell[mask_valid], color='purple', label='CIB-only r_ell')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Multipole ell')
    plt.ylabel('Cross-correlation r_ell')
    plt.title('Cross-correlation between CIB-only Output and Truth')
    plt.xscale('log')
    plt.xlim(np.min(ells[mask_valid]), np.max(ells[mask_valid]))
    plt.ylim(-0.5, 1.0)
    plt.legend()
    plt.grid(True, which='both', ls='-', alpha=0.2)
    plt.tight_layout()
    plot_filename1 = 'data/cib_only_rell_1_' + timestamp + '.png'
    plt.savefig(plot_filename1, dpi=300)
    plt.close()
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    for i in range(3):
        t, p = saved_maps[i]
        im0 = axes[i, 0].imshow(t, cmap='viridis', origin='lower')
        axes[i, 0].set_title('Ground Truth tSZ (Patch ' + str(i+1) + ')')
        axes[i, 0].axis('off')
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        im1 = axes[i, 1].imshow(p, cmap='viridis', origin='lower')
        axes[i, 1].set_title('CIB-only Reconstruction (Patch ' + str(i+1) + ')')
        axes[i, 1].axis('off')
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plot_filename2 = 'data/cib_only_maps_2_' + timestamp + '.png'
    plt.savefig(plot_filename2, dpi=300)
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.hist(pixel_corrs, bins=30, color='green', alpha=0.7)
    plt.axvline(mean_pixel_corr, color='k', linestyle='dashed', linewidth=2, label='Mean: ' + str(round(mean_pixel_corr, 4)))
    plt.xlabel('Pixel-wise Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pixel-wise Correlations (CIB-only vs Truth)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_filename3 = 'data/cib_only_hist_3_' + timestamp + '.png'
    plt.savefig(plot_filename3, dpi=300)
    plt.close()

if __name__ == '__main__':
    main()