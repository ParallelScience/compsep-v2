# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import torch
import scipy.ndimage as ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import time

mpl.rcParams['text.usetex'] = False

from step_1 import CompSepDataset
from step_2 import UNet

def find_clusters_and_integrate(t_map, p_map, threshold=3e-6, radius_arcmin=5.0, pixel_size_arcmin=1.17):
    radius_px = radius_arcmin / pixel_size_arcmin
    size = int(2 * radius_px) + 1
    local_max = ndimage.maximum_filter(t_map, size=size) == t_map
    mask = t_map > threshold
    peaks = local_max & mask
    y_coords, x_coords = np.where(peaks)
    clusters = []
    Y, X = np.ogrid[:t_map.shape[0], :t_map.shape[1]]
    for y, x in zip(y_coords, x_coords):
        mass_proxy = t_map[y, x]
        dist_sq = (Y - y)**2 + (X - x)**2
        aperture_mask = dist_sq <= radius_px**2
        true_ysz = np.sum(t_map[aperture_mask])
        pred_ysz = np.sum(p_map[aperture_mask])
        if true_ysz > 0 and pred_ysz > 0:
            clusters.append({'mass_proxy': mass_proxy, 'true_ysz': true_ysz, 'pred_ysz': pred_ysz})
    return clusters

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
    model_path = 'data/sr_dae_model_curriculum.pth'
    if not os.path.exists(model_path):
        model_path = 'data/sr_dae_model.pth'
    print('Loading model from: ' + model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_clusters = []
    print('Finding clusters and computing Y_SZ...')
    with torch.no_grad():
        for x, noise_vars, y in test_loader:
            x = x.to(device)
            noise_vars = torch.log10(noise_vars + 1e-8).to(device)
            pred = model(x, noise_vars)
            pred_np = pred.cpu().numpy()[:, 0, :, :] / 1e6
            y_np = y.cpu().numpy()
            for i in range(y_np.shape[0]):
                t_map = y_np[i]
                p_map = pred_np[i]
                clusters = find_clusters_and_integrate(t_map, p_map, threshold=3e-6)
                all_clusters.extend(clusters)
    print('Total clusters found: ' + str(len(all_clusters)))
    if len(all_clusters) == 0:
        print('No clusters found! Check the threshold.')
        return
    mass_proxies = np.array([c['mass_proxy'] for c in all_clusters])
    true_ysz = np.array([c['true_ysz'] for c in all_clusters])
    pred_ysz = np.array([c['pred_ysz'] for c in all_clusters])
    np.savez('data/ysz_m_relation.npz', mass_proxies=mass_proxies, true_ysz=true_ysz, pred_ysz=pred_ysz)
    print('Y_SZ-M relation data saved to data/ysz_m_relation.npz')
    min_mass = np.min(mass_proxies)
    max_mass = np.max(mass_proxies)
    bins = np.logspace(np.log10(min_mass), np.log10(max_mass), num=10)
    bin_centers = []
    true_scatter = []
    pred_scatter = []
    print('\nScatter by Mass Proxy Bin:')
    for i in range(len(bins)-1):
        mask = (mass_proxies >= bins[i]) & (mass_proxies < bins[i+1])
        count = np.sum(mask)
        if count > 5:
            m_bin = mass_proxies[mask]
            t_bin = true_ysz[mask]
            p_bin = pred_ysz[mask]
            log_m = np.log10(m_bin)
            log_t = np.log10(t_bin)
            log_p = np.log10(p_bin)
            p_t = np.polyfit(log_m, log_t, 1)
            sc_t = np.std(log_t - np.polyval(p_t, log_m))
            p_p = np.polyfit(log_m, log_p, 1)
            sc_p = np.std(log_p - np.polyval(p_p, log_m))
            center = np.sqrt(bins[i]*bins[i+1])
            bin_centers.append(center)
            true_scatter.append(sc_t)
            pred_scatter.append(sc_p)
            print('Bin ' + str(round(bins[i], 6)) + ' - ' + str(round(bins[i+1], 6)) + ' (N=' + str(count) + '): True Scatter = ' + str(round(sc_t, 4)) + ', Pred Scatter = ' + str(round(sc_p, 4)))
    log_m_all = np.log10(mass_proxies)
    log_t_all = np.log10(true_ysz)
    log_p_all = np.log10(pred_ysz)
    p_t_all = np.polyfit(log_m_all, log_t_all, 1)
    sc_t_all = np.std(log_t_all - np.polyval(p_t_all, log_m_all))
    p_p_all = np.polyfit(log_m_all, log_p_all, 1)
    sc_p_all = np.std(log_p_all - np.polyval(p_p_all, log_m_all))
    print('\nOverall True Scatter: ' + str(round(sc_t_all, 4)))
    print('Overall Pred Scatter: ' + str(round(sc_p_all, 4)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(mass_proxies, true_ysz, alpha=0.5, label='Ground Truth', color='blue', s=10)
    axes[0].scatter(mass_proxies, pred_ysz, alpha=0.5, label='Reconstructed', color='red', s=10)
    m_line = np.logspace(np.log10(min_mass), np.log10(max_mass), 100)
    axes[0].plot(m_line, 10**np.polyval(p_t_all, np.log10(m_line)), color='darkblue', linestyle='--', label='True Fit')
    axes[0].plot(m_line, 10**np.polyval(p_p_all, np.log10(m_line)), color='darkred', linestyle='--', label='Pred Fit')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Peak tSZ Amplitude (Mass Proxy)')
    axes[0].set_ylabel('Aperture-integrated Y_SZ')
    axes[0].set_title('Y_SZ - M Relation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if len(bin_centers) > 0:
        axes[1].plot(bin_centers, true_scatter, marker='o', color='blue', label='True Scatter')
        axes[1].plot(bin_centers, pred_scatter, marker='s', color='red', label='Pred Scatter')
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Peak tSZ Amplitude (Mass Proxy)')
        axes[1].set_ylabel('Scatter (std of log residuals)')
        axes[1].set_title('Y_SZ - M Scatter vs Mass Proxy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = 'data/ysz_m_relation_1_' + str(timestamp) + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)

if __name__ == '__main__':
    main()