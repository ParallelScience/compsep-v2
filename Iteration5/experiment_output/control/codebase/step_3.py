# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams['text.usetex'] = False

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath('codebase'))
    sys.path.insert(0, '/home/node/data/compsep_data')
    import utils
    from step_2 import SR_DAE

    BASE = '/home/node/data/compsep_data/cut_maps'
    DATA_DIR = 'data'
    TSZ_SCALE = 1e6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits = np.load(os.path.join(DATA_DIR, 'splits.npz'))
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']
    top_5_percent_idx = splits['top_5_percent_idx']

    standard_test_idx = np.setdiff1d(test_idx, top_5_percent_idx)

    norm_stats = np.load(os.path.join(DATA_DIR, 'normalization_stats.npz'))
    obs_mean = torch.tensor(norm_stats['obs_mean'], dtype=torch.float32).to(device)
    obs_std = torch.tensor(norm_stats['obs_std'], dtype=torch.float32).to(device)

    model = SR_DAE().to(device)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'sr_dae_weights.pth'), map_location=device))
    model.eval()

    print('Identifying high-noise SO realizations...')
    so_noise_90 = np.load(BASE + '/so_noise/90.npy', mmap_mode='r')
    so_noise_150 = np.load(BASE + '/so_noise/150.npy', mmap_mode='r')
    so_noise_217 = np.load(BASE + '/so_noise/217.npy', mmap_mode='r')

    n_so = 3000
    so_rms = np.zeros(n_so)
    for i in range(n_so):
        var_90 = np.mean(so_noise_90[i]**2)
        var_150 = np.mean(so_noise_150[i]**2)
        var_217 = np.mean(so_noise_217[i]**2)
        so_rms[i] = np.sqrt(var_90 + var_150 + var_217)

    threshold_95 = np.percentile(so_rms, 95)
    high_noise_so_idx = np.where(so_rms >= threshold_95)[0]
    print('Found ' + str(len(high_noise_so_idx)) + ' high-noise SO realizations (RMS >= ' + str(round(threshold_95, 2)) + ')')

    frequencies = [90, 150, 217, 353, 545, 857]

    def load_patch(p, i_so=None, i_planck=None, pure_noise=False):
        if i_so is None:
            i_so = np.random.randint(3000)
        if i_planck is None:
            i_planck = np.random.randint(100)
        obs = np.zeros((6, 256, 256), dtype=np.float32)
        for i, freq in enumerate(frequencies):
            if not pure_noise:
                signal = np.load(BASE + '/stacked_' + str(freq) + '.npy', mmap_mode='r')[p]
            else:
                signal = 0.0
            if freq <= 217:
                noise = np.load(BASE + '/so_noise/' + str(freq) + '.npy', mmap_mode='r')[i_so]
            else:
                raw = np.load(BASE + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy', mmap_mode='r')[p]
                if freq == 353:
                    noise = raw * 1e6
                else:
                    noise = raw * 1e6 * utils.jysr2uk(freq)
            obs[i] = signal + noise
        tsz = np.load(BASE + '/tsz.npy', mmap_mode='r')[p]
        return torch.tensor(obs).unsqueeze(0), torch.tensor(tsz).unsqueeze(0).unsqueeze(0)

    print('Evaluating Standard Test Set...')
    standard_mses = []
    with torch.no_grad():
        for p in standard_test_idx:
            obs, tsz = load_patch(p)
            obs = obs.to(device)
            tsz = tsz.to(device) * TSZ_SCALE
            obs_norm = (obs - obs_mean) / obs_std
            pred = model(obs_norm[:, :3], obs_norm[:, 3:])
            mse = torch.mean((pred - tsz)**2).item()
            standard_mses.append(mse)
    mean_standard_mse = np.mean(standard_mses)
    print('Standard Test MSE: ' + str(round(mean_standard_mse, 4)))

    print('Evaluating OOD Massive Clusters...')
    ood_cluster_mses = []
    with torch.no_grad():
        for p in top_5_percent_idx:
            obs, tsz = load_patch(p)
            obs = obs.to(device)
            tsz = tsz.to(device) * TSZ_SCALE
            obs_norm = (obs - obs_mean) / obs_std
            pred = model(obs_norm[:, :3], obs_norm[:, 3:])
            mse = torch.mean((pred - tsz)**2).item()
            ood_cluster_mses.append(mse)
    mean_ood_cluster_mse = np.mean(ood_cluster_mses)
    print('OOD Massive Clusters MSE: ' + str(round(mean_ood_cluster_mse, 4)))

    print('Evaluating OOD High Noise...')
    ood_noise_mses = []
    with torch.no_grad():
        for p in standard_test_idx:
            i_so = np.random.choice(high_noise_so_idx)
            obs, tsz = load_patch(p, i_so=i_so)
            obs = obs.to(device)
            tsz = tsz.to(device) * TSZ_SCALE
            obs_norm = (obs - obs_mean) / obs_std
            pred = model(obs_norm[:, :3], obs_norm[:, 3:])
            mse = torch.mean((pred - tsz)**2).item()
            ood_noise_mses.append(mse)
    mean_ood_noise_mse = np.mean(ood_noise_mses)
    print('OOD High Noise MSE: ' + str(round(mean_ood_noise_mse, 4)))

    def integrated_gradients(obs_norm, model, steps=50):
        baseline = torch.zeros_like(obs_norm)
        scaled_inputs = [baseline + (float(i) / steps) * (obs_norm - baseline) for i in range(0, steps + 1)]
        grads = []
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad_(True)
            pred = model(scaled_input[:, :3], scaled_input[:, 3:])
            score = pred.sum()
            model.zero_grad()
            score.backward()
            grads.append(scaled_input.grad.data.cpu().numpy())
        avg_grads = np.mean(grads, axis=0)
        integrated_grad = (obs_norm.cpu().numpy() - baseline.cpu().numpy()) * avg_grads
        return integrated_grad[0]

    tsz_max_vals = [np.max(np.load(BASE + '/tsz.npy', mmap_mode='r')[p]) for p in top_5_percent_idx]
    high_snr_p = top_5_percent_idx[np.argmax(tsz_max_vals)]
    tsz_max_vals_std = [np.max(np.load(BASE + '/tsz.npy', mmap_mode='r')[p]) for p in standard_test_idx]
    low_snr_p = standard_test_idx[np.argmin(tsz_max_vals_std)]

    obs_high, tsz_high = load_patch(high_snr_p)
    obs_high_norm = (obs_high.to(device) - obs_mean) / obs_std
    ig_high = integrated_gradients(obs_high_norm, model)
    obs_low, tsz_low = load_patch(low_snr_p)
    obs_low_norm = (obs_low.to(device) - obs_mean) / obs_std
    ig_low = integrated_gradients(obs_low_norm, model)

    so_saliency_high = np.sum(np.abs(ig_high[:3]), axis=0)
    pl_saliency_high = np.sum(np.abs(ig_high[3:]), axis=0)
    so_saliency_low = np.sum(np.abs(ig_low[:3]), axis=0)
    pl_saliency_low = np.sum(np.abs(ig_low[3:]), axis=0)

    print('Running Null Tests on pure noise realizations...')
    null_preds = []
    with torch.no_grad():
        for i in range(50):
            p = standard_test_idx[i % len(standard_test_idx)]
            obs, _ = load_patch(p, pure_noise=True)
            obs = obs.to(device)
            obs_norm = (obs - obs_mean) / obs_std
            pred = model(obs_norm[:, :3], obs_norm[:, 3:])
            null_preds.append(pred.cpu().numpy().flatten())
    null_preds = np.concatenate(null_preds)
    null_mean = np.mean(null_preds)
    null_std = np.std(null_preds)
    print('Null Test Output - Mean: ' + str(round(null_mean, 6)) + ', Std: ' + str(round(null_std, 6)))

    fig = plt.figure(figsize=(18, 12))
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(tsz_high[0, 0].numpy(), cmap='viridis')
    ax1.set_title('High-SNR: True tSZ')
    plt.colorbar(im1, ax=ax1)
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(so_saliency_high, cmap='magma')
    ax2.set_title('High-SNR: SO Saliency (IG)')
    plt.colorbar(im2, ax=ax2)
    ax3 = plt.subplot(3, 3, 3)
    im3 = ax3.imshow(pl_saliency_high, cmap='magma')
    ax3.set_title('High-SNR: Planck Saliency (IG)')
    plt.colorbar(im3, ax=ax3)
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.imshow(tsz_low[0, 0].numpy(), cmap='viridis')
    ax4.set_title('Low-SNR: True tSZ')
    plt.colorbar(im4, ax=ax4)
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.imshow(so_saliency_low, cmap='magma')
    ax5.set_title('Low-SNR: SO Saliency (IG)')
    plt.colorbar(im5, ax=ax5)
    ax6 = plt.subplot(3, 3, 6)
    im6 = ax6.imshow(pl_saliency_low, cmap='magma')
    ax6.set_title('Low-SNR: Planck Saliency (IG)')
    plt.colorbar(im6, ax=ax6)
    ax7 = plt.subplot(3, 3, (7, 8))
    counts, bins, _ = ax7.hist(null_preds, bins=100, density=True, alpha=0.6, color='blue', label='Null Test Output')
    x = np.linspace(bins[0], bins[-1], 100)
    p_norm = norm.pdf(x, null_mean, null_std)
    ax7.plot(x, p_norm, 'k', linewidth=2, label='Gaussian Fit (mu=' + str(round(null_mean, 2)) + ', sigma=' + str(round(null_std, 2)) + ')')
    ax7.set_title('Null Test: Pure Noise Output Distribution')
    ax7.set_xlabel('Predicted tSZ Signal')
    ax7.set_ylabel('Density')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax8 = plt.subplot(3, 3, 9)
    categories = ['Standard', 'OOD Clusters', 'OOD High Noise']
    mses = [mean_standard_mse, mean_ood_cluster_mse, mean_ood_noise_mse]
    bars = ax8.bar(categories, mses, color=['green', 'red', 'orange'])
    ax8.set_title('MSE Comparison')
    ax8.set_ylabel('Mean Squared Error')
    for bar in bars:
        yval = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2, yval, str(round(yval, 2)), ha='center', va='bottom')
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = 'ood_stress_test_3_' + str(timestamp) + '.png'
    plot_filepath = os.path.join(DATA_DIR, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print('Plot saved to ' + plot_filepath)
    np.savez(os.path.join(DATA_DIR, 'ood_metrics.npz'), mean_standard_mse=mean_standard_mse, mean_ood_cluster_mse=mean_ood_cluster_mse, mean_ood_noise_mse=mean_ood_noise_mse, null_mean=null_mean, null_std=null_std)
    print('OOD metrics saved to ' + os.path.join(DATA_DIR, 'ood_metrics.npz'))