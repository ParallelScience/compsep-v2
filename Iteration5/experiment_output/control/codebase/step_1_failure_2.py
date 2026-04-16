# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import multiprocessing as mp
import time
import utils
plt.rcParams['text.usetex'] = False
BASE = '/home/node/data/compsep_data/cut_maps'
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
frequencies = [90, 150, 217, 353, 545, 857]
n_freq = len(frequencies)
PS = 5 * 60 / 256
def init_worker():
    os.environ['OMP_NUM_THREADS'] = '1'
def process_patch(args):
    p, i_planck, i_so = args
    obs = np.zeros((n_freq, 256, 256))
    cib_mean = np.zeros(n_freq)
    for i, freq in enumerate(frequencies):
        signal = np.load(BASE + '/stacked_' + str(freq) + '.npy')[p]
        if freq <= 217:
            noise = np.load(BASE + '/so_noise/' + str(freq) + '.npy')[i_so]
        else:
            raw = np.load(BASE + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy')[p]
            if freq == 353:
                noise = raw * 1e6
            else:
                noise = raw * 1e6 * utils.jysr2uk(freq)
        obs[i] = signal + noise
        cib = np.load(BASE + '/cib_' + str(freq) + '.npy')[p]
        cib_mean[i] = np.mean(cib) * utils.jysr2uk(freq)
    tsz = np.load(BASE + '/tsz.npy')[p]
    obs_flat = obs.reshape(n_freq, -1)
    sum_xxT = obs_flat @ obs_flat.T
    sum_x = np.sum(obs_flat, axis=1)
    ell, cl_tsz = utils.powers(tsz, tsz, ps=PS)
    tsz_response = np.array([utils.tsz(f) for f in frequencies])
    noise_obs = obs - tsz_response[:, None, None] * tsz
    n_ell = len(ell)
    cl_noise = np.zeros((n_ell, n_freq, n_freq))
    for i in range(n_freq):
        for j in range(i, n_freq):
            _, cl_n = utils.powers(noise_obs[i], noise_obs[j], ps=PS)
            cl_noise[:, i, j] = cl_n
            if i != j:
                cl_noise[:, j, i] = cl_n
    return sum_xxT, sum_x, cl_tsz, cl_noise, ell, cib_mean
def evaluate_patch(args):
    p, i_planck, i_so, w_cilc, W_2D = args
    obs = np.zeros((n_freq, 256, 256))
    for i, freq in enumerate(frequencies):
        signal = np.load(BASE + '/stacked_' + str(freq) + '.npy')[p]
        if freq <= 217:
            noise = np.load(BASE + '/so_noise/' + str(freq) + '.npy')[i_so]
        else:
            raw = np.load(BASE + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy')[p]
            if freq == 353:
                noise = raw * 1e6
            else:
                noise = raw * 1e6 * utils.jysr2uk(freq)
        obs[i] = signal + noise
    tsz_true = np.load(BASE + '/tsz.npy')[p]
    tsz_cilc = np.tensordot(w_cilc, obs, axes=(0, 0))
    obs_fft = np.fft.fft2(obs, axes=(1, 2))
    tsz_wf_fft = np.sum(W_2D * obs_fft, axis=0)
    tsz_wf = np.real(np.fft.ifft2(tsz_wf_fft))
    mse_cilc = np.mean((tsz_cilc - tsz_true)**2)
    mse_wf = np.mean((tsz_wf - tsz_true)**2)
    ell, cl_res_cilc = utils.powers(tsz_cilc - tsz_true, tsz_cilc - tsz_true, ps=PS)
    _, cl_res_wf = utils.powers(tsz_wf - tsz_true, tsz_wf - tsz_true, ps=PS)
    return mse_cilc, mse_wf, cl_res_cilc, cl_res_wf, tsz_true, tsz_cilc, tsz_wf
if __name__ == '__main__':
    tsz_response = np.array([utils.tsz(f) for f in frequencies])
    n_patch = 1523
    max_tsz = np.zeros(n_patch)
    for p in range(n_patch):
        tsz = np.load(BASE + '/tsz.npy')[p]
        max_tsz[p] = np.max(tsz)
    top_5_percent_idx = np.argsort(max_tsz)[-int(0.05 * n_patch):]
    all_patches = np.arange(n_patch)
    safe_patches = np.setdiff1d(all_patches, top_5_percent_idx)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(safe_patches)
    n_train = int(0.70 * n_patch)
    train_patches = safe_patches[:n_train]
    test_patches = np.concatenate([safe_patches[n_train:], top_5_percent_idx])
    tasks = []
    for p in train_patches:
        i_planck = rng.integers(100)
        i_so = rng.integers(3000)
        tasks.append((p, i_planck, i_so))
    with mp.Pool(processes=16, initializer=init_worker) as pool:
        results = pool.map(process_patch, tasks)
    sum_xxT_total = np.zeros((n_freq, n_freq))
    sum_x_total = np.zeros(n_freq)
    cl_tsz_total = None
    cl_noise_total = None
    ell_bins = None
    mean_cib_spectrum = np.zeros(n_freq)
    n_pixels_per_patch = 256 * 256
    n_train_actual = len(train_patches)
    total_pixels = n_train_actual * n_pixels_per_patch
    for res in results:
        sum_xxT, sum_x, cl_tsz, cl_noise, ell, cib_mean = res
        sum_xxT_total += sum_xxT
        sum_x_total += sum_x
        if cl_tsz_total is None:
            cl_tsz_total = np.zeros_like(cl_tsz)
            cl_noise_total = np.zeros_like(cl_noise)
            ell_bins = ell
        cl_tsz_total += cl_tsz
        cl_noise_total += cl_noise
        mean_cib_spectrum += cib_mean
    mean_x = sum_x_total / total_pixels
    C_data = (sum_xxT_total / total_pixels) - np.outer(mean_x, mean_x)
    cl_tsz_mean = cl_tsz_total / n_train_actual
    cl_noise_mean = cl_noise_total / n_train_actual
    mean_cib_spectrum /= n_train_actual
    scale_factor = 1e-6
    tsz_scaled = tsz_response * scale_factor
    A = np.vstack([tsz_scaled, mean_cib_spectrum]).T
    F = np.array([1.0, 0.0])
    C_inv = np.linalg.pinv(C_data, rcond=1e-8)
    term1 = C_inv @ A
    term2 = np.linalg.pinv(A.T @ C_inv @ A, rcond=1e-8)
    w_cilc_scaled = term1 @ term2 @ F
    w_cilc = w_cilc_scaled * scale_factor
    n_ell = len(ell_bins)
    W_ell = np.zeros((n_ell, n_freq))
    A_wf = tsz_response.reshape(-1, 1)
    for l in range(n_ell):
        C_s = cl_tsz_mean[l]
        C_n = cl_noise_mean[l]
        reg = np.eye(n_freq) * np.max(np.diag(C_n)) * 1e-4
        C_n_reg = C_n + reg
        matrix_to_invert = A_wf @ np.array([[C_s]]) @ A_wf.T + C_n_reg
        term_inv = np.linalg.pinv(matrix_to_invert, rcond=1e-8)
        W_ell[l] = (C_s * A_wf.T) @ term_inv
    rad_per_pix = (5 * np.pi / 180) / 256
    lx = 2 * np.pi * np.fft.fftfreq(256, d=rad_per_pix)
    ly = 2 * np.pi * np.fft.fftfreq(256, d=rad_per_pix)
    LX, LY = np.meshgrid(lx, ly)
    L2D = np.sqrt(LX**2 + LY**2)
    W_2D = np.zeros((n_freq, 256, 256))
    for i in range(n_freq):
        interp = interp1d(ell_bins, W_ell[:, i], bounds_error=False, fill_value=(W_ell[0, i], W_ell[-1, i]))
        W_2D[i] = interp(L2D)
    eval_tasks = []
    for p in test_patches:
        i_planck = rng.integers(100)
        i_so = rng.integers(3000)
        eval_tasks.append((p, i_planck, i_so, w_cilc, W_2D))
    with mp.Pool(processes=16, initializer=init_worker) as pool:
        eval_results = pool.map(evaluate_patch, eval_tasks)
    mse_cilc_list = []
    mse_wf_list = []
    cl_res_cilc_mean = np.zeros_like(ell_bins)
    cl_res_wf_mean = np.zeros_like(ell_bins)
    example_true = None
    example_cilc = None
    example_wf = None
    for idx, res in enumerate(eval_results):
        mse_cilc, mse_wf, cl_res_cilc, cl_res_wf, tsz_true, tsz_cilc, tsz_wf = res
        mse_cilc_list.append(mse_cilc)
        mse_wf.append(mse_wf)
        cl_res_cilc_mean += cl_res_cilc
        cl_res_wf_mean += cl_res_wf
        if idx == 0:
            example_true = tsz_true
            example_cilc = tsz_cilc
            example_wf = tsz_wf
    n_test_actual = len(test_patches)
    cl_res_cilc_mean /= n_test_actual
    cl_res_wf_mean /= n_test_actual
    mean_mse_cilc = np.mean(mse_cilc_list)
    mean_mse_wf = np.mean(mse_wf_list)
    timestamp = int(time.time())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = np.percentile(example_true, 1)
    vmax = np.percentile(example_true, 99)
    im0 = axes[0].imshow(example_true, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0].set_title('Ground Truth tSZ')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Compton-y')
    im1 = axes[1].imshow(example_cilc, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1].set_title('cILC Reconstruction')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Compton-y')
    im2 = axes[2].imshow(example_wf, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[2].set_title('Wiener Filter Reconstruction')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Compton-y')
    plt.tight_layout()
    plot1_path = os.path.join(DATA_DIR, 'reconstruction_maps_1_' + str(timestamp) + '.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    _, cl_true = utils.powers(example_true, example_true, ps=PS)
    valid = ell_bins > 0
    ax.plot(ell_bins[valid], cl_true[valid], label='True tSZ', color='black', linestyle='--')
    ax.plot(ell_bins[valid], cl_res_cilc_mean[valid], label='cILC Residuals', color='red')
    ax.plot(ell_bins[valid], cl_res_wf_mean[valid], label='Wiener Filter Residuals', color='blue')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Multipole ell')
    ax.set_ylabel('C_ell')
    ax.set_title('Angular Power Spectra of Reconstruction Residuals')
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plot2_path = os.path.join(DATA_DIR, 'residual_power_spectra_1_' + str(timestamp) + '.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    np.savez(os.path.join(DATA_DIR, 'baseline_metrics.npz'), mse_cilc=mean_mse_cilc, mse_wf=mean_mse_wf, ell=ell_bins, cl_res_cilc=cl_res_cilc_mean, cl_res_wf=cl_res_wf_mean, w_cilc=w_cilc, W_ell=W_ell, tsz_response=tsz_response, ksz_response=ksz_response, jysr2uk_factors=jysr2uk_factors, mean_cib_spectrum=mean_cib_spectrum)