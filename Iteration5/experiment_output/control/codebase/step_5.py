# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
import multiprocessing as mp

plt.rcParams['text.usetex'] = False

def init_worker():
    os.environ['OMP_NUM_THREADS'] = '1'

def process_train_patch(args):
    sys.path.insert(0, '/home/node/data/compsep_data')
    import utils
    import numpy as np
    p, i_planck, i_so, BASE, frequencies, n_freq, PS = args
    obs = np.zeros((n_freq, 256, 256))
    cib_mean = np.zeros(n_freq)
    for i, freq in enumerate(frequencies):
        signal = np.load(BASE + '/stacked_' + str(freq) + '.npy', mmap_mode='r')[p]
        if freq <= 217:
            noise = np.load(BASE + '/so_noise/' + str(freq) + '.npy', mmap_mode='r')[i_so]
        else:
            raw = np.load(BASE + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy', mmap_mode='r')[p]
            if freq == 353:
                noise = raw * 1e6
            else:
                noise = raw * 1e6 * utils.jysr2uk(freq)
        obs[i] = signal + noise
        if freq >= 353:
            cib = np.load(BASE + '/cib_' + str(freq) + '.npy', mmap_mode='r')[p]
            cib_mean[i] = np.mean(cib) * utils.jysr2uk(freq)
    tsz = np.load(BASE + '/tsz.npy', mmap_mode='r')[p]
    obs_flat = obs.reshape(n_freq, -1)
    sum_xxT = obs_flat @ obs_flat.T
    sum_x = np.sum(obs_flat, axis=1)
    cl_tsz, ell = utils.powers(tsz, tsz, ps=PS)
    tsz_response = np.array([utils.tsz(f) for f in frequencies])
    noise_obs = obs - tsz_response[:, None, None] * tsz
    n_ell = len(ell)
    cl_noise = np.zeros((n_ell, n_freq, n_freq))
    for i in range(n_freq):
        for j in range(i, n_freq):
            cl_n, _ = utils.powers(noise_obs[i], noise_obs[j], ps=PS)
            cl_noise[:, i, j] = cl_n
            if i != j:
                cl_noise[:, j, i] = cl_n
    return sum_xxT, sum_x, cl_tsz, cl_noise, ell, cib_mean

def evaluate_test_patch(args):
    sys.path.insert(0, '/home/node/data/compsep_data')
    import utils
    import numpy as np
    p, i_planck, i_so, w_cilc, W_2D, BASE, frequencies, n_freq, PS = args
    obs = np.zeros((n_freq, 256, 256))
    for i, freq in enumerate(frequencies):
        signal = np.load(BASE + '/stacked_' + str(freq) + '.npy', mmap_mode='r')[p]
        if freq <= 217:
            noise = np.load(BASE + '/so_noise/' + str(freq) + '.npy', mmap_mode='r')[i_so]
        else:
            raw = np.load(BASE + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy', mmap_mode='r')[p]
            if freq == 353:
                noise = raw * 1e6
            else:
                noise = raw * 1e6 * utils.jysr2uk(freq)
        obs[i] = signal + noise
    tsz_true = np.load(BASE + '/tsz.npy', mmap_mode='r')[p]
    tsz_cilc = np.tensordot(w_cilc, obs, axes=(0, 0))
    obs_fft = np.fft.fft2(obs, axes=(1, 2))
    tsz_wf_fft = np.sum(W_2D * obs_fft, axis=0)
    tsz_wf = np.real(np.fft.ifft2(tsz_wf_fft))
    cl_true, ell = utils.powers(tsz_true, tsz_true, ps=PS)
    cl_cilc_cross, _ = utils.powers(tsz_cilc, tsz_true, ps=PS)
    cl_wf_cross, _ = utils.powers(tsz_wf, tsz_true, ps=PS)
    cl_res_cilc, _ = utils.powers(tsz_cilc - tsz_true, tsz_cilc - tsz_true, ps=PS)
    cl_res_wf, _ = utils.powers(tsz_wf - tsz_true, tsz_wf - tsz_true, ps=PS)
    y_true = np.sum(tsz_true)
    y_cilc = np.sum(tsz_cilc)
    y_wf = np.sum(tsz_wf)
    return cl_true, cl_cilc_cross, cl_wf_cross, cl_res_cilc, cl_res_wf, y_true, y_cilc, y_wf, ell

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath('codebase'))
    sys.path.insert(0, '/home/node/data/compsep_data')
    import utils
    BASE = '/home/node/data/compsep_data/cut_maps'
    DATA_DIR = 'data'
    PS = 5 * 60 / 256
    frequencies = [90, 150, 217, 353, 545, 857]
    n_freq = len(frequencies)
    splits = np.load(os.path.join(DATA_DIR, 'splits.npz'))
    train_idx = splits['train_idx']
    cdm_data = np.load(os.path.join(DATA_DIR, 'cdm_results.npz'))
    cdm_means = cdm_data['cdm_means']
    cdm_vars = cdm_data['cdm_vars']
    true_tsz = cdm_data['true_tsz']
    peak_mass_proxy = cdm_data['peak_mass_proxy']
    test_idx = cdm_data['test_idx']
    dae_data = np.load(os.path.join(DATA_DIR, 'dae_results.npz'))
    dae_preds = dae_data['dae_preds']
    rng = np.random.default_rng(seed=42)
    tasks_train = [(p, rng.integers(100), rng.integers(3000), BASE, frequencies, n_freq, PS) for p in train_idx]
    with mp.Pool(processes=16, initializer=init_worker) as pool:
        results = pool.map(process_train_patch, tasks_train)
    sum_xxT_total = np.zeros((n_freq, n_freq))
    sum_x_total = np.zeros(n_freq)
    cl_tsz_total = None
    cl_noise_total = None
    ell_bins = None
    mean_cib_spectrum = np.zeros(n_freq)
    n_train_actual = len(tasks_train)
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
    mean_x = sum_x_total / (n_train_actual * 256 * 256)
    C_data = (sum_xxT_total / (n_train_actual * 256 * 256)) - np.outer(mean_x, mean_x)
    cl_tsz_mean = cl_tsz_total / n_train_actual
    cl_noise_mean = cl_noise_total / n_train_actual
    mean_cib_spectrum /= n_train_actual
    tsz_response = np.array([utils.tsz(f) for f in frequencies])
    scale_factor = 1e-6
    tsz_scaled = tsz_response * scale_factor
    A = np.vstack([tsz_scaled, mean_cib_spectrum]).T
    F = np.array([1.0, 0.0])
    C_inv = np.linalg.pinv(C_data, rcond=1e-8)
    w_cilc = (C_inv @ A @ np.linalg.pinv(A.T @ C_inv @ A, rcond=1e-8) @ F) * scale_factor
    W_ell = np.zeros((len(ell_bins), n_freq))
    A_wf = tsz_response.reshape(-1, 1)
    for l in range(len(ell_bins)):
        C_s = cl_tsz_mean[l]
        C_n = cl_noise_mean[l] + np.eye(n_freq) * np.max(np.diag(cl_noise_mean[l])) * 1e-4
        W_ell[l] = (C_s * A_wf.T) @ np.linalg.pinv(A_wf @ np.array([[C_s]]) @ A_wf.T + C_n, rcond=1e-8)
    W_2D = np.zeros((n_freq, 256, 256))
    rad_per_pix = (5 * np.pi / 180) / 256
    lx = 2 * np.pi * np.fft.fftfreq(256, d=rad_per_pix)
    ly = 2 * np.pi * np.fft.fftfreq(256, d=rad_per_pix)
    L2D = np.sqrt(np.meshgrid(lx, lx)[0]**2 + np.meshgrid(ly, ly)[1]**2)
    for i in range(n_freq):
        W_2D[i] = interp1d(ell_bins, W_ell[:, i], bounds_error=False, fill_value=(W_ell[0, i], W_ell[-1, i]))(L2D)
    tasks_test = [(p, rng.integers(100), rng.integers(3000), w_cilc, W_2D, BASE, frequencies, n_freq, PS) for p in test_idx]
    with mp.Pool(processes=16, initializer=init_worker) as pool:
        test_results = pool.map(evaluate_test_patch, tasks_test)
    cl_true_all, cl_cilc_cross_all, cl_wf_cross_all, cl_res_cilc_all, cl_res_wf_all, y_true_all, y_cilc_all, y_wf_all = [], [], [], [], [], [], [], []
    for res in test_results:
        cl_true, cl_cilc_cross, cl_wf_cross, cl_res_cilc, cl_res_wf, y_true_val, y_cilc_val, y_wf_val, ell = res
        cl_true_all.append(cl_true)
        cl_cilc_cross_all.append(cl_cilc_cross)
        cl_wf_cross_all.append(cl_wf_cross)
        cl_res_cilc_all.append(cl_res_cilc)
        cl_res_wf_all.append(cl_res_wf)
        y_true_all.append(y_true_val)
        y_cilc_all.append(y_cilc_val)
        y_wf_all.append(y_wf_val)
    cl_true_mean = np.mean(cl_true_all, axis=0)
    cl_cilc_cross_mean = np.mean(cl_cilc_cross_all, axis=0)
    cl_wf_cross_mean = np.mean(cl_wf_cross_all, axis=0)
    cl_res_cilc_mean = np.mean(cl_res_cilc_all, axis=0)
    cl_res_wf_mean = np.mean(cl_res_wf_all, axis=0)
    cl_cdm_cross_all, cl_dae_cross_all, cl_res_cdm_all, cl_res_dae_all = [], [], [], []
    for i in range(len(test_idx)):
        true_map = true_tsz[i]
        cdm_map = cdm_means[i]
        dae_map = dae_preds[i]
        cl_cdm_cross, _ = utils.powers(cdm_map, true_map, ps=PS)
        cl_dae_cross, _ = utils.powers(dae_map, true_map, ps=PS)
        cl_res_cdm, _ = utils.powers(cdm_map - true_map, cdm_map - true_map, ps=PS)
        cl_res_dae, _ = utils.powers(dae_map - true_map, dae_map - true_map, ps=PS)
        cl_cdm_cross_all.append(cl_cdm_cross)
        cl_dae_cross_all.append(cl_dae_cross)
        cl_res_cdm_all.append(cl_res_cdm)
        cl_res_dae_all.append(cl_res_dae)
    cl_cdm_cross_mean = np.mean(cl_cdm_cross_all, axis=0)
    cl_dae_cross_mean = np.mean(cl_dae_cross_all, axis=0)
    cl_res_cdm_mean = np.mean(cl_res_cdm_all, axis=0)
    cl_res_dae_mean = np.mean(cl_res_dae_all, axis=0)
    T_cilc = cl_cilc_cross_mean / cl_true_mean
    T_wf = cl_wf_cross_mean / cl_true_mean
    T_cdm = cl_cdm_cross_mean / cl_true_mean
    T_dae = cl_dae_cross_mean / cl_true_mean
    ell_mask = (ell_bins >= 1000) & (ell_bins <= 5000)
    gain_cdm = cl_res_cilc_mean[ell_mask] / cl_res_cdm_mean[ell_mask]
    gain_dae = cl_res_cilc_mean[ell_mask] / cl_res_dae_mean[ell_mask]
    gain_wf = cl_res_cilc_mean[ell_mask] / cl_res_wf_mean[ell_mask]
    y_true_cdm = np.sum(true_tsz, axis=(1, 2))
    y_cdm = np.sum(cdm_means, axis=(1, 2))
    y_dae = np.sum(dae_preds, axis=(1, 2))
    y_cilc_arr = np.array(y_cilc_all)
    y_wf_arr = np.array(y_wf_all)
    y_true_arr = np.array(y_true_all)
    def compute_scatter_bias(y_pred, y_true_vals):
        rel_err = (y_pred - y_true_vals) / np.abs(y_true_vals + 1e-12)
        bias = np.mean(rel_err)
        scatter = np.std(rel_err)
        return bias, scatter
    bias_cdm, scatter_cdm = compute_scatter_bias(y_cdm, y_true_cdm)
    bias_dae, scatter_dae = compute_scatter_bias(y_dae, y_true_cdm)
    bias_cilc, scatter_cilc = compute_scatter_bias(y_cilc_arr, y_true_arr)
    bias_wf, scatter_wf = compute_scatter_bias(y_wf_arr, y_true_arr)
    pit_values = norm.cdf(true_tsz.flatten(), loc=cdm_means.flatten(), scale=np.sqrt(cdm_vars.flatten() + 1e-12))
    pit_mean = np.mean(pit_values)
    pit_std = np.std(pit_values)
    rng_pit = np.random.default_rng(42)
    pit_sample = rng_pit.choice(pit_values, size=min(1000000, len(pit_values)), replace=False)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].hist(pit_sample, bins=50, density=True, alpha=0.7, color='blue')
    axs[0, 0].axhline(1.0, color='k', linestyle='--')
    axs[0, 0].set_xlabel('PIT Value')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].set_title('(a) PIT Histogram (CDM)')
    axs[0, 1].scatter(peak_mass_proxy, y_true_cdm, label='True', alpha=0.6, s=15, color='black')
    axs[0, 1].scatter(peak_mass_proxy, y_cilc_arr, label='cILC', alpha=0.6, s=15, color='red')
    axs[0, 1].scatter(peak_mass_proxy, y_cdm, label='CDM', alpha=0.6, s=15, color='blue')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_yscale('symlog', linthresh=1e-3)
    axs[0, 1].set_xlabel('Peak tSZ (Mass Proxy)')
    axs[0, 1].set_ylabel('Integrated Y_SZ')
    axs[0, 1].set_title('(b) Y_SZ - M Relation')
    axs[0, 1].legend()
    axs[1, 0].plot(ell_bins, T_cilc, label='cILC', color='red')
    axs[1, 0].plot(ell_bins, T_wf, label='WF', color='orange')
    axs[1, 0].plot(ell_bins, T_dae, label='DAE', color='green')
    axs[1, 0].plot(ell_bins, T_cdm, label='CDM', color='blue')
    axs[1, 0].axhline(1.0, color='k', linestyle='--')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xlabel('Multipole ell')
    axs[1, 0].set_ylabel('Transfer Function T(ell)')
    axs[1, 0].set_title('(c) Transfer Function')
    axs[1, 0].legend()
    axs[1, 1].plot(ell_bins[ell_mask], gain_wf, label='WF / cILC', color='orange')
    axs[1, 1].plot(ell_bins[ell_mask], gain_dae, label='DAE / cILC', color='green')
    axs[1, 1].plot(ell_bins[ell_mask], gain_cdm, label='CDM / cILC', color='blue')
    axs[1, 1].axhline(1.0, color='k', linestyle='--')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xlabel('Multipole ell')
    axs[1, 1].set_ylabel('Gain (MSE_cILC / MSE_model)')
    axs[1, 1].set_title('(d) Reconstruction Gain Ratio (ell in [1000, 5000])')
    axs[1, 1].legend()
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = 'validation_results_5_' + str(timestamp) + '.png'
    plot_filepath = os.path.join(DATA_DIR, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print('Plot saved to ' + plot_filepath)
    np.savez(os.path.join(DATA_DIR, 'validation_metrics.npz'),
             bias_cilc=bias_cilc, scatter_cilc=scatter_cilc,
             bias_wf=bias_wf, scatter_wf=scatter_wf,
             bias_dae=bias_dae, scatter_dae=scatter_dae,
             bias_cdm=bias_cdm, scatter_cdm=scatter_cdm,
             mean_gain_wf=np.mean(gain_wf), mean_gain_dae=np.mean(gain_dae), mean_gain_cdm=np.mean(gain_cdm),
             pit_mean=pit_mean, pit_std=pit_std)
    print('Validation metrics saved to ' + os.path.join(DATA_DIR, 'validation_metrics.npz'))