# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
BASE = '/home/node/data/compsep_data/cut_maps'
def get_beam_window(N, L_rad, fwhm_in, fwhm_out):
    if fwhm_out <= fwhm_in:
        return np.ones((N, N))
    fwhm_diff = np.sqrt(fwhm_out**2 - fwhm_in**2)
    sigma_rad = (fwhm_diff / np.sqrt(8 * np.log(2))) * np.pi / (180 * 60)
    kx = np.fft.fftfreq(N, d=L_rad/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L_rad/N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    ell = np.sqrt(KX**2 + KY**2)
    beam = np.exp(-0.5 * ell**2 * sigma_rad**2)
    return beam
def main():
    N = 256
    L_rad = 5.0 * np.pi / 180.0
    beam_150_to_90 = get_beam_window(N, L_rad, 1.4, 2.2)
    beam_217_to_150 = get_beam_window(N, L_rad, 1.0, 1.4)
    beam_217_to_353 = get_beam_window(N, L_rad, 1.0, 4.5)
    beam_cib353 = get_beam_window(N, L_rad, 1.0, 4.5)
    beam_cib545 = get_beam_window(N, L_rad, 1.0, 4.72)
    beam_cib857 = get_beam_window(N, L_rad, 1.0, 4.42)
    so_noise_90 = np.load(BASE + '/so_noise/90.npy')
    so_noise_150 = np.load(BASE + '/so_noise/150.npy')
    so_noise_217 = np.load(BASE + '/so_noise/217.npy')
    stacked_90 = np.load(BASE + '/stacked_90.npy')
    stacked_150 = np.load(BASE + '/stacked_150.npy')
    stacked_217 = np.load(BASE + '/stacked_217.npy')
    stacked_353 = np.load(BASE + '/stacked_353.npy')
    cib_353 = np.load(BASE + '/cib_353.npy') * utils.jysr2uk(353)
    cib_545 = np.load(BASE + '/cib_545.npy') * utils.jysr2uk(545)
    cib_857 = np.load(BASE + '/cib_857.npy') * utils.jysr2uk(857)
    n_patch = 1523
    rng = np.random.default_rng(seed=42)
    i_so_indices = rng.integers(0, 3000, size=n_patch)
    i_planck_indices = rng.integers(0, 100, size=n_patch)
    features = np.zeros((n_patch, 6, N, N), dtype=np.float32)
    local_stds = np.zeros((n_patch, 6))
    patches_by_planck = {}
    for i in range(n_patch):
        p_idx = i_planck_indices[i]
        if p_idx not in patches_by_planck:
            patches_by_planck[p_idx] = []
        patches_by_planck[p_idx].append(i)
    for p_idx, patch_indices in patches_by_planck.items():
        pn_353_full = np.load(BASE + '/planck_noise/planck_noise_353_' + str(p_idx) + '.npy', mmap_mode='r')
        pn_545_full = np.load(BASE + '/planck_noise/planck_noise_545_' + str(p_idx) + '.npy', mmap_mode='r')
        pn_857_full = np.load(BASE + '/planck_noise/planck_noise_857_' + str(p_idx) + '.npy', mmap_mode='r')
        for i in patch_indices:
            pn_353_i = pn_353_full[i].copy() * 1e6
            pn_545_i = pn_545_full[i].copy() * 1e6 * utils.jysr2uk(545)
            pn_857_i = pn_857_full[i].copy() * 1e6 * utils.jysr2uk(857)
            i_so = i_so_indices[i]
            obs_90 = stacked_90[i] + so_noise_90[i_so]
            obs_150 = stacked_150[i] + so_noise_150[i_so]
            obs_217 = stacked_217[i] + so_noise_217[i_so]
            obs_353 = stacked_353[i] + pn_353_i
            fft_150_i = np.fft.fft2(obs_150)
            sm_150 = np.real(np.fft.ifft2(fft_150_i * beam_150_to_90))
            diff_150_90 = sm_150 - obs_90
            fft_217_i = np.fft.fft2(obs_217)
            sm_217_to_150 = np.real(np.fft.ifft2(fft_217_i * beam_217_to_150))
            diff_217_150 = sm_217_to_150 - obs_150
            sm_217_to_353 = np.real(np.fft.ifft2(fft_217_i * beam_217_to_353))
            diff_353_217 = obs_353 - sm_217_to_353
            fft_n150 = np.fft.fft2(so_noise_150[i_so])
            sm_n150 = np.real(np.fft.ifft2(fft_n150 * beam_150_to_90))
            n_diff_150_90 = sm_n150 - so_noise_90[i_so]
            local_std_150_90 = np.std(n_diff_150_90)
            fft_n217 = np.fft.fft2(so_noise_217[i_so])
            sm_n217_to_150 = np.real(np.fft.ifft2(fft_n217 * beam_217_to_150))
            n_diff_217_150 = sm_n217_to_150 - so_noise_150[i_so]
            local_std_217_150 = np.std(n_diff_217_150)
            sm_n217_to_353 = np.real(np.fft.ifft2(fft_n217 * beam_217_to_353))
            n_diff_353_217 = pn_353_i - sm_n217_to_353
            local_std_353_217 = np.std(n_diff_353_217)
            local_std_cib353 = np.std(pn_353_i)
            local_std_cib545 = np.std(pn_545_i)
            local_std_cib857 = np.std(pn_857_i)
            if local_std_150_90 == 0: local_std_150_90 = 1.0
            if local_std_217_150 == 0: local_std_217_150 = 1.0
            if local_std_353_217 == 0: local_std_353_217 = 1.0
            if local_std_cib353 == 0: local_std_cib353 = 1.0
            if local_std_cib545 == 0: local_std_cib545 = 1.0
            if local_std_cib857 == 0: local_std_cib857 = 1.0
            local_stds[i, 0] = local_std_150_90
            local_stds[i, 1] = local_std_217_150
            local_stds[i, 2] = local_std_353_217
            local_stds[i, 3] = local_std_cib353
            local_stds[i, 4] = local_std_cib545
            local_stds[i, 5] = local_std_cib857
            features[i, 0] = diff_150_90 / local_std_150_90
            features[i, 1] = diff_217_150 / local_std_217_150
            features[i, 2] = diff_353_217 / local_std_353_217
            fft_cib353 = np.fft.fft2(cib_353[i])
            cib_aux_353 = np.real(np.fft.ifft2(fft_cib353 * beam_cib353)) + pn_353_i
            fft_cib545 = np.fft.fft2(cib_545[i])
            cib_aux_545 = np.real(np.fft.ifft2(fft_cib545 * beam_cib545)) + pn_545_i
            fft_cib857 = np.fft.fft2(cib_857[i])
            cib_aux_857 = np.real(np.fft.ifft2(fft_cib857 * beam_cib857)) + pn_857_i
            features[i, 3] = cib_aux_353 / local_std_cib353
            features[i, 4] = cib_aux_545 / local_std_cib545
            features[i, 5] = cib_aux_857 / local_std_cib857
    for c in range(3, 6):
        ch_data = features[:, c, :, :]
        median = np.median(ch_data)
        q75, q25 = np.percentile(ch_data, [75, 25])
        iqr = q75 - q25
        if iqr == 0: iqr = 1.0
        features[:, c, :, :] = (ch_data - median) / iqr
    np.save('data/features.npy', features)
    np.savez('data/noise_indices.npz', i_so=i_so_indices, i_planck=i_planck_indices)
if __name__ == '__main__':
    main()