# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['OMP_NUM_THREADS'] = '16'
import numpy as np
import sys
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
def compute_cilc():
    BASE = '/home/node/data/compsep_data/cut_maps'
    freqs = [90, 150, 217]
    n_patch = 1523
    n_so = 3000
    batch_size = 100
    print('Loading SO noise maps...')
    so_noise = {f: np.load(BASE + '/so_noise/' + str(f) + '.npy') for f in freqs}
    print('Computing noise covariance matrix N(k)...')
    N_k = np.zeros((3, 3, 256, 129), dtype=np.complex128)
    for i in range(0, n_so, batch_size):
        n_batch = np.array([np.fft.rfft2(so_noise[f][i:i+batch_size]) for f in freqs])
        N_k += np.einsum('ibxy, jbxy -> ijxy', n_batch, n_batch.conj())
    N_k /= n_so
    print('Loading signal maps and adding noise...')
    rng = np.random.default_rng(seed=42)
    so_indices = rng.integers(0, n_so, size=n_patch)
    stacked = {f: np.load(BASE + '/stacked_' + str(f) + '.npy') for f in freqs}
    data_k = np.zeros((n_patch, 3, 256, 129), dtype=np.complex128)
    for p in range(n_patch):
        for idx_f, f in enumerate(freqs):
            signal = stacked[f][p]
            noise = so_noise[f][so_indices[p]]
            data_k[p, idx_f] = np.fft.rfft2(signal + noise)
    print('Computing global data covariance matrix C_global(k)...')
    C_global = np.zeros((3, 3, 256, 129), dtype=np.complex128)
    for i in range(0, n_patch, batch_size):
        d_batch = data_k[i:i+batch_size]
        C_global += np.einsum('bixy, bjxy -> ijxy', d_batch, d_batch.conj())
    C_global /= n_patch
    print('Setting up cILC constraints...')
    A = np.zeros((3, 2))
    A[0, 0] = utils.tsz(90)
    A[1, 0] = utils.tsz(150)
    A[2, 0] = utils.tsz(217)
    A[:, 1] = 1.0
    print('Computing per-patch cILC weights and y-maps...')
    cilc_weights = np.zeros((n_patch, 3, 256, 129), dtype=np.complex128)
    cilc_y_maps = np.zeros((n_patch, 256, 256), dtype=np.float64)
    A_T = A.T
    e1 = np.array([[1.0], [0.0]])
    for i in range(0, n_patch, batch_size):
        d_batch = data_k[i:i+batch_size]
        b_size = d_batch.shape[0]
        C_p = np.einsum('bixy, bjxy -> bijxy', d_batch, d_batch.conj())
        C_p += C_global[None, ...] + N_k[None, ...]
        C_p_reshaped = C_p.transpose(0, 3, 4, 1, 2).reshape(-1, 3, 3)
        C_inv = np.linalg.inv(C_p_reshaped)
        C_inv_A = np.matmul(C_inv, A)
        A_T_C_inv_A = np.matmul(A_T, C_inv_A)
        inv_term = np.linalg.inv(A_T_C_inv_A)
        right_term = np.matmul(inv_term, e1)
        W = np.matmul(C_inv_A, right_term)
        W = W[..., 0]
        W_batch = W.reshape(b_size, 256, 129, 3).transpose(0, 3, 1, 2)
        cilc_weights[i:i+batch_size] = W_batch
        y_k_batch = np.sum(W_batch.conj() * d_batch, axis=1)
        y_map_batch = np.fft.irfft2(y_k_batch, s=(256, 256))
        cilc_y_maps[i:i+batch_size] = y_map_batch
    print('Saving results...')
    np.save('data/cilc_weights.npy', cilc_weights)
    np.save('data/noise_cov.npy', N_k)
    np.save('data/cilc_y_maps.npy', cilc_y_maps)
    print('cILC computation complete.')
    print('Saved cilc_weights.npy: shape ' + str(cilc_weights.shape))
    print('Saved noise_cov.npy: shape ' + str(N_k.shape))
    print('Saved cilc_y_maps.npy: shape ' + str(cilc_y_maps.shape))
    print('cILC y-maps stats - Mean: ' + str(np.mean(cilc_y_maps)) + ', Std: ' + str(np.std(cilc_y_maps)))
    print('cILC y-maps stats - Min: ' + str(np.min(cilc_y_maps)) + ', Max: ' + str(np.max(cilc_y_maps)))
    print('\nLoading ground truth tSZ maps for evaluation...')
    tsz_gt = np.load(BASE + '/tsz.npy')
    y_mean = np.mean(cilc_y_maps)
    t_mean = np.mean(tsz_gt)
    y_std = np.std(cilc_y_maps)
    t_std = np.std(tsz_gt)
    cov = np.mean((cilc_y_maps - y_mean) * (tsz_gt - t_mean))
    corr = cov / (y_std * t_std)
    print('Global pixel-wise Pearson correlation between cILC y-maps and ground truth tSZ: ' + str(corr))
    corrs = []
    for p in range(n_patch):
        c = np.corrcoef(cilc_y_maps[p].flatten(), tsz_gt[p].flatten())[0, 1]
        corrs.append(c)
    corrs = np.array(corrs)
    print('Per-patch Pearson correlation stats - Mean: ' + str(np.mean(corrs)) + ', Std: ' + str(np.std(corrs)) + ', Min: ' + str(np.min(corrs)) + ', Max: ' + str(np.max(corrs)))
    rmse = np.sqrt(np.mean((cilc_y_maps - tsz_gt)**2))
    print('Global RMSE between cILC y-maps and ground truth tSZ: ' + str(rmse))
    print('Ground truth tSZ stats - Mean: ' + str(np.mean(tsz_gt)) + ', Std: ' + str(np.std(tsz_gt)))
    print('Ground truth tSZ stats - Min: ' + str(np.min(tsz_gt)) + ', Max: ' + str(np.max(tsz_gt)))
    print('\nComputing average power spectra...')
    ell_n = 199
    ps_y = np.zeros(ell_n)
    ps_tsz = np.zeros(ell_n)
    ps_cross = np.zeros(ell_n)
    for p in range(n_patch):
        ell, py = utils.powers(cilc_y_maps[p], cilc_y_maps[p])
        _, pt = utils.powers(tsz_gt[p], tsz_gt[p])
        _, px = utils.powers(cilc_y_maps[p], tsz_gt[p])
        ps_y += py
        ps_tsz += pt
        ps_cross += px
    ps_y /= n_patch
    ps_tsz /= n_patch
    ps_cross /= n_patch
    r_ell = ps_cross / np.sqrt(ps_y * ps_tsz)
    print('Scale-dependent cross-correlation r_ell (binned):')
    bin_size = len(ell) // 10
    for i in range(10):
        start = i * bin_size
        end = (i + 1) * bin_size if i < 9 else len(ell)
        ell_mean = np.mean(ell[start:end])
        r_ell_mean = np.mean(r_ell[start:end])
        print('  ell ~ ' + str(ell_mean) + ': r_ell = ' + str(r_ell_mean))
if __name__ == '__main__':
    compute_cilc()