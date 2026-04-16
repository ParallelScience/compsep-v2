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
class DataLoader:
    def __init__(self, base_dir='/home/node/data/compsep_data/cut_maps'):
        self.base_dir = base_dir
        self.frequencies = [90, 150, 217, 353, 545, 857]
    def load_signal(self, freq, i_patch):
        return np.load(self.base_dir + '/stacked_' + str(freq) + '.npy')[i_patch]
    def load_so_noise(self, freq, i_so):
        return np.load(self.base_dir + '/so_noise/' + str(freq) + '.npy')[i_so]
    def load_planck_noise(self, freq, i_patch, i_planck):
        raw = np.load(self.base_dir + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy')[i_patch]
        if freq == 353:
            return raw * 1e6
        else:
            return raw * 1e6 * utils.jysr2uk(freq)
    def load_observed(self, i_patch, i_so, i_planck):
        patches = {}
        for freq in self.frequencies:
            signal = self.load_signal(freq, i_patch)
            if freq <= 217:
                noise = self.load_so_noise(freq, i_so)
            else:
                noise = self.load_planck_noise(freq, i_patch, i_planck)
            patches[freq] = signal + noise
        return patches
    def load_ground_truth(self, i_patch):
        cmb = np.load(self.base_dir + '/lensed_cmb.npy')[i_patch]
        tsz = np.load(self.base_dir + '/tsz.npy')[i_patch]
        ksz = np.load(self.base_dir + '/ksz.npy')[i_patch]
        cib90 = np.load(self.base_dir + '/cib_90.npy')[i_patch]
        cib150 = np.load(self.base_dir + '/cib_150.npy')[i_patch]
        cib217 = np.load(self.base_dir + '/cib_217.npy')[i_patch]
        cib353 = np.load(self.base_dir + '/cib_353.npy')[i_patch]
        cib545 = np.load(self.base_dir + '/cib_545.npy')[i_patch]
        cib857 = np.load(self.base_dir + '/cib_857.npy')[i_patch]
        return {'cmb': cmb, 'tsz': tsz, 'ksz': ksz, 'cib_90': cib90, 'cib_150': cib150, 'cib_217': cib217, 'cib_353': cib353, 'cib_545': cib545, 'cib_857': cib857}
def apply_cilc(patches_dict, W):
    fft_n = np.stack([np.fft.fft2(patches_dict[90], norm='ortho'), np.fft.fft2(patches_dict[150], norm='ortho'), np.fft.fft2(patches_dict[217], norm='ortho')], axis=-1)
    cilc_fft = np.sum(W * fft_n, axis=-1)
    return np.real(np.fft.ifft2(cilc_fft, norm='ortho'))
if __name__ == '__main__':
    print('Initializing DataLoader...')
    loader = DataLoader()
    print('Computing empirical noise covariance matrices for SO LAT bands...')
    base_dir = '/home/node/data/compsep_data/cut_maps'
    freqs = [90, 150, 217]
    noise_maps = {}
    for f in freqs:
        noise_maps[f] = np.load(base_dir + '/so_noise/' + str(f) + '.npy')
    fft_maps = {}
    for f in freqs:
        fft_maps[f] = np.fft.fft2(noise_maps[f], norm='ortho')
    cov = np.zeros((256, 256, 3, 3), dtype=np.float64)
    for i, f1 in enumerate(freqs):
        for j, f2 in enumerate(freqs):
            cross = np.mean(fft_maps[f1] * np.conj(fft_maps[f2]), axis=0)
            cov[:, :, i, j] = np.real(cross)
    print('Covariance matrix computed. Shape: ' + str(cov.shape))
    print('  Max variance 90GHz: ' + str(np.max(cov[:,:,0,0])))
    print('  Max variance 150GHz: ' + str(np.max(cov[:,:,1,1])))
    print('  Max variance 217GHz: ' + str(np.max(cov[:,:,2,2])))
    print('\nConstructing cILC weights...')
    a_tsz = np.array([utils.tsz(f) for f in freqs])
    a_ksz = np.array([utils.ksz(f) for f in freqs])
    print('  tSZ spectral response: ' + str(a_tsz))
    print('  kSZ spectral response: ' + str(a_ksz))
    A = np.column_stack((a_tsz, a_ksz))
    e = np.array([1.0, 0.0])
    trace_cov = np.trace(cov, axis1=2, axis2=3)
    reg = np.zeros_like(cov)
    reg[:, :, 0, 0] = trace_cov * 1e-4
    reg[:, :, 1, 1] = trace_cov * 1e-4
    reg[:, :, 2, 2] = trace_cov * 1e-4
    cov_reg = cov + reg
    inv_cov = np.linalg.inv(cov_reg)
    inv_cov_A = np.einsum('...ij,jk->...ik', inv_cov, A)
    At_inv_cov_A = np.einsum('ji,...jk->...ik', A, inv_cov_A)
    inv_At_inv_cov_A = np.linalg.inv(At_inv_cov_A)
    term2 = np.einsum('...ij,j->...i', inv_At_inv_cov_A, e)
    W = np.einsum('...ij,...j->...i', inv_cov_A, term2)
    print('cILC weights computed. Shape: ' + str(W.shape))
    print('\nVerifying constraints...')
    w_tsz = np.einsum('...i,i->...', W, A[:, 0])
    w_ksz = np.einsum('...i,i->...', W, A[:, 1])
    a_cmb = np.array([1.0, 1.0, 1.0])
    w_cmb = np.einsum('...i,i->...', W, a_cmb)
    print('  Max absolute error for tSZ preservation (W^T a_tsz = 1): ' + str(np.max(np.abs(w_tsz - 1.0))))
    print('  Max absolute error for kSZ nulling (W^T a_ksz = 0): ' + str(np.max(np.abs(w_ksz - 0.0))))
    print('  Max absolute error for CMB nulling (W^T a_cmb = 0): ' + str(np.max(np.abs(w_cmb - 0.0))))
    print('\nSaving weights and covariance matrix...')
    np.save('data/cilc_weights.npy', W)
    np.save('data/noise_cov.npy', cov)
    print('Saved to data/cilc_weights.npy and data/noise_cov.npy')
    print('\nTesting cILC on the first noise realization...')
    n90 = noise_maps[90][0]
    n150 = noise_maps[150][0]
    n217 = noise_maps[217][0]
    fft_n = np.stack([np.fft.fft2(n90, norm='ortho'), np.fft.fft2(n150, norm='ortho'), np.fft.fft2(n217, norm='ortho')], axis=-1)
    cilc_fft = np.sum(W * fft_n, axis=-1)
    cilc_map = np.real(np.fft.ifft2(cilc_fft, norm='ortho'))
    print('  Original noise std: 90GHz=' + str(np.std(n90)) + ', 150GHz=' + str(np.std(n150)) + ', 217GHz=' + str(np.std(n217)))
    print('  cILC noise std: ' + str(np.std(cilc_map)))
    print('Step 1 completed successfully.')