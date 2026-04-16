# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['OMP_NUM_THREADS'] = '8'
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.abspath('codebase'))
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
from step_2 import SR_DAE
BASE_DIR = '/home/node/data/compsep_data/cut_maps'
DATA_DIR = 'data'
STATS_PATH = os.path.join(DATA_DIR, 'channel_stats.npz')
MODEL_PATH = os.path.join(DATA_DIR, 'sr_dae_model.pth')
class InferenceDataset(Dataset):
    def __init__(self, base_dir, stats_path, shuffle_cib=False):
        self.base_dir = base_dir
        self.frequencies = [90, 150, 217, 353, 545, 857]
        stats = np.load(stats_path)
        self.mean_x = stats['mean_x']
        self.std_x = stats['std_x']
        self.mean_y = stats['mean_y']
        self.std_y = stats['std_y']
        self.signals = {}
        for freq in self.frequencies:
            self.signals[freq] = np.load(os.path.join(base_dir, 'stacked_' + str(freq) + '.npy'), mmap_mode='r')
        self.tsz = np.load(os.path.join(base_dir, 'tsz.npy'), mmap_mode='r')
        self.so_noise = {}
        for freq in [90, 150, 217]:
            self.so_noise[freq] = np.load(os.path.join(base_dir, 'so_noise', str(freq) + '.npy'), mmap_mode='r')
        self.planck_noise = {}
        for freq in [353, 545, 857]:
            noise_path = os.path.join(base_dir, 'planck_noise', 'planck_noise_' + str(freq) + '_0.npy')
            self.planck_noise[freq] = np.load(noise_path, mmap_mode='r')
        self.shuffle_cib = shuffle_cib
        if shuffle_cib:
            np.random.seed(42)
            self.cib_indices = np.random.permutation(1523)
        else:
            self.cib_indices = np.arange(1523)
    def __len__(self):
        return 1523
    def __getitem__(self, idx):
        i_so = 0
        x = np.zeros((6, 256, 256), dtype=np.float32)
        for c, freq in enumerate(self.frequencies):
            if freq >= 353:
                patch_idx = self.cib_indices[idx]
            else:
                patch_idx = idx
            signal = self.signals[freq][patch_idx]
            if freq <= 217:
                noise = self.so_noise[freq][i_so]
            else:
                raw_noise = self.planck_noise[freq][patch_idx]
                if freq == 353:
                    noise = raw_noise * 1e6
                else:
                    noise = raw_noise * 1e6 * utils.jysr2uk(freq)
            x[c] = signal + noise
        y = self.tsz[idx].astype(np.float32)
        x = (x - self.mean_x[:, None, None]) / self.std_x[:, None, None]
        y = (y - self.mean_y) / self.std_y
        y = np.expand_dims(y, axis=0)
        return torch.from_numpy(x), torch.from_numpy(y), idx
def run_inference(model, dataloader, device, std_y, mean_y):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y, idx in dataloader:
            x = x.to(device, non_blocking=True)
            x_primary = x[:, :3, :, :]
            x_auxiliary = x[:, 3:, :, :]
            with torch.cuda.amp.autocast():
                pred = model(x_primary, x_auxiliary)
            pred = pred.float().cpu().numpy() * std_y + mean_y
            y = y.numpy() * std_y + mean_y
            preds.append(pred)
            targets.append(y)
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    return preds, targets
def compute_r_ell(preds, targets):
    cl_pred_pred = []
    cl_true_true = []
    cl_pred_true = []
    ell = None
    for i in range(preds.shape[0]):
        p = preds[i, 0]
        t = targets[i, 0]
        try:
            res_pp = utils.powers(p, p, ps=5, ell_n=199, window_alpha=None)
            res_tt = utils.powers(t, t, ps=5, ell_n=199, window_alpha=None)
            res_pt = utils.powers(p, t, ps=5, ell_n=199, window_alpha=None)
            if ell is None:
                ell = res_pp[0] if isinstance(res_pp, tuple) else np.arange(len(res_pp))
            pp = res_pp[1] if isinstance(res_pp, tuple) else res_pp
            tt = res_tt[1] if isinstance(res_tt, tuple) else res_tt
            pt = res_pt[1] if isinstance(res_pt, tuple) else res_pt
            cl_pred_pred.append(pp)
            cl_true_true.append(tt)
            cl_pred_true.append(pt)
        except Exception as e:
            pass
    cl_pred_pred = np.mean(cl_pred_pred, axis=0)
    cl_true_true = np.mean(cl_true_true, axis=0)
    cl_pred_true = np.mean(cl_pred_true, axis=0)
    denom = np.sqrt(cl_pred_pred * cl_true_true)
    r_ell = np.divide(cl_pred_true, denom, out=np.zeros_like(cl_pred_true), where=denom > 0)
    return ell, r_ell
if __name__ == '__main__':
    stats = np.load(STATS_PATH)
    mean_y = stats['mean_y']
    std_y = stats['std_y']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SR_DAE().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    dataset_std = InferenceDataset(BASE_DIR, STATS_PATH, shuffle_cib=False)
    loader_std = DataLoader(dataset_std, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    preds_std, targets_std = run_inference(model, loader_std, device, std_y, mean_y)
    dataset_null = InferenceDataset(BASE_DIR, STATS_PATH, shuffle_cib=True)
    loader_null = DataLoader(dataset_null, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    preds_null, targets_null = run_inference(model, loader_null, device, std_y, mean_y)
    ell, r_ell_std = compute_r_ell(preds_std, targets_std)
    _, r_ell_null = compute_r_ell(preds_null, targets_null)
    mass_proxy = np.sum(targets_std[:, 0], axis=(1, 2))
    y_sz_pred_std = np.sum(preds_std[:, 0], axis=(1, 2))
    y_sz_pred_null = np.sum(preds_null[:, 0], axis=(1, 2))
    raw_150 = np.load(os.path.join(BASE_DIR, 'stacked_150.npy'))
    so_noise_150 = np.load(os.path.join(BASE_DIR, 'so_noise', '150.npy'))
    raw_150_total = raw_150 + so_noise_150[0]
    y_sz_raw = np.sum(raw_150_total, axis=(1, 2)) / utils.tsz(150)
    num_bins = 10
    bins = np.percentile(mass_proxy, np.linspace(0, 100, num_bins + 1))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    scatter_std, scatter_null, scatter_raw = [], [], []
    for i in range(num_bins):
        mask = (mass_proxy >= bins[i]) & (mass_proxy <= bins[i+1])
        if np.sum(mask) == 0:
            scatter_std.append(0); scatter_null.append(0); scatter_raw.append(0)
            continue
        scatter_std.append(np.std(y_sz_pred_std[mask] - mass_proxy[mask]))
        scatter_null.append(np.std(y_sz_pred_null[mask] - mass_proxy[mask]))
        scatter_raw.append(np.std(y_sz_raw[mask] - mass_proxy[mask]))
    np.savez(os.path.join(DATA_DIR, 'validation_metrics.npz'), ell=ell, r_ell_std=r_ell_std, r_ell_null=r_ell_null, mass_proxy=mass_proxy, y_sz_pred_std=y_sz_pred_std, y_sz_pred_null=y_sz_pred_null, y_sz_raw=y_sz_raw, bin_centers=bin_centers, scatter_std=scatter_std, scatter_null=scatter_null, scatter_raw=scatter_raw)
    np.save(os.path.join(DATA_DIR, 'preds_std_subset.npy'), preds_std[:50])
    np.save(os.path.join(DATA_DIR, 'targets_subset.npy'), targets_std[:50])
    np.save(os.path.join(DATA_DIR, 'preds_null_subset.npy'), preds_null[:50])