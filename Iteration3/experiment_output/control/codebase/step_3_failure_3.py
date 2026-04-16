# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import utils
from step_2 import SR_DAE
os.environ['OMP_NUM_THREADS'] = '8'
BASE_DIR = '/home/node/data/compsep_data/cut_maps'
DATA_DIR = 'data'
STATS_PATH = os.path.join(DATA_DIR, 'channel_stats.npz')
class CompSepDataset(Dataset):
    def __init__(self, base_dir, stats_path, valid_indices=None, transform=True):
        self.base_dir = base_dir
        self.transform = transform
        self.frequencies = [90, 150, 217, 353, 545, 857]
        self.n_planck = 100
        self.n_so = 3000
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
        if valid_indices is None:
            self.valid_indices = np.arange(1523)
        else:
            self.valid_indices = valid_indices
    def __len__(self):
        return len(self.valid_indices)
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        i_so = np.random.randint(self.n_so)
        i_planck = np.random.randint(self.n_planck)
        x = np.zeros((6, 256, 256), dtype=np.float32)
        for c, freq in enumerate(self.frequencies):
            signal = self.signals[freq][real_idx]
            if freq <= 217:
                noise = self.so_noise[freq][i_so]
            else:
                noise_path = os.path.join(self.base_dir, 'planck_noise', 'planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy')
                raw_noise = np.load(noise_path, mmap_mode='r')[real_idx]
                if freq == 353:
                    noise = raw_noise * 1e6
                else:
                    noise = raw_noise * 1e6 * utils.jysr2uk(freq)
            if self.transform:
                noise_scale = np.random.uniform(0.9, 1.1)
                noise = noise * noise_scale
            x[c] = signal + noise
        y = self.tsz[real_idx].astype(np.float32)
        if self.transform:
            k = np.random.randint(4)
            if k > 0:
                x = np.rot90(x, k, axes=(1, 2))
                y = np.rot90(y, k, axes=(0, 1))
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=0)
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=1)
        x = (x - self.mean_x[:, None, None]) / self.std_x[:, None, None]
        y = (y - self.mean_y) / self.std_y
        y = np.expand_dims(y, axis=0)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
class SpectralLoss(nn.Module):
    def __init__(self):
        super().__init__()
        h, w = 256, 256
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        center_y, center_x = h // 2, w // 2
        r = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
        bins = 199
        r_max = r.max()
        masks = []
        for i in range(bins):
            r_min = i * r_max / bins
            r_max_bin = (i + 1) * r_max / bins
            mask = (r >= r_min) & (r < r_max_bin)
            masks.append(mask.float())
        self.register_buffer('masks', torch.stack(masks))
        self.register_buffer('mask_sums', self.masks.sum(dim=(1,2)).clamp(min=1))
        window_y = torch.hann_window(h).unsqueeze(1)
        window_x = torch.hann_window(w).unsqueeze(0)
        self.register_buffer('window', window_y * window_x)
        self.use_utils = True
    def forward(self, pred, target):
        pred_mean = pred.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)
        if self.use_utils:
            try:
                res_p = utils.powers(pred_mean[0, 0].detach().cpu().numpy(), pred_mean[0, 0].detach().cpu().numpy(), ps=10, ell_n=199, window_alpha=None)
                res_t = utils.powers(target_mean[0, 0].detach().cpu().numpy(), target_mean[0, 0].detach().cpu().numpy(), ps=10, ell_n=199, window_alpha=None)
                cl_p = res_p[1] if isinstance(res_p, tuple) else res_p
                cl_t = res_t[1] if isinstance(res_t, tuple) else res_t
                if not isinstance(cl_p, torch.Tensor):
                    cl_p = torch.tensor(cl_p, device=pred.device, dtype=pred.dtype)
                if not isinstance(cl_t, torch.Tensor):
                    cl_t = torch.tensor(cl_t, device=target.device, dtype=target.dtype)
                return nn.functional.mse_loss(torch.log1p(cl_p), torch.log1p(cl_t))
            except Exception:
                self.use_utils = False
        pred_w = pred_mean * self.window
        target_w = target_mean * self.window
        pred_fft = torch.fft.fftshift(torch.fft.fft2(pred_w), dim=(-2, -1))
        target_fft = torch.fft.fftshift(torch.fft.fft2(target_w), dim=(-2, -1))
        pred_ps = torch.abs(pred_fft)**2
        target_ps = torch.abs(target_fft)**2
        pred_ps = pred_ps.squeeze(1)
        target_ps = target_ps.squeeze(1)
        radial_pred = torch.einsum('bhw,chw->bc', pred_ps, self.masks) / self.mask_sums
        radial_target = torch.einsum('bhw,chw->bc', target_ps, self.masks) / self.mask_sums
        return nn.functional.mse_loss(torch.log1p(radial_pred), torch.log1p(radial_target))
class CompositeLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.1, lambda3=0.1):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.l1_loss = nn.L1Loss()
        self.register_buffer('weight_x', torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]]))
        self.register_buffer('weight_y', torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]]))
        self.spectral_loss_fn = SpectralLoss()
    def feature_matching_loss(self, pred, target):
        weight_x = self.weight_x.to(dtype=pred.dtype)
        weight_y = self.weight_y.to(dtype=pred.dtype)
        pred_dx = nn.functional.conv2d(pred, weight_x, padding=1)
        pred_dy = nn.functional.conv2d(pred, weight_y, padding=1)
        target_dx = nn.functional.conv2d(target, weight_x, padding=1)
        target_dy = nn.functional.conv2d(target, weight_y, padding=1)
        return self.l1_loss(pred_dx, target_dx) + self.l1_loss(pred_dy, target_dy)
    def forward(self, pred, target, compute_spectral=True):
        l1 = self.l1_loss(pred, target)
        fm = self.feature_matching_loss(pred, target)
        loss = self.lambda1 * l1 + self.lambda2 * fm
        if compute_spectral:
            subset_size = min(4, pred.shape[0])
            spec = self.spectral_loss_fn(pred[:subset_size], target[:subset_size])
            loss = loss + self.lambda3 * spec
            return loss, l1, fm, spec
        else:
            return loss, l1, fm, torch.tensor(0.0, device=pred.device)
if __name__ == '__main__':
    print('Initializing training pipeline...')
    tsz_path = os.path.join(BASE_DIR, 'tsz.npy')
    print('Loading tSZ maps to compute curriculum thresholds...')
    tsz_data = np.load(tsz_path, mmap_mode='r')
    max_tsz = np.array([np.max(tsz_data[i]) for i in range(1523)])
    threshold = np.percentile(max_tsz, 75)
    high_intensity_indices = np.where(max_tsz > threshold)[0]
    all_indices = np.arange(1523)
    print('Curriculum Learning: Stage 1 threshold (75th percentile max tSZ) = ' + str(round(threshold, 6)))
    print('Stage 1 patches: ' + str(len(high_intensity_indices)))
    print('Stage 2 patches: ' + str(len(all_indices)))
    epochs = 30
    stage1_epochs = int(0.3 * epochs)
    batch_size = 16
    dataset_stage1 = CompSepDataset(BASE_DIR, STATS_PATH, valid_indices=high_intensity_indices, transform=True)
    dataset_stage2 = CompSepDataset(BASE_DIR, STATS_PATH, valid_indices=all_indices, transform=True)
    loader_stage1 = DataLoader(dataset_stage1, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    loader_stage2 = DataLoader(dataset_stage2, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    model = SR_DAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = len(loader_stage1) * stage1_epochs + len(loader_stage2) * (epochs - stage1_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps)
    criterion = CompositeLoss(lambda1=1.0, lambda2=0.1, lambda3=0.1).to(device)
    scaler = torch.cuda.amp.GradScaler()
    loss_history = []
    print('Starting training...')
    for epoch in range(epochs):
        if epoch < stage1_epochs:
            loader = loader_stage1
            stage_name = 'Stage 1'
        else:
            loader = loader_stage2
            stage_name = 'Stage 2'
        model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_fm = 0.0
        epoch_spec = 0.0
        spec_count = 0
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_primary = x[:, :3, :, :]
            x_auxiliary = x[:, 3:, :, :]
            optimizer.zero_grad(set_to_none=True)
            compute_spectral = (batch_idx % 10 == 0)
            with torch.cuda.amp.autocast():
                pred = model(x_primary, x_auxiliary)
                loss, l1, fm, spec = criterion(pred, y, compute_spectral=compute_spectral)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            epoch_l1 += l1.item()
            epoch_fm += fm.item()
            if compute_spectral:
                epoch_spec += spec.item()
                spec_count += 1
        avg_loss = epoch_loss / len(loader)
        avg_l1 = epoch_l1 / len(loader)
        avg_fm = epoch_fm / len(loader)
        avg_spec = epoch_spec / max(1, spec_count)
        loss_history.append(avg_loss)
        print('Epoch ' + str(epoch+1) + '/' + str(epochs) + ' [' + stage_name + '] - Loss: ' + str(round(avg_loss, 4)) + ' | L1: ' + str(round(avg_l1, 4)) + ' | FM: ' + str(round(avg_fm, 4)) + ' | Spec: ' + str(round(avg_spec, 4)))
    model_save_path = os.path.join(DATA_DIR, 'sr_dae_model.pth')
    history_save_path = os.path.join(DATA_DIR, 'loss_history.npy')
    torch.save(model.state_dict(), model_save_path)
    np.save(history_save_path, np.array(loss_history))
    print('Training complete. Model saved to ' + model_save_path)
    print('Loss history saved to ' + history_save_path)