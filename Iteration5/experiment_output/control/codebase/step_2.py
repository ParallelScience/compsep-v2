# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import utils
os.environ['OMP_NUM_THREADS'] = '1'
BASE = '/home/node/data/compsep_data/cut_maps'
DATA_DIR = 'data'
TSZ_SCALE = 1e6
class CompSepDataset(Dataset):
    def __init__(self, patch_indices, base_dir, epoch_size=None):
        self.patch_indices = patch_indices
        self.base_dir = base_dir
        self.epoch_size = epoch_size if epoch_size else len(patch_indices)
        self.frequencies = [90, 150, 217, 353, 545, 857]
        self.stacked = {f: np.load(base_dir + '/stacked_' + str(f) + '.npy', mmap_mode='r') for f in self.frequencies}
        self.so_noise = {f: np.load(base_dir + '/so_noise/' + str(f) + '.npy', mmap_mode='r') for f in [90, 150, 217]}
        self.cib = {f: np.load(base_dir + '/cib_' + str(f) + '.npy', mmap_mode='r') for f in [353, 545, 857]}
        self.tsz = np.load(base_dir + '/tsz.npy', mmap_mode='r')
    def __len__(self):
        return self.epoch_size
    def __getitem__(self, idx):
        p = self.patch_indices[idx % len(self.patch_indices)]
        i_planck = np.random.randint(100)
        i_so = np.random.randint(3000)
        obs = np.zeros((6, 256, 256), dtype=np.float32)
        cib = np.zeros((3, 256, 256), dtype=np.float32)
        for i, freq in enumerate(self.frequencies):
            signal = self.stacked[freq][p]
            if freq <= 217:
                noise = self.so_noise[freq][i_so]
            else:
                raw = np.load(self.base_dir + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy', mmap_mode='r')[p]
                if freq == 353:
                    noise = raw * 1e6
                else:
                    noise = raw * 1e6 * utils.jysr2uk(freq)
            obs[i] = signal + noise
            if freq >= 353:
                cib_idx = i - 3
                cib[cib_idx] = self.cib[freq][p] * utils.jysr2uk(freq)
        tsz = self.tsz[p]
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(cib, dtype=torch.float32), torch.tensor(tsz, dtype=torch.float32).unsqueeze(0)
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True), nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)
class GatedCrossAttention(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.attn = nn.Sequential(nn.Conv2d(in_c * 2, in_c, 1), nn.Sigmoid())
        self.proj = nn.Conv2d(in_c, in_c, 1)
    def forward(self, x_so, x_planck):
        gate = self.attn(torch.cat([x_so, x_planck], dim=1))
        out = self.proj(x_planck) * gate
        return x_so + out
class SR_DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1_so = ConvBlock(3, 64)
        self.enc2_so = ConvBlock(64, 128)
        self.enc3_so = ConvBlock(128, 256)
        self.enc4_so = ConvBlock(256, 512)
        self.enc1_pl = ConvBlock(3, 64)
        self.enc2_pl = ConvBlock(64, 128)
        self.enc3_pl = ConvBlock(128, 256)
        self.enc4_pl = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bot_so = ConvBlock(512, 1024)
        self.bot_pl = ConvBlock(512, 1024)
        self.cross_bot = GatedCrossAttention(1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.cross4 = GatedCrossAttention(512)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.cross3 = GatedCrossAttention(256)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.cross2 = GatedCrossAttention(128)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.cross1 = GatedCrossAttention(64)
        self.dec1 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, 1, 1)
    def forward(self, x_so, x_pl):
        e1_so = self.enc1_so(x_so)
        e1_pl = self.enc1_pl(x_pl)
        e2_so = self.enc2_so(self.pool(e1_so))
        e2_pl = self.enc2_pl(self.pool(e1_pl))
        e3_so = self.enc3_so(self.pool(e2_so))
        e3_pl = self.enc3_pl(self.pool(e2_pl))
        e4_so = self.enc4_so(self.pool(e3_so))
        e4_pl = self.enc4_pl(self.pool(e3_pl))
        b_so = self.bot_so(self.pool(e4_so))
        b_pl = self.bot_pl(self.pool(e4_pl))
        b = self.cross_bot(b_so, b_pl)
        d4 = self.up4(b)
        e4 = self.cross4(e4_so, e4_pl)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        e3 = self.cross3(e3_so, e3_pl)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        e2 = self.cross2(e2_so, e2_pl)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        e1 = self.cross1(e1_so, e1_pl)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)
class PowerSpectrumLoss(nn.Module):
    def __init__(self, mean_cl, ell, ps=5*60/256, device='cuda'):
        super().__init__()
        self.ps = ps
        self.mean_cl = torch.tensor(mean_cl, dtype=torch.float32, device=device)
        self.ell = torch.tensor(ell, dtype=torch.float32, device=device)
        weighting = self.ell ** 3
        self.weighting = weighting / weighting.mean()
        rad_per_pix = (ps * np.pi / 180)
        lx = 2 * np.pi * np.fft.fftfreq(256, d=rad_per_pix)
        ly = 2 * np.pi * np.fft.fftfreq(256, d=rad_per_pix)
        L2D = np.sqrt(np.meshgrid(lx, lx)[0]**2 + np.meshgrid(ly, ly)[1]**2)
        ell_min = 2 * np.pi / (256 * rad_per_pix)
        ell_max = np.max(L2D)
        bins = np.linspace(ell_min, ell_max, len(ell) + 1)
        bin_indices = np.digitize(L2D.flatten(), bins) - 1
        B = torch.zeros((len(ell), 256*256), device=device)
        for i in range(len(ell)):
            mask = (bin_indices == i)
            if mask.sum() > 0:
                B[i, mask] = 1.0 / mask.sum()
        self.B = B
        self.rad_per_pix = rad_per_pix
    def forward(self, pred):
        pred_unscaled = pred / TSZ_SCALE
        pred_fft = torch.fft.fft2(pred_unscaled)
        norm = (256 * 256)
        pred_ps2d = torch.abs(pred_fft)**2 * (self.rad_per_pix**2) / norm
        pred_ps2d_flat = pred_ps2d.view(pred.size(0), -1)
        pred_ps1d = torch.matmul(pred_ps2d_flat, self.B.T)
        eps = 1e-12
        log_pred = torch.log(pred_ps1d + eps)
        log_target = torch.log(self.mean_cl.unsqueeze(0) + eps)
        return torch.mean(self.weighting.unsqueeze(0) * (log_pred - log_target)**2)
def ncc_loss(pred, target, cib_maps):
    residual = pred - target
    res_mean = residual.mean(dim=(2,3), keepdim=True)
    cib_mean = cib_maps.mean(dim=(2,3), keepdim=True)
    res_std = residual.std(dim=(2,3), keepdim=True) + 1e-8
    cib_std = cib_maps.std(dim=(2,3), keepdim=True) + 1e-8
    res_norm = (residual - res_mean) / res_std
    cib_norm = (cib_maps - cib_mean) / cib_std
    ncc = (res_norm * cib_norm).mean(dim=(2,3))
    return torch.mean(ncc**2)
def compute_normalization_stats(dataset, num_samples=200):
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    obs_sum = torch.zeros(6)
    obs_sq_sum = torch.zeros(6)
    count = 0
    for obs, cib, tsz in loader:
        obs_sum += obs.sum(dim=(0, 2, 3))
        obs_sq_sum += (obs**2).sum(dim=(0, 2, 3))
        count += obs.size(0) * obs.size(2) * obs.size(3)
        if count >= num_samples * 256 * 256:
            break
    mean = obs_sum / count
    std = torch.sqrt(obs_sq_sum / count - mean**2)
    return mean.view(1, 6, 1, 1), std.view(1, 6, 1, 1)
if __name__ == '__main__':
    print('Initializing Data Pipeline and Splits...')
    n_patch = 1523
    max_tsz = np.zeros(n_patch)
    for p in range(n_patch):
        tsz = np.load(BASE + '/tsz.npy', mmap_mode='r')[p]
        max_tsz[p] = np.max(tsz)
    top_5_percent_idx = np.argsort(max_tsz)[-int(0.05 * n_patch):]
    safe_patches = np.setdiff1d(np.arange(n_patch), top_5_percent_idx)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(safe_patches)
    n_train = int(0.70 * n_patch)
    n_val = int(0.15 * n_patch)
    train_idx = safe_patches[:n_train]
    val_idx = safe_patches[n_train:n_train+n_val]
    test_idx = np.concatenate([safe_patches[n_train+n_val:], top_5_percent_idx])
    np.savez(os.path.join(DATA_DIR, 'splits.npz'), train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, top_5_percent_idx=top_5_percent_idx)
    print('Splits created: Train=' + str(len(train_idx)) + ', Val=' + str(len(val_idx)) + ', Test=' + str(len(test_idx)))
    cl_list = []
    ell = None
    for p in train_idx:
        tsz = np.load(BASE + '/tsz.npy', mmap_mode='r')[p]
        cl, ell_bins = utils.powers(tsz, tsz, ps=5*60/256)
        cl_list.append(cl)
        if ell is None:
            ell = ell_bins
    mean_cl = np.mean(cl_list, axis=0)
    train_dataset = CompSepDataset(train_idx, BASE)
    val_dataset = CompSepDataset(val_idx, BASE)
    print('Computing normalization statistics...')
    obs_mean, obs_std = compute_normalization_stats(train_dataset)
    np.savez(os.path.join(DATA_DIR, 'normalization_stats.npz'), obs_mean=obs_mean.numpy(), obs_std=obs_std.numpy())
    print('Normalization Mean: ' + str(obs_mean.flatten().numpy()))
    print('Normalization Std: ' + str(obs_std.flatten().numpy()))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SR_DAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    l1_criterion = nn.L1Loss()
    spec_criterion = PowerSpectrumLoss(mean_cl, ell, device=device)
    lambda_1, lambda_2, num_epochs = 0.1, 0.1, 20
    train_losses, val_losses = [], []
    obs_mean, obs_std = obs_mean.to(device), obs_std.to(device)
    print('\nStarting SR-DAE Training...')
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_l1, train_spec, train_corr = 0, 0, 0, 0
        for obs, cib, tsz in train_loader:
            obs, cib, tsz = obs.to(device), cib.to(device), tsz.to(device) * TSZ_SCALE
            obs_norm = (obs - obs_mean) / obs_std
            optimizer.zero_grad()
            pred = model(obs_norm[:, :3, :, :], obs_norm[:, 3:, :, :])
            loss_l1 = l1_criterion(pred, tsz)
            loss_spec = spec_criterion(pred)
            loss_corr = ncc_loss(pred, tsz, cib)
            loss = loss_l1 + lambda_1 * loss_spec + lambda_2 * loss_corr
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_l1 += loss_l1.item()
            train_spec += loss_spec.item()
            train_corr += loss_corr.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs, cib, tsz in val_loader:
                obs, cib, tsz = obs.to(device), cib.to(device), tsz.to(device) * TSZ_SCALE
                obs_norm = (obs - obs_mean) / obs_std
                pred = model(obs_norm[:, :3, :, :], obs_norm[:, 3:, :, :])
                val_loss += (l1_criterion(pred, tsz) + lambda_1 * spec_criterion(pred) + lambda_2 * ncc_loss(pred, tsz, cib)).item()
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print('Epoch ' + str(epoch+1) + '/' + str(num_epochs) + ' | Train Loss: ' + str(round(train_loss, 4)) + ' (L1: ' + str(round(train_l1/len(train_loader), 4)) + ', Spec: ' + str(round(train_spec/len(train_loader), 4)) + ', Corr: ' + str(round(train_corr/len(train_loader), 4)) + ') | Val Loss: ' + str(round(val_loss, 4)))
    torch.save(model.state_dict(), os.path.join(DATA_DIR, 'sr_dae_weights.pth'))
    np.savez(os.path.join(DATA_DIR, 'training_history.npz'), train_losses=train_losses, val_losses=val_losses)
    print('\nTraining completed successfully.')
    print('Final Train Loss: ' + str(round(train_losses[-1], 4)))
    print('Final Val Loss: ' + str(round(val_losses[-1], 4)))
    print('Model weights saved to ' + os.path.join(DATA_DIR, 'sr_dae_weights.pth'))
    print('Training history saved to ' + os.path.join(DATA_DIR, 'training_history.npz'))