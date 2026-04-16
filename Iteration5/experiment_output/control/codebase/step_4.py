# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath('codebase'))
    sys.path.insert(0, '/home/node/data/compsep_data')
    import utils

    BASE = '/home/node/data/compsep_data/cut_maps'
    DATA_DIR = 'data'
    TSZ_SCALE = 1e6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class CompSepDataset(Dataset):
        def __init__(self, patch_indices, base_dir, epoch_size=None):
            self.patch_indices = patch_indices
            self.base_dir = base_dir
            self.epoch_size = epoch_size if epoch_size else len(patch_indices)
            self.frequencies = [90, 150, 217, 353, 545, 857]
            self.stacked = {f: np.load(base_dir + '/stacked_' + str(f) + '.npy', mmap_mode='r') for f in self.frequencies}
            self.so_noise = {f: np.load(base_dir + '/so_noise/' + str(f) + '.npy', mmap_mode='r') for f in [90, 150, 217]}
            self.tsz = np.load(base_dir + '/tsz.npy', mmap_mode='r')
        def __len__(self):
            return self.epoch_size
        def __getitem__(self, idx):
            p = self.patch_indices[idx % len(self.patch_indices)]
            i_planck = np.random.randint(100)
            i_so = np.random.randint(3000)
            obs = np.zeros((6, 256, 256), dtype=np.float32)
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
            tsz = self.tsz[p]
            return torch.tensor(obs, dtype=torch.float32), torch.tensor(tsz, dtype=torch.float32).unsqueeze(0)

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

    class SinusoidalPositionEmbeddings(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
        def forward(self, time):
            device = time.device
            half_dim = self.dim // 2
            embeddings = math.log(10000) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
            embeddings = time[:, None] * embeddings[None, :]
            embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
            return embeddings

    class CDM(nn.Module):
        def __init__(self, pretrained_dae=None):
            super().__init__()
            self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(256), nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 1024))
            self.enc1_so = ConvBlock(4, 64)
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
            self.time_emb_bot = nn.Linear(1024, 1024)
            self.time_emb4 = nn.Linear(1024, 512)
            self.time_emb3 = nn.Linear(1024, 256)
            self.time_emb2 = nn.Linear(1024, 128)
            self.time_emb1 = nn.Linear(1024, 64)
            if pretrained_dae is not None:
                self.load_pretrained(pretrained_dae)
        def load_pretrained(self, dae):
            with torch.no_grad():
                self.enc1_so.conv[0].weight[:, :3, :, :] = dae.enc1_so.conv[0].weight
                self.enc1_so.conv[0].weight[:, 3:, :, :] = 0.0
                self.enc1_so.conv[0].bias = dae.enc1_so.conv[0].bias
                for i in range(1, len(self.enc1_so.conv)):
                    if hasattr(self.enc1_so.conv[i], 'weight'):
                        self.enc1_so.conv[i].weight.copy_(dae.enc1_so.conv[i].weight)
                    if hasattr(self.enc1_so.conv[i], 'bias') and self.enc1_so.conv[i].bias is not None:
                        self.enc1_so.conv[i].bias.copy_(dae.enc1_so.conv[i].bias)
                layers_to_copy = ['enc2_so', 'enc3_so', 'enc4_so', 'enc1_pl', 'enc2_pl', 'enc3_pl', 'enc4_pl', 'bot_so', 'bot_pl', 'cross_bot', 'up4', 'cross4', 'dec4', 'up3', 'cross3', 'dec3', 'up2', 'cross2', 'dec2', 'up1', 'cross1', 'dec1', 'final']
                for layer_name in layers_to_copy:
                    getattr(self, layer_name).load_state_dict(getattr(dae, layer_name).state_dict())
        def forward(self, x_t, t, x_so, x_pl):
            t_emb = self.time_mlp(t)
            x_so_in = torch.cat([x_so, x_t], dim=1)
            e1_so = self.enc1_so(x_so_in)
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
            b = b + self.time_emb_bot(t_emb)[:, :, None, None]
            d4 = self.up4(b)
            e4 = self.cross4(e4_so, e4_pl)
            d4 = self.dec4(torch.cat([d4, e4], dim=1))
            d4 = d4 + self.time_emb4(t_emb)[:, :, None, None]
            d3 = self.up3(d4)
            e3 = self.cross3(e3_so, e3_pl)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d3 = d3 + self.time_emb3(t_emb)[:, :, None, None]
            d2 = self.up2(d3)
            e2 = self.cross2(e2_so, e2_pl)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d2 = d2 + self.time_emb2(t_emb)[:, :, None, None]
            d1 = self.up1(d2)
            e1 = self.cross1(e1_so, e1_pl)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            d1 = d1 + self.time_emb1(t_emb)[:, :, None, None]
            return self.final(d1)

    def linear_beta_schedule(timesteps):
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    timesteps = 1000
    betas = linear_beta_schedule(timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(denoise_model, x_start, t, x_so, x_pl, noise=None, loss_type='l2'):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, x_so, x_pl)
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def ddim_sample(model, shape, x_so, x_pl, ddim_timesteps=50, eta=0.0):
        device = next(model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)
        step_size = timesteps // ddim_timesteps
        time_steps = list(reversed(range(0, timesteps, step_size)))
        for i, step in enumerate(time_steps):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            pred_noise = model(img, t, x_so, x_pl)
            alpha_bar_t = extract(alphas_cumprod, t, img.shape)
            prev_step = step - step_size
            if prev_step >= 0:
                alpha_bar_t_prev = extract(alphas_cumprod, torch.full((b,), prev_step, device=device, dtype=torch.long), img.shape)
            else:
                alpha_bar_t_prev = torch.ones_like(alpha_bar_t)
            sigma = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            pred_x0 = (img - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma**2) * pred_noise
            if sigma.max() > 0:
                noise = torch.randn_like(img)
            else:
                noise = 0.0
            img = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma * noise
        return img

    splits = np.load(os.path.join(DATA_DIR, 'splits.npz'))
    train_idx = splits['train_idx']
    test_idx = splits['test_idx']
    norm_stats = np.load(os.path.join(DATA_DIR, 'normalization_stats.npz'))
    obs_mean = torch.tensor(norm_stats['obs_mean'], dtype=torch.float32).to(device)
    obs_std = torch.tensor(norm_stats['obs_std'], dtype=torch.float32).to(device)
    
    print("Computing tSZ normalization statistics...")
    tsz_sum = 0
    tsz_sq_sum = 0
    count = 0
    for p in train_idx:
        tsz = np.load(BASE + '/tsz.npy', mmap_mode='r')[p] * TSZ_SCALE
        tsz_sum += np.sum(tsz)
        tsz_sq_sum += np.sum(tsz**2)
        count += tsz.size
    tsz_mean = tsz_sum / count
    tsz_std = np.sqrt(tsz_sq_sum / count - tsz_mean**2)
    print("tSZ Mean (scaled): " + str(round(tsz_mean, 4)) + ", tSZ Std (scaled): " + str(round(tsz_std, 4)))
    np.savez(os.path.join(DATA_DIR, 'tsz_norm_stats.npz'), tsz_mean=tsz_mean, tsz_std=tsz_std)
    
    dae = SR_DAE()
    dae.load_state_dict(torch.load(os.path.join(DATA_DIR, 'sr_dae_weights.pth'), map_location='cpu'))
    cdm = CDM(pretrained_dae=dae).to(device)
    
    epochs = 15
    batch_size = 16
    optimizer = torch.optim.AdamW(cdm.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    train_dataset = CompSepDataset(train_idx, BASE)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    print("\nStarting CDM Training...")
    for epoch in range(epochs):
        cdm.train()
        train_loss = 0
        for obs, tsz in train_loader:
            obs = obs.to(device)
            tsz = tsz.to(device) * TSZ_SCALE
            obs_norm = (obs - obs_mean) / obs_std
            x_so = obs_norm[:, :3, :, :]
            x_pl = obs_norm[:, 3:, :, :]
            x_start = (tsz - tsz_mean) / tsz_std
            t = torch.randint(0, timesteps, (obs.shape[0],), device=device).long()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = p_losses(cdm, x_start, t, x_so, x_pl, loss_type='l2')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        scheduler.step()
        print("Epoch " + str(epoch+1) + "/" + str(epochs) + " | Loss: " + str(round(train_loss / len(train_loader), 4)))
        
    torch.save(cdm.state_dict(), os.path.join(DATA_DIR, 'cdm_weights.pth'))
    print("Saved CDM weights to data/cdm_weights.pth")
    
    print("\nGenerating realizations for test set...")
    N_realizations = 10
    test_dataset = CompSepDataset(test_idx, BASE)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    cdm.eval()
    all_cdm_means = []
    all_cdm_vars = []
    all_true_tsz = []
    all_peak_mass_proxy = []
    
    with torch.no_grad():
        for batch_idx, (obs, tsz) in enumerate(test_loader):
            obs = obs.to(device)
            obs_norm = (obs - obs_mean) / obs_std
            x_so = obs_norm[:, :3, :, :]
            x_pl = obs_norm[:, 3:, :, :]
            b = obs.shape[0]
            realizations = []
            for n in range(N_realizations):
                with torch.cuda.amp.autocast():
                    gen_norm = ddim_sample(cdm, (b, 1, 256, 256), x_so, x_pl, ddim_timesteps=50)
                gen_tsz = (gen_norm * tsz_std + tsz_mean) / TSZ_SCALE
                realizations.append(gen_tsz.cpu().numpy())
            realizations = np.stack(realizations, axis=1)
            mean_map = np.mean(realizations, axis=1).squeeze(1)
            var_map = np.var(realizations, axis=1).squeeze(1)
            all_cdm_means.append(mean_map)
            all_cdm_vars.append(var_map)
            all_true_tsz.append(tsz.numpy().squeeze(1))
            for i in range(b):
                peak_val = np.max(tsz[i].numpy())
                all_peak_mass_proxy.append(peak_val)
            if (batch_idx + 1) % 5 == 0:
                print("Processed batch " + str(batch_idx+1) + "/" + str(len(test_loader)))
                
    all_cdm_means = np.concatenate(all_cdm_means, axis=0)
    all_cdm_vars = np.concatenate(all_cdm_vars, axis=0)
    all_true_tsz = np.concatenate(all_true_tsz, axis=0)
    all_peak_mass_proxy = np.array(all_peak_mass_proxy)
    
    np.savez(os.path.join(DATA_DIR, 'cdm_results.npz'), cdm_means=all_cdm_means, cdm_vars=all_cdm_vars, true_tsz=all_true_tsz, peak_mass_proxy=all_peak_mass_proxy, test_idx=test_idx)
    print("Saved CDM results to data/cdm_results.npz")
    
    print("\nGenerating DAE predictions for test set...")
    dae.eval()
    dae = dae.to(device)
    all_dae_preds = []
    
    with torch.no_grad():
        for obs, _ in test_loader:
            obs = obs.to(device)
            obs_norm = (obs - obs_mean) / obs_std
            x_so = obs_norm[:, :3, :, :]
            x_pl = obs_norm[:, 3:, :, :]
            pred = dae(x_so, x_pl)
            all_dae_preds.append((pred.cpu().numpy().squeeze(1)) / TSZ_SCALE)
            
    all_dae_preds = np.concatenate(all_dae_preds, axis=0)
    np.savez(os.path.join(DATA_DIR, 'dae_results.npz'), dae_preds=all_dae_preds)
    print("Saved DAE results to data/dae_results.npz")
    
    scaled_cdm_mse = np.mean((all_cdm_means * TSZ_SCALE - all_true_tsz * TSZ_SCALE)**2)
    scaled_dae_mse = np.mean((all_dae_preds * TSZ_SCALE - all_true_tsz * TSZ_SCALE)**2)
    print("\nEvaluation on Test Set:")
    print("Scaled CDM Mean MSE: " + str(round(scaled_cdm_mse, 4)))
    print("Scaled DAE Mean MSE: " + str(round(scaled_dae_mse, 4)))