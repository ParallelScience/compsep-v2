# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import scipy.ndimage
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.abspath('codebase'))
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
from step_2 import CompSepDataset
from step_3 import SR_DAE
from step_4 import CompositeLoss, get_lambda4_schedule
def generate_gnfw_template(n_pixels=256, pix_size_arcmin=1.17, theta_c=2.0, peak_amplitude=5e-5):
    x = np.arange(n_pixels) - n_pixels // 2
    y = np.arange(n_pixels) - n_pixels // 2
    xx, yy = np.meshgrid(x, y)
    r_arcmin = np.sqrt(xx**2 + yy**2) * pix_size_arcmin
    alpha = 1.051
    beta = 5.4905
    gamma = 0.3081
    z = np.linspace(-10 * theta_c, 10 * theta_c, 200)
    dz = z[1] - z[0]
    y_proj = np.zeros((n_pixels, n_pixels))
    for i in range(len(z)):
        r_3d = np.sqrt(r_arcmin**2 + z[i]**2)
        cx = np.clip(r_3d / theta_c, 1e-4, np.inf)
        p = 1.0 / ((cx)**gamma * (1 + cx**alpha)**((beta - gamma)/alpha))
        y_proj += p * dz
    y_proj = y_proj / np.max(y_proj) * peak_amplitude
    return y_proj
def apply_beam(map_2d, fwhm_arcmin, pix_size_arcmin=1.17):
    sigma = fwhm_arcmin / (2.355 * pix_size_arcmin)
    return scipy.ndimage.gaussian_filter(map_2d, sigma=sigma)
if __name__ == '__main__':
    data_dir = 'data/'
    scaling_data = np.load(os.path.join(data_dir, 'scaling_params.npz'), allow_pickle=True)
    scaling_params = scaling_data['scaling_params'].item()
    tsz_clip_threshold = scaling_data['tsz_clip_threshold'].item()
    train_indices = scaling_data['train_indices']
    val_indices = scaling_data['val_indices']
    base_dir = '/home/node/data/compsep_data/cut_maps'
    train_dataset = CompSepDataset(train_indices, base_dir, scaling_params, tsz_clip_threshold, split='train')
    val_dataset = CompSepDataset(val_indices, base_dir, scaling_params, tsz_clip_threshold, split='val')
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SR_DAE(main_in_channels=3, aux_in_channels=3, out_channels=1).to(device)
    epochs = 30
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = CompositeLoss(device=device)
    lambda4_schedule = get_lambda4_schedule(epochs=epochs, start_epoch=5, end_epoch=20)
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_epoch = -1
    for epoch in range(epochs):
        model.train()
        criterion.lambda4 = lambda4_schedule[epoch]
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).to(device)
            optimizer.zero_grad()
            with autocast(enabled=True):
                outputs = model(inputs)
                loss, _, _, _, _ = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.unsqueeze(1).to(device)
                with autocast(enabled=True):
                    outputs = model(inputs)
                    loss, _, _, _, _ = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(data_dir, 'best_model.pth'))
    model.load_state_dict(torch.load(os.path.join(data_dir, 'best_model.pth')))
    model.eval()
    mass_scales = [{'name': 'Low Mass', 'theta_c': 1.0, 'peak': 1e-5}, {'name': 'Medium Mass', 'theta_c': 2.0, 'peak': 5e-5}, {'name': 'High Mass', 'theta_c': 4.0, 'peak': 1e-4}]
    beams = {90: 2.2, 150: 1.4, 217: 1.0, 353: 4.5, 545: 4.72, 857: 4.42}
    frequencies = [90, 150, 217, 353, 545, 857]
    n_injections = 10
    so_noise = {freq: np.load(os.path.join(base_dir, 'so_noise', str(freq) + '.npy'), mmap_mode='r') for freq in [90, 150, 217]}
    planck_noise = {freq: np.load(os.path.join(base_dir, 'planck_noise', 'planck_noise_' + str(freq) + '_0.npy'), mmap_mode='r') for freq in [353, 545, 857]}
    results = []
    for scale in mass_scales:
        template = generate_gnfw_template(theta_c=scale['theta_c'], peak_amplitude=scale['peak'])
        true_Y = np.sum(template)
        recovery_fractions = []
        radial_residuals = []
        for i in range(n_injections):
            inputs = []
            for freq in frequencies:
                sig = template * utils.tsz(freq)
                sig_smoothed = apply_beam(sig, beams[freq])
                if freq <= 217:
                    noise = so_noise[freq][i]
                else:
                    raw_noise = planck_noise[freq][i]
                    noise = raw_noise * 1e6 if freq == 353 else raw_noise * 1e6 * utils.jysr2uk(freq)
                channel_data = (sig_smoothed + noise - scaling_params[freq]['median']) / (scaling_params[freq]['iqr'] + 1e-8)
                inputs.append(channel_data)
            inputs_tensor = torch.from_numpy(np.stack(inputs, axis=0).astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                with autocast(enabled=True):
                    pred = model(inputs_tensor)
            pred_np = pred.squeeze().cpu().numpy()
            pred_clipped = np.clip(pred_np, -0.999 * tsz_clip_threshold, 0.999 * tsz_clip_threshold)
            pred_physical = tsz_clip_threshold * np.arctanh(pred_clipped / tsz_clip_threshold)
            recovery_fractions.append(np.sum(pred_physical) / true_Y)
            n_pixels = 256
            r_arcmin = np.sqrt((np.arange(n_pixels) - n_pixels // 2)[:, None]**2 + (np.arange(n_pixels) - n_pixels // 2)[None, :]**2) * 1.17
            bins = np.linspace(0, 10, 11)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            true_prof, _ = np.histogram(r_arcmin, bins=bins, weights=template)
            counts, _ = np.histogram(r_arcmin, bins=bins)
            pred_prof, _ = np.histogram(r_arcmin, bins=bins, weights=pred_physical)
            radial_residuals.append((pred_prof - true_prof) / np.maximum(counts, 1))
        results.append({'name': scale['name'], 'recovery_mean': np.mean(recovery_fractions), 'recovery_std': np.std(recovery_fractions), 'radial_bins': bin_centers, 'radial_residuals': np.mean(radial_residuals, axis=0)})
    np.savez(os.path.join(data_dir, 'signal_injection_results.npz'), results=np.array(results, dtype=object))