# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.abspath('codebase'))
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
from step_4 import SR_DAE, configure_training, train_step
from step_6 import FineTuneLoss
os.environ['OMP_NUM_THREADS'] = '16'
data_dir = 'data/'
class AblatedDataset(Dataset):
    def __init__(self, features_path, targets_path, masks_path):
        features_mmap = np.load(features_path, mmap_mode='r')
        self.features = torch.from_numpy(np.array(features_mmap[:, :3, :, :])).float()
        self.targets = torch.from_numpy(np.load(targets_path)).float()
        self.masks = torch.from_numpy(np.load(masks_path)).float()
        self.length = self.features.shape[0]
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.masks[idx]
def train_ablated_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training ablated model (without CIB features)...')
    train_dataset = AblatedDataset(os.path.join(data_dir, 'train_features.npy'), os.path.join(data_dir, 'train_targets.npy'), os.path.join(data_dir, 'train_masks.npy'))
    val_dataset = AblatedDataset(os.path.join(data_dir, 'val_features.npy'), os.path.join(data_dir, 'val_targets.npy'), os.path.join(data_dir, 'val_masks.npy'))
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = SR_DAE(in_channels=3, out_channels=1, init_features=32).to(device)
    epochs = 30
    optimizer, scheduler, criterion = configure_training(model, lr=1e-3, weight_decay=1e-4, epochs=epochs, steps_per_epoch=len(train_loader))
    criterion = criterion.to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, targets, masks = [b.to(device, non_blocking=True) for b in batch]
            train_step(model, (inputs, targets, masks), criterion, optimizer, scaler=scaler, clip_val=1.0)
            scheduler.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, masks = [b.to(device, non_blocking=True) for b in batch]
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss, _, _ = criterion(outputs, targets, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(data_dir, 'best_ablated_model.pth'))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print('Epoch ' + str(epoch+1) + '/' + str(epochs) + ' - Val Loss: ' + str(round(val_loss, 4)))
    print('\nFine-tuning ablated model...')
    model.load_state_dict(torch.load(os.path.join(data_dir, 'best_ablated_model.pth'), map_location=device))
    epochs_ft = 15
    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=epochs_ft * len(train_loader))
    criterion_ft = FineTuneLoss(mask_weight=1e3, gamma=2.0, spectral_weight=0.1, wavelet_weight=1.0).to(device)
    best_val_loss_ft = float('inf')
    for epoch in range(epochs_ft):
        model.train()
        for batch in train_loader:
            inputs, targets, masks = [b.to(device, non_blocking=True) for b in batch]
            optimizer_ft.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss, _, _, _ = criterion_ft(outputs, targets, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_ft)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer_ft)
            scaler.update()
            scheduler_ft.step()
        model.eval()
        val_loss = 0.0
        valid_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, masks = [b.to(device, non_blocking=True) for b in batch]
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss, _, _, _ = criterion_ft(outputs, targets, masks)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    valid_batches += 1
        if valid_batches > 0:
            val_loss /= valid_batches
            if val_loss < best_val_loss_ft:
                best_val_loss_ft = val_loss
                torch.save(model.state_dict(), os.path.join(data_dir, 'finetuned_ablated_model.pth'))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print('FT Epoch ' + str(epoch+1) + '/' + str(epochs_ft) + ' - Val Loss: ' + str(round(val_loss, 4)))
    print('Ablated model training completed.')
def compute_r_ell(pred, true):
    ell, cl_pred_true = utils.powers(pred, true, ps=5.0)
    _, cl_pred_pred = utils.powers(pred, pred, ps=5.0)
    _, cl_true_true = utils.powers(true, true, ps=5.0)
    r_ell = cl_pred_true / np.sqrt(cl_pred_pred * cl_true_true + 1e-15)
    return ell, r_ell
def evaluate_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_features = np.load(os.path.join(data_dir, 'test_features.npy'))
    test_targets = np.load(os.path.join(data_dir, 'test_targets.npy'))
    stats = np.load(os.path.join(data_dir, 'tsz_norm_stats.npz'))
    train_mean = stats['mean']
    train_std = stats['std']
    full_model = SR_DAE(in_channels=6, out_channels=1, init_features=32).to(device)
    full_model.load_state_dict(torch.load(os.path.join(data_dir, 'finetuned_model.pth'), map_location=device))
    full_model.eval()
    ablated_model = SR_DAE(in_channels=3, out_channels=1, init_features=32).to(device)
    ablated_model.load_state_dict(torch.load(os.path.join(data_dir, 'finetuned_ablated_model.pth'), map_location=device))
    ablated_model.eval()
    r_ell_full_list = []
    r_ell_ablated_list = []
    print('\nEvaluating models on test set...')
    with torch.no_grad():
        for i in range(len(test_features)):
            feat = torch.from_numpy(test_features[i:i+1]).float().to(device)
            pred_full = full_model(feat).cpu().numpy()[0, 0]
            pred_full_unnorm = pred_full * train_std + train_mean
            feat_ablated = feat[:, :3, :, :]
            pred_ablated = ablated_model(feat_ablated).cpu().numpy()[0, 0]
            pred_ablated_unnorm = pred_ablated * train_std + train_mean
            true_unnorm = test_targets[i] * train_std + train_mean
            ell, r_full = compute_r_ell(pred_full_unnorm, true_unnorm)
            _, r_ablated = compute_r_ell(pred_ablated_unnorm, true_unnorm)
            r_ell_full_list.append(r_full)
            r_ell_ablated_list.append(r_ablated)
    r_ell_full_mean = np.nanmean(r_ell_full_list, axis=0)
    r_ell_ablated_mean = np.nanmean(r_ell_ablated_list, axis=0)
    print('\nPerformance Metrics (r_ell):')
    indices_to_print = [10, 50, 100, 150]
    for idx in indices_to_print:
        if idx < len(ell):
            print('ell = ' + str(round(ell[idx], 1)) + ': Full r_ell = ' + str(round(r_ell_full_mean[idx], 4)) + ', Ablated r_ell = ' + str(round(r_ell_ablated_mean[idx], 4)))
    np.savez(os.path.join(data_dir, 'r_ell_results.npz'), ell=ell, r_ell_full=r_ell_full_mean, r_ell_ablated=r_ell_ablated_mean)
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
def gnfw_pressure(x):
    P0 = 8.403
    c500 = 1.177
    alpha = 1.0510
    beta = 5.4905
    gamma = 0.3081
    cx = c500 * x
    return P0 / ((cx)**gamma * (1 + cx**alpha)**((beta - gamma)/alpha))
def projected_gnfw(theta_arcmin, theta_500_arcmin):
    x_2d = theta_arcmin / theta_500_arcmin
    y_proj = np.zeros_like(x_2d)
    z_norm = np.linspace(0, 5, 200)
    dz = z_norm[1] - z_norm[0]
    for i in range(len(z_norm)):
        r_norm = np.sqrt(x_2d**2 + z_norm[i]**2)
        r_norm = np.clip(r_norm, 1e-4, None)
        y_proj += gnfw_pressure(r_norm) * dz
    return y_proj * 2
def generate_cluster_y(M_500_log, N=256, pixel_scale=1.171875):
    M_14 = 10**(M_500_log - 14.0)
    y_peak = 1e-5 * (M_14)**(5/3)
    theta_500 = 2.0 * (M_14)**(1/3)
    x = np.arange(N) - N//2
    y = np.arange(N) - N//2
    X, Y = np.meshgrid(x, y)
    R_arcmin = np.sqrt(X**2 + Y**2) * pixel_scale
    prof = projected_gnfw(R_arcmin, theta_500)
    prof = prof / np.max(prof) * y_peak
    return prof
def get_local_stds(patch_i):
    noise_idx = np.load(os.path.join(data_dir, 'noise_indices.npz'))
    i_so = noise_idx['i_so'][patch_i]
    i_planck = noise_idx['i_planck'][patch_i]
    BASE = '/home/node/data/compsep_data/cut_maps'
    so_noise_90 = np.load(BASE + '/so_noise/90.npy', mmap_mode='r')[i_so].copy()
    so_noise_150 = np.load(BASE + '/so_noise/150.npy', mmap_mode='r')[i_so].copy()
    so_noise_217 = np.load(BASE + '/so_noise/217.npy', mmap_mode='r')[i_so].copy()
    pn_353 = np.load(BASE + '/planck_noise/planck_noise_353_' + str(i_planck) + '.npy', mmap_mode='r')[patch_i].copy() * 1e6
    N = 256
    L_rad = 5.0 * np.pi / 180.0
    beam_150_to_90 = get_beam_window(N, L_rad, 1.4, 2.2)
    beam_217_to_150 = get_beam_window(N, L_rad, 1.0, 1.4)
    beam_217_to_353 = get_beam_window(N, L_rad, 1.0, 4.5)
    fft_n150 = np.fft.fft2(so_noise_150)
    sm_n150 = np.real(np.fft.ifft2(fft_n150 * beam_150_to_90))
    n_diff_150_90 = sm_n150 - so_noise_90
    local_std_150_90 = np.std(n_diff_150_90)
    fft_n217 = np.fft.fft2(so_noise_217)
    sm_n217_to_150 = np.real(np.fft.ifft2(fft_n217 * beam_217_to_150))
    n_diff_217_150 = sm_n217_to_150 - so_noise_150
    local_std_217_150 = np.std(n_diff_217_150)
    sm_n217_to_353 = np.real(np.fft.ifft2(fft_n217 * beam_217_to_353))
    n_diff_353_217 = pn_353 - sm_n217_to_353
    local_std_353_217 = np.std(n_diff_353_217)
    if local_std_150_90 == 0: local_std_150_90 = 1.0
    if local_std_217_150 == 0: local_std_217_150 = 1.0
    if local_std_353_217 == 0: local_std_353_217 = 1.0
    return [local_std_150_90, local_std_217_150, local_std_353_217]
def inject_cluster_features(cluster_y, base_features, local_stds):
    N = 256
    L_rad = 5.0 * np.pi / 180.0
    beam_150_to_90 = get_beam_window(N, L_rad, 1.4, 2.2)
    beam_217_to_150 = get_beam_window(N, L_rad, 1.0, 1.4)
    beam_217_to_353 = get_beam_window(N, L_rad, 1.0, 4.5)
    tsz_90 = utils.tsz(90)
    tsz_150 = utils.tsz(150)
    tsz_217 = utils.tsz(217)
    tsz_353 = utils.tsz(353)
    c_90 = cluster_y * tsz_90
    c_150 = cluster_y * tsz_150
    c_217 = cluster_y * tsz_217
    c_353 = cluster_y * tsz_353
    fft_c150 = np.fft.fft2(c_150)
    sm_c150 = np.real(np.fft.ifft2(fft_c150 * beam_150_to_90))
    fft_c217 = np.fft.fft2(c_217)
    sm_c217_to_150 = np.real(np.fft.ifft2(fft_c217 * beam_217_to_150))
    sm_c217_to_353 = np.real(np.fft.ifft2(fft_c217 * beam_217_to_353))
    d_150_90 = sm_c150 - c_90
    d_217_150 = sm_c217_to_150 - c_150
    d_353_217 = c_353 - sm_c217_to_353
    new_features = base_features.copy()
    new_features[0] += d_150_90 / local_stds[0]
    new_features[1] += d_217_150 / local_stds[1]
    new_features[2] += d_353_217 / local_stds[2]
    return new_features
def determine_reconstruction_threshold():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_model = SR_DAE(in_channels=6, out_channels=1, init_features=32).to(device)
    full_model.load_state_dict(torch.load(os.path.join(data_dir, 'finetuned_model.pth'), map_location=device))
    full_model.eval()
    test_features = np.load(os.path.join(data_dir, 'test_features.npy'))
    test_targets = np.load(os.path.join(data_dir, 'test_targets.npy'))
    split = np.load(os.path.join(data_dir, 'split_indices.npz'))
    test_idx = split['test']
    stats = np.load(os.path.join(data_dir, 'tsz_norm_stats.npz'))
    train_mean = stats['mean']
    train_std = stats['std']
    true_unnorm = test_targets * train_std + train_mean
    max_y_per_patch = np.max(true_unnorm, axis=(1, 2))
    null_patch_idx = np.argmin(max_y_per_patch)
    global_patch_i = test_idx[null_patch_idx]
    base_feat = test_features[null_patch_idx]
    print('\nSelected null patch ' + str(null_patch_idx) + ' (global index ' + str(global_patch_i) + ') with max y = ' + str(round(max_y_per_patch[null_patch_idx], 2)))
    local_stds = get_local_stds(global_patch_i)
    masses = [13.5, 14.0, 14.5, 15.0]
    print('\nCluster Injection Results:')
    print('log M_500  | Injected Peak y    | Reconstructed Peak y   | Recovery Fraction')
    print('-' * 75)
    results = []
    base_feat_tensor = torch.from_numpy(base_feat).unsqueeze(0).float().to(device)
    with torch.no_grad():
        base_pred_norm = full_model(base_feat_tensor).cpu().numpy()[0, 0]
    base_pred_unnorm = base_pred_norm * train_std + train_mean
    base_center_slice = base_pred_unnorm[128-10:128+10, 128-10:128+10]
    base_peak = np.max(base_center_slice)
    for M_log in masses:
        cluster_y = generate_cluster_y(M_log)
        injected_peak = np.max(cluster_y)
        new_feat = inject_cluster_features(cluster_y, base_feat, local_stds)
        feat_tensor = torch.from_numpy(new_feat).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred_norm = full_model(feat_tensor).cpu().numpy()[0, 0]
        pred_unnorm = pred_norm * train_std + train_mean
        center_slice = pred_unnorm[128-10:128+10, 128-10:128+10]
        recon_peak = np.max(center_slice)
        net_recon_peak = recon_peak - base_peak
        recovery_frac = net_recon_peak / injected_peak if injected_peak > 0 else 0
        print(str(M_log) + '       | ' + str(round(injected_peak, 2)) + 'e-05 | ' + str(round(net_recon_peak, 2)) + 'e-05 | ' + str(round(recovery_frac, 2)))
        results.append({'M_log': M_log, 'injected_peak': injected_peak, 'recon_peak': net_recon_peak, 'recovery_frac': recovery_frac})
    np.save(os.path.join(data_dir, 'cluster_injection_results.npy'), results)
if __name__ == '__main__':
    train_ablated_model()
    evaluate_models()
    determine_reconstruction_threshold()
    print('\nStep 7 completed successfully.')