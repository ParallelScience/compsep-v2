# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from step_1 import CompSepDataset
from step_2 import UNet, CompositeLoss

mpl.rcParams['text.usetex'] = False

def evaluate_benchmark(model, loader, device):
    model.eval()
    true_ysz_list = []
    pred_ysz_list = []
    mass_proxy_list = []
    with torch.no_grad():
        for x, noise_vars, y in loader:
            x = x.to(device)
            noise_vars = torch.log10(noise_vars + 1e-8).to(device)
            pred = model(x, noise_vars)
            pred_np = pred.cpu().numpy()[:, 0, :, :] / 1e6
            y_np = y.cpu().numpy()
            for i in range(y_np.shape[0]):
                t_map = y_np[i]
                p_map = pred_np[i]
                peak_idx = np.unravel_index(np.argmax(t_map), t_map.shape)
                mass_proxy = t_map[peak_idx]
                Y, X = np.ogrid[:256, :256]
                dist_sq = (Y - peak_idx[0])**2 + (X - peak_idx[1])**2
                mask = dist_sq <= (4.27)**2
                true_ysz = np.sum(t_map[mask])
                pred_ysz = np.sum(p_map[mask])
                true_ysz_list.append(true_ysz)
                pred_ysz_list.append(pred_ysz)
                mass_proxy_list.append(mass_proxy)
    true_ysz_arr = np.array(true_ysz_list)
    pred_ysz_arr = np.array(pred_ysz_list)
    mass_proxy_arr = np.array(mass_proxy_list)
    valid = (true_ysz_arr > 0) & (pred_ysz_arr > 0) & (mass_proxy_arr > 0)
    print('Valid clusters for scatter calculation: ' + str(np.sum(valid)) + ' / ' + str(len(true_ysz_arr)))
    log_true = np.log10(true_ysz_arr[valid])
    log_pred = np.log10(pred_ysz_arr[valid])
    log_m = np.log10(mass_proxy_arr[valid])
    if len(log_m) > 2:
        p_true = np.polyfit(log_m, log_true, 1)
        scatter_true = np.std(log_true - np.polyval(p_true, log_m))
        p_pred = np.polyfit(log_m, log_pred, 1)
        scatter_pred = np.std(log_pred - np.polyval(p_pred, log_m))
        scatter_diff = np.std(log_pred - log_true)
    else:
        scatter_true = np.nan
        scatter_pred = np.nan
        scatter_diff = np.nan
    return scatter_true, scatter_pred, scatter_diff, true_ysz_arr, pred_ysz_arr, mass_proxy_arr

def train_stage(model, loader, epochs, device):
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion = CompositeLoss(max_lambda_spec=0.1, total_epochs=epochs)
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        valid_batches = 0
        for x, noise_vars, y in loader:
            x = x.to(device)
            noise_vars = torch.log10(noise_vars + 1e-8).to(device)
            y = y.unsqueeze(1).to(device) * 1e6
            optimizer.zero_grad()
            pred = model(x, noise_vars)
            loss, l1, spec, lambda_spec = criterion(pred, y, epoch)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            valid_batches += 1
        if valid_batches > 0:
            train_loss /= valid_batches
    print('Final Training Loss: ' + str(round(train_loss, 4)))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    with open('data/splits.json', 'r') as f:
        splits = json.load(f)
    train_idx = np.array(splits['train'])
    val_idx = np.array(splits['val'])
    test_idx = np.array(splits['test'])
    tsz = np.load('/home/node/data/compsep_data/cut_maps/tsz.npy', mmap_mode='r')
    tsz_intensity = np.mean(np.abs(tsz), axis=(1, 2))
    train_intensities = tsz_intensity[train_idx]
    sorted_train_indices = train_idx[np.argsort(train_intensities)[::-1]]
    top20_train_idx = sorted_train_indices[:int(0.2 * len(train_idx))]
    val_test_idx = np.concatenate([val_idx, test_idx])
    val_test_intensities = tsz_intensity[val_test_idx]
    sorted_val_test = val_test_idx[np.argsort(val_test_intensities)[::-1]]
    benchmark_idx = sorted_val_test[:int(0.1 * len(val_test_idx))]
    top20_train_dataset = CompSepDataset(top20_train_idx.tolist(), split='train')
    full_train_dataset = CompSepDataset(train_idx.tolist(), split='train')
    benchmark_dataset = CompSepDataset(benchmark_idx.tolist(), split='val')
    top20_train_loader = DataLoader(top20_train_dataset, batch_size=16, shuffle=True, num_workers=0)
    full_train_loader = DataLoader(full_train_dataset, batch_size=16, shuffle=True, num_workers=0)
    benchmark_loader = DataLoader(benchmark_dataset, batch_size=16, shuffle=False, num_workers=0)
    model = UNet(in_channels=6, out_channels=1, cond_dim=6, features=[64, 128, 256, 512]).to(device)
    model.load_state_dict(torch.load('data/sr_dae_model.pth', map_location=device))
    print('\n--- Initial Evaluation ---')
    sc_true_init, sc_pred_init, sc_diff_init, t_ysz_init, p_ysz_init, m_init = evaluate_benchmark(model, benchmark_loader, device)
    print('True Y_SZ-M Scatter: ' + str(round(sc_true_init, 4)))
    print('Pred Y_SZ-M Scatter: ' + str(round(sc_pred_init, 4)))
    print('Pred vs True Scatter: ' + str(round(sc_diff_init, 4)))
    print('\n--- Stage 1: Fine-tuning on Top 20% ---')
    train_stage(model, top20_train_loader, epochs=10, device=device)
    sc_true_s1, sc_pred_s1, sc_diff_s1, t_ysz_s1, p_ysz_s1, m_s1 = evaluate_benchmark(model, benchmark_loader, device)
    print('True Y_SZ-M Scatter: ' + str(round(sc_true_s1, 4)))
    print('Pred Y_SZ-M Scatter: ' + str(round(sc_pred_s1, 4)))
    print('Pred vs True Scatter: ' + str(round(sc_diff_s1, 4)))
    print('\n--- Stage 2: Fine-tuning on Full Dataset ---')
    train_stage(model, full_train_loader, epochs=10, device=device)
    sc_true_s2, sc_pred_s2, sc_diff_s2, t_ysz_s2, p_ysz_s2, m_s2 = evaluate_benchmark(model, benchmark_loader, device)
    print('True Y_SZ-M Scatter: ' + str(round(sc_true_s2, 4)))
    print('Pred Y_SZ-M Scatter: ' + str(round(sc_pred_s2, 4)))
    print('Pred vs True Scatter: ' + str(round(sc_diff_s2, 4)))
    metrics = {'initial': {'scatter_true': float(sc_true_init), 'scatter_pred': float(sc_pred_init), 'scatter_diff': float(sc_diff_init)}, 'stage1': {'scatter_true': float(sc_true_s1), 'scatter_pred': float(sc_pred_s1), 'scatter_diff': float(sc_diff_s1)}, 'stage2': {'scatter_true': float(sc_true_s2), 'scatter_pred': float(sc_pred_s2), 'scatter_diff': float(sc_diff_s2)}}
    with open('data/curriculum_metrics.json', 'w') as f:
        json.dump(metrics, f)
    print('\nMetrics saved to data/curriculum_metrics.json')
    torch.save(model.state_dict(), 'data/sr_dae_model_curriculum.pth')
    print('Curriculum model saved to data/sr_dae_model_curriculum.pth')
    stages = ['Initial', 'Stage 1 (Top 20%)', 'Stage 2 (Full)']
    scatters = [sc_pred_init, sc_pred_s1, sc_pred_s2]
    plt.figure(figsize=(8, 6))
    plt.plot(stages, scatters, marker='o', linestyle='-', color='b', label='Reconstructed Y_SZ-M Scatter')
    plt.axhline(sc_true_init, color='r', linestyle='--', label='True Y_SZ-M Scatter')
    plt.ylabel('Scatter (std of log residuals)')
    plt.title('Change in Y_SZ-M Scatter across Curriculum Stages')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'data/curriculum_scatter_1_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)

if __name__ == '__main__':
    main()