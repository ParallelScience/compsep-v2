# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
os.environ['OMP_NUM_THREADS'] = '1'
from step_1 import FlamingoDataset
from step_2 import DualBranchUNet
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
def pytorch_powers(a, b, ps=5.0, ell_n=199):
    B, C, H, W = a.shape
    res = (ps * np.pi / 180.0) / H
    wy = torch.hann_window(H, device=a.device).view(1, 1, H, 1)
    wx = torch.hann_window(W, device=a.device).view(1, 1, 1, W)
    window = wy * wx
    a_win = a * window
    b_win = b * window
    A = torch.fft.fft2(a_win) * (res**2)
    B_fft = torch.fft.fft2(b_win) * (res**2)
    P = torch.real(A * torch.conj(B_fft)) / ((ps * np.pi / 180.0)**2)
    freq_y = torch.fft.fftfreq(H, d=res, device=a.device)
    freq_x = torch.fft.fftfreq(W, d=res, device=a.device)
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')
    ell = 2 * np.pi * torch.sqrt(fx**2 + fy**2)
    ell_flat = ell.view(-1)
    P_flat = P.view(B, -1)
    ell_bins = torch.linspace(100, 5000, steps=ell_n+1, device=a.device)
    bin_indices = torch.bucketize(ell_flat, ell_bins) - 1
    mask = (bin_indices >= 0) & (bin_indices < ell_n)
    bin_indices = bin_indices[mask]
    P_flat = P_flat[:, mask]
    binned_P = torch.zeros(B, ell_n, device=a.device)
    binned_P.scatter_add_(1, bin_indices.unsqueeze(0).expand(B, -1), P_flat)
    counts = torch.zeros(ell_n, device=a.device)
    counts.scatter_add_(0, bin_indices, torch.ones_like(bin_indices, dtype=torch.float32))
    binned_P = binned_P / counts.clamp(min=1).unsqueeze(0)
    return binned_P
def compute_spectral_loss(pred, target, ps=5.0):
    pred_cpu = pred.cpu()
    target_cpu = target.cpu()
    try:
        p0 = pred_cpu[0, 0]
        t0 = target_cpu[0, 0]
        ell, cl_pred = utils.powers(p0, p0, ps=ps)
        if not isinstance(cl_pred, torch.Tensor) or not cl_pred.requires_grad:
            raise ValueError('utils.powers broke the computation graph.')
        B = pred_cpu.shape[0]
        loss = 0.0
        for i in range(B):
            p = pred_cpu[i, 0]
            t = target_cpu[i, 0]
            _, cl_p = utils.powers(p, p, ps=ps)
            _, cl_t = utils.powers(t, t, ps=ps)
            loss += torch.mean(torch.abs(torch.log(cl_p + 1e-8) - torch.log(cl_t + 1e-8)))
        return (loss / B).to(pred.device)
    except Exception:
        cl_p = pytorch_powers(pred_cpu, pred_cpu, ps=ps)
        cl_t = pytorch_powers(target_cpu, target_cpu, ps=ps)
        loss = torch.mean(torch.abs(torch.log(cl_p + 1e-8) - torch.log(cl_t + 1e-8)))
        return loss.to(pred.device)
def sobel_filter(x):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    grad_x = nn.functional.conv2d(x, sobel_x, padding=1)
    grad_y = nn.functional.conv2d(x, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    return grad_mag
def compute_edge_loss(pred, target):
    pred_grad = sobel_filter(pred)
    target_grad = sobel_filter(target)
    return nn.functional.l1_loss(pred_grad, target_grad)
if __name__ == '__main__':
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    print('Initializing datasets...')
    train_dataset = FlamingoDataset(split='train')
    val_dataset = FlamingoDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    model = DualBranchUNet().to(device)
    lambda_1 = 1.0
    lambda_2 = 0.1
    lambda_3 = 0.5
    print('Loss weights: lambda_1 (L1) = ' + str(lambda_1) + ', lambda_2 (Spectral) = ' + str(lambda_2) + ', lambda_3 (Edge) = ' + str(lambda_3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 30
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
    l1_loss_fn = nn.L1Loss()
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    print('\nStarting training loop...')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            so_inputs = batch['so_inputs'].to(device)
            so_vars = batch['so_vars'].to(device)
            cib_inputs = batch['cib_inputs'].to(device)
            tsz_gt = batch['tsz_gt'].to(device)
            optimizer.zero_grad()
            pred = model(so_inputs, so_vars, cib_inputs)
            loss_l1 = l1_loss_fn(pred, tsz_gt)
            loss_spec = compute_spectral_loss(pred, tsz_gt, ps=5.0)
            loss_edge = compute_edge_loss(pred, tsz_gt)
            loss = lambda_1 * loss_l1 + lambda_2 * loss_spec + lambda_3 * loss_edge
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        val_spec = 0.0
        val_edge = 0.0
        with torch.no_grad():
            for batch in val_loader:
                so_inputs = batch['so_inputs'].to(device)
                so_vars = batch['so_vars'].to(device)
                cib_inputs = batch['cib_inputs'].to(device)
                tsz_gt = batch['tsz_gt'].to(device)
                pred = model(so_inputs, so_vars, cib_inputs)
                loss_l1 = l1_loss_fn(pred, tsz_gt)
                loss_spec = compute_spectral_loss(pred, tsz_gt, ps=5.0)
                loss_edge = compute_edge_loss(pred, tsz_gt)
                loss = lambda_1 * loss_l1 + lambda_2 * loss_spec + lambda_3 * loss_edge
                val_loss += loss.item()
                val_l1 += loss_l1.item()
                val_spec += loss_spec.item()
                val_edge += loss_edge.item()
        val_loss /= len(val_loader)
        val_l1 /= len(val_loader)
        val_spec /= len(val_loader)
        val_edge /= len(val_loader)
        print('Epoch ' + str(epoch+1) + '/' + str(epochs) + ' - Train Loss: ' + str(round(train_loss, 4)) + ' - Val Loss: ' + str(round(val_loss, 4)) + ' (L1: ' + str(round(val_l1, 4)) + ', Spec: ' + str(round(val_spec, 4)) + ', Edge: ' + str(round(val_edge, 4)) + ')')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'data/best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered at epoch ' + str(epoch+1))
                break
    print('\nTraining completed in ' + str(round(time.time() - start_time, 2)) + ' seconds.')
    print('Final Validation Loss Breakdown - L1: ' + str(round(val_l1, 4)) + ', Spectral: ' + str(round(val_spec, 4)) + ', Edge: ' + str(round(val_edge, 4)))
    print('Best model saved to data/best_model.pth')