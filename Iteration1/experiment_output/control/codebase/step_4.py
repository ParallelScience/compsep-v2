# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

class HaarWaveletLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        lh = torch.tensor([[-1., 1.], [-1., 1.]]) / 2.0
        hl = torch.tensor([[-1., -1.], [1., 1.]]) / 2.0
        hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        self.filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1).to(device)
    def forward(self, pred, target):
        pred_coeffs = F.conv2d(pred, self.filters, stride=2)
        target_coeffs = F.conv2d(target, self.filters, stride=2)
        loss = F.l1_loss(pred_coeffs[:, 1:], target_coeffs[:, 1:])
        return loss

class PseudoClLoss(nn.Module):
    def __init__(self, n_pixels=256, patch_size_deg=5.0, device='cuda'):
        super().__init__()
        self.n_pixels = n_pixels
        self.patch_size_rad = patch_size_deg * np.pi / 180.0
        self.pix_size_rad = self.patch_size_rad / n_pixels
        window = self._create_tukey_window(n_pixels, alpha=0.2).to(device)
        self.window = window.view(1, 1, n_pixels, n_pixels)
        freqs = torch.fft.fftfreq(n_pixels, d=self.pix_size_rad).to(device)
        kx, ky = torch.meshgrid(freqs, freqs, indexing='ij')
        k_rad = torch.sqrt(kx**2 + ky**2)
        self.ell = k_rad * 2 * np.pi
        self.ell_mask = (self.ell > 1000).unsqueeze(0).unsqueeze(0)
    def _create_tukey_window(self, n, alpha=0.2):
        w = torch.ones(n)
        n_taper = int(alpha * n / 2)
        x = torch.linspace(0, 1, n_taper)
        taper = 0.5 * (1 - torch.cos(np.pi * x))
        w[:n_taper] = taper
        w[-n_taper:] = torch.flip(taper, dims=[0])
        w2d = w.unsqueeze(0) * w.unsqueeze(1)
        return w2d
    def forward(self, pred, target):
        p_win = pred * self.window
        t_win = target * self.window
        p_fft = torch.fft.fft2(p_win)
        t_fft = torch.fft.fft2(t_win)
        p_ps = torch.abs(p_fft)**2
        t_ps = torch.abs(t_fft)**2
        p_ps = p_ps / (self.n_pixels**2)
        t_ps = t_ps / (self.n_pixels**2)
        diff = torch.abs(p_ps - t_ps) * self.ell_mask
        loss = diff.sum() / (self.ell_mask.sum() * pred.size(0) + 1e-8)
        return loss

class FluxLoss(nn.Module):
    def __init__(self, kernel_size=15, sigma=3.0, device='cuda'):
        super().__init__()
        self.kernel_size = kernel_size
        x = torch.arange(kernel_size).float() - kernel_size // 2
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.kernel = kernel.view(1, 1, kernel_size, kernel_size).to(device)
        self.padding = kernel_size // 2
    def forward(self, pred, target):
        global_loss = torch.abs(pred.sum(dim=(1,2,3)) - target.sum(dim=(1,2,3))).mean()
        p_blur = F.conv2d(pred, self.kernel, padding=self.padding)
        t_blur = F.conv2d(target, self.kernel, padding=self.padding)
        local_loss = F.l1_loss(p_blur, t_blur)
        return global_loss + local_loss

class CompositeLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.1, lambda3=0.01, lambda4=0.0, device='cuda'):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.l1 = nn.L1Loss()
        self.wavelet = HaarWaveletLoss(device=device)
        self.pseudo_cl = PseudoClLoss(device=device)
        self.flux = FluxLoss(device=device)
    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_wav = self.wavelet(pred, target)
        loss_cl = self.pseudo_cl(pred, target)
        loss_flux = self.flux(pred, target)
        total_loss = (self.lambda1 * loss_l1 + self.lambda2 * loss_wav + self.lambda3 * loss_cl + self.lambda4 * loss_flux)
        return total_loss, loss_l1, loss_wav, loss_cl, loss_flux

def get_lambda4_schedule(epochs=50, start_epoch=10, end_epoch=30, max_val=0.1):
    schedule = []
    for epoch in range(epochs):
        if epoch < start_epoch:
            val = 0.0
        elif epoch >= end_epoch:
            val = max_val
        else:
            val = max_val * (epoch - start_epoch) / float(end_epoch - start_epoch)
        schedule.append(val)
    return schedule

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    comp_loss = CompositeLoss(device=device)
    x = torch.randn(2, 1, 256, 256, device=device, requires_grad=True)
    y = torch.randn(2, 1, 256, 256, device=device)
    loss, l1, wav, cl, flux = comp_loss(x, y)
    loss.backward()
    print('Loss components computed and backward pass verified.')
    schedule = get_lambda4_schedule()
    df = pd.DataFrame({'epoch': range(len(schedule)), 'lambda4': schedule})
    df.to_csv('data/lambda_schedule.csv', index=False)
    print('Saved lambda schedule to data/lambda_schedule.csv')