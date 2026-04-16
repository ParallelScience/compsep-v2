# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.att = AttentionBlock(F_g=out_ch, F_l=out_ch, F_int=out_ch // 2)
        self.conv = ConvBlock(out_ch * 2, out_ch)
    def forward(self, g, x):
        g = self.up(g)
        x = self.att(g=g, x=x)
        out = torch.cat([x, g], dim=1)
        out = self.conv(out)
        return out
class SR_DAE(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, init_features=32):
        super().__init__()
        features = init_features
        self.e1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e3 = ConvBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e4 = ConvBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(features * 8, features * 16)
        self.d4 = UpBlock(features * 16, features * 8)
        self.d3 = UpBlock(features * 8, features * 4)
        self.d2 = UpBlock(features * 4, features * 2)
        self.d1 = UpBlock(features * 2, features)
        self.out_conv = nn.Conv2d(features, out_channels, kernel_size=1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool1(e1))
        e3 = self.e3(self.pool2(e2))
        e4 = self.e4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.d4(b, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)
        out = self.out_conv(d1)
        return out
class FocalL1Loss(nn.Module):
    def __init__(self, mask_weight=1e3, gamma=2.0):
        super().__init__()
        self.mask_weight = mask_weight
        self.gamma = gamma
    def forward(self, pred, target, mask):
        l1 = torch.abs(pred - target)
        focal_weight = (1.0 - torch.exp(-l1)) ** self.gamma
        loss = l1 * focal_weight
        weighted_loss = loss * (1.0 + self.mask_weight * mask)
        return weighted_loss.mean()
def get_ell_bin_indices(N=256, ps=5.0, ell_n=199):
    L_rad = ps * np.pi / 180.0
    kx = torch.fft.fftfreq(N, d=L_rad/N) * 2 * np.pi
    ky = torch.fft.fftfreq(N, d=L_rad/N) * 2 * np.pi
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    ell = torch.sqrt(KX**2 + KY**2)
    ell_max = torch.max(ell).item()
    ell_bins = torch.linspace(0, ell_max, ell_n + 1)
    bin_indices = torch.bucketize(ell, ell_bins) - 1
    bin_indices = torch.clamp(bin_indices, 0, ell_n - 1)
    return bin_indices
class SpectralLoss(nn.Module):
    def __init__(self, N=256, ps=5.0, ell_n=199):
        super().__init__()
        self.N = N
        self.L_rad = ps * np.pi / 180.0
        self.ell_n = ell_n
        self.register_buffer('bin_indices', get_ell_bin_indices(N, ps, ell_n))
        bin_counts = torch.bincount(self.bin_indices.flatten(), minlength=ell_n)
        self.register_buffer('bin_counts', bin_counts)
    def compute_power(self, x):
        B = x.shape[0]
        X_fft = torch.fft.fft2(x) * (self.L_rad / self.N)**2
        P2D = torch.abs(X_fft)**2 / (self.L_rad**2)
        P2D_flat = P2D.view(B, -1)
        bin_indices_flat = self.bin_indices.view(-1).unsqueeze(0).expand(B, -1)
        P1D = torch.zeros(B, self.ell_n, device=x.device)
        P1D.scatter_add_(1, bin_indices_flat, P2D_flat)
        counts = self.bin_counts.unsqueeze(0).float()
        counts = torch.clamp(counts, min=1.0)
        P1D = P1D / counts
        return P1D
    def forward(self, pred, target):
        P1D_pred = self.compute_power(pred)
        P1D_target = self.compute_power(target)
        loss = torch.mean((torch.log10(P1D_pred + 1e-8) - torch.log10(P1D_target + 1e-8))**2)
        return loss
class CompositeLoss(nn.Module):
    def __init__(self, mask_weight=1e3, gamma=2.0, spectral_weight=0.1):
        super().__init__()
        self.focal_l1 = FocalL1Loss(mask_weight, gamma)
        self.spectral = SpectralLoss(N=256, ps=5.0, ell_n=199)
        self.spectral_weight = spectral_weight
    def forward(self, pred, target, mask):
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        l1_loss = self.focal_l1(pred, target, mask)
        spec_loss = self.spectral(pred, target)
        total_loss = l1_loss + self.spectral_weight * spec_loss
        return total_loss, l1_loss, spec_loss
def configure_training(model, lr=1e-3, weight_decay=1e-4, epochs=50, steps_per_epoch=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
    criterion = CompositeLoss(mask_weight=1e3, gamma=2.0, spectral_weight=0.1)
    return optimizer, scheduler, criterion
def train_step(model, batch, criterion, optimizer, scaler=None, clip_val=1.0):
    inputs, targets, masks = batch
    optimizer.zero_grad()
    if scaler is not None:
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss, l1, spec = criterion(outputs, targets, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(inputs)
        loss, l1, spec = criterion(outputs, targets, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
    return loss.item(), l1.item(), spec.item()
if __name__ == '__main__':
    model = SR_DAE(in_channels=6, out_channels=1, init_features=32)
    criterion = CompositeLoss(mask_weight=1e3, gamma=2.0, spectral_weight=0.1)
    optimizer, scheduler, _ = configure_training(model, epochs=1, steps_per_epoch=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    dummy_input = torch.randn(2, 6, 256, 256, device=device)
    dummy_target = torch.randn(2, 256, 256, device=device)
    dummy_mask = torch.randint(0, 2, (2, 256, 256), device=device).float()
    loss, l1, spec = train_step(model, (dummy_input, dummy_target, dummy_mask), criterion, optimizer, scaler=None, clip_val=1.0)
    print("Dummy pass successful.")
    print("Total Loss: " + str(round(loss, 4)) + ", L1 Loss: " + str(round(l1, 4)) + ", Spectral Loss: " + str(round(spec, 4)))