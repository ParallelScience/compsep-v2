# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from step_1 import FLAMINGODataset
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)
class GatedCrossAttention(nn.Module):
    def __init__(self, primary_channels, aux_channels):
        super().__init__()
        self.attn = nn.Sequential(nn.Conv2d(primary_channels + aux_channels, primary_channels, 1), nn.Sigmoid())
        self.v = nn.Conv2d(aux_channels, primary_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, primary, aux):
        attn_map = self.attn(torch.cat([primary, aux], dim=1))
        aux_v = self.v(aux)
        return primary + self.gamma * (attn_map * aux_v)
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
class SRDAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1_p = DoubleConv(3, 64)
        self.enc2_p = DoubleConv(64, 128)
        self.enc3_p = DoubleConv(128, 256)
        self.enc4_p = DoubleConv(256, 512)
        self.enc1_a = DoubleConv(3, 64)
        self.enc2_a = DoubleConv(64, 128)
        self.enc3_a = DoubleConv(128, 256)
        self.enc4_a = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.gca1 = GatedCrossAttention(64, 64)
        self.gca2 = GatedCrossAttention(128, 128)
        self.gca3 = GatedCrossAttention(256, 256)
        self.gca4 = GatedCrossAttention(512, 512)
        self.bot_p = DoubleConv(512, 1024)
        self.bot_a = DoubleConv(512, 1024)
        self.gca_bot = GatedCrossAttention(1024, 1024)
        self.dec4 = UNetDecoderBlock(1024, 512)
        self.dec3 = UNetDecoderBlock(512, 256)
        self.dec2 = UNetDecoderBlock(256, 128)
        self.dec1 = UNetDecoderBlock(128, 64)
        self.final = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        p = x[:, :3, :, :]
        a = x[:, 3:, :, :]
        p1 = self.enc1_p(p)
        a1 = self.enc1_a(a)
        p1 = self.gca1(p1, a1)
        p2 = self.enc2_p(self.pool(p1))
        a2 = self.enc2_a(self.pool(a1))
        p2 = self.gca2(p2, a2)
        p3 = self.enc3_p(self.pool(p2))
        a3 = self.enc3_a(self.pool(a2))
        p3 = self.gca3(p3, a3)
        p4 = self.enc4_p(self.pool(p3))
        a4 = self.enc4_a(self.pool(a3))
        p4 = self.gca4(p4, a4)
        bot_p = self.bot_p(self.pool(p4))
        bot_a = self.bot_a(self.pool(a4))
        bot = self.gca_bot(bot_p, bot_a)
        d4 = self.dec4(bot, p4)
        d3 = self.dec3(d4, p3)
        d2 = self.dec2(d3, p2)
        d1 = self.dec1(d2, p1)
        return self.final(d1)
def get_ell_2d(npix=256, pixel_size_arcmin=1.171875):
    dx = pixel_size_arcmin * np.pi / (180 * 60)
    kx = torch.fft.fftfreq(npix, d=dx)
    ky = torch.fft.fftfreq(npix, d=dx)
    k = torch.sqrt(kx[None, :]**2 + ky[:, None]**2)
    ell = 2 * np.pi * k
    return ell
def power_spectrum_loss(y_pred, y_true, ell_min=1000, ell_max=5000):
    B, C, H, W = y_pred.shape
    ell = get_ell_2d(H).to(y_pred.device)
    win_y = torch.hann_window(H).view(1, 1, H, 1).to(y_pred.device)
    win_x = torch.hann_window(W).view(1, 1, 1, W).to(y_pred.device)
    window = win_y * win_x
    fft_pred = torch.fft.fft2(y_pred * window)
    fft_true = torch.fft.fft2(y_true * window)
    ps_pred = torch.abs(fft_pred)**2
    ps_true = torch.abs(fft_true)**2
    mask = (ell >= ell_min) & (ell <= ell_max)
    loss = torch.mean(((ps_pred[:, :, mask] - ps_true[:, :, mask]) / (ps_true[:, :, mask] + 1.0))**2)
    return loss
def correlation_loss(y_pred, y_true, x):
    R = y_true - y_pred
    cib_proxies = x[:, 3:, :, :]
    R_flat = R.view(R.shape[0], 1, -1)
    C_flat = cib_proxies.view(cib_proxies.shape[0], 3, -1)
    R_flat = R_flat - R_flat.mean(dim=2, keepdim=True)
    C_flat = C_flat - C_flat.mean(dim=2, keepdim=True)
    R_var = torch.sum(R_flat**2, dim=2, keepdim=True)
    C_var = torch.sum(C_flat**2, dim=2, keepdim=True)
    cov = torch.sum(R_flat * C_flat, dim=2, keepdim=True)
    corr = cov / torch.sqrt(R_var * C_var + 1e-8)
    loss = torch.mean(corr**2)
    return loss
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = FLAMINGODataset('train', splits_file='data/splits.npz', stats_file='data/normalization_stats.npz', augment=True)
    val_dataset = FLAMINGODataset('val', splits_file='data/splits.npz', stats_file='data/normalization_stats.npz', augment=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    model = SRDAE().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)
    epochs = 100
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    lambda_1 = 0.1
    lambda_2 = 0.1
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            l1 = F.l1_loss(y_pred, y)
            l_spec = power_spectrum_loss(y_pred, y)
            l_corr = correlation_loss(y_pred, y, x)
            loss = l1 + lambda_1 * l_spec + lambda_2 * l_corr
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                l1 = F.l1_loss(y_pred, y)
                l_spec = power_spectrum_loss(y_pred, y)
                l_corr = correlation_loss(y_pred, y, x)
                loss = l1 + lambda_1 * l_spec + lambda_2 * l_corr
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'data/best_srdae.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    train_model()