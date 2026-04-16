# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from step_1 import CompSepDataset
class FiLM(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, num_features), nn.ReLU(), nn.Linear(num_features, num_features * 2))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
    def forward(self, x, cond):
        params = self.mlp(cond)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1)
        return x * (1 + gamma) + beta
class AttentionGate(nn.Module):
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
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.film1 = FiLM(out_ch, cond_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.film2 = FiLM(out_ch, cond_dim)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, cond):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.film1(x, cond)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.film2(x, cond)
        x = self.relu(x)
        return x
class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, cond_dim=6, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        in_ch = in_channels
        for feature in features:
            self.downs.append(ConvBlock(in_ch, feature, cond_dim))
            in_ch = feature
        self.bottleneck = ConvBlock(features[-1], features[-1]*2, cond_dim)
        self.attention_gates = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.attention_gates.append(AttentionGate(F_g=feature, F_l=feature, F_int=feature//2))
            self.ups.append(ConvBlock(feature*2, feature, cond_dim))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x, cond):
        skip_connections = []
        for down in self.downs:
            x = down(x, cond)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x, cond)
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            skip_connection = self.attention_gates[i//2](g=x, x=skip_connection)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip, cond)
        return self.final_conv(x)
def compute_power_spectrum_2d(img, pixel_size_arcmin=1.17):
    B, C, H, W = img.shape
    fft2 = torch.fft.fft2(img)
    fft2_shifted = torch.fft.fftshift(fft2, dim=(-2, -1))
    power = torch.abs(fft2_shifted)**2
    center = H // 2
    y, x = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device), indexing='ij')
    r = torch.sqrt((x - center)**2 + (y - center)**2)
    max_r = int(min(H, W) / 2)
    r_int = r.long().flatten()
    mask = r_int < max_r
    r_int_masked = r_int[mask]
    power_flat = power.view(B, -1)[:, mask]
    ps1d = torch.zeros(B, max_r, device=img.device)
    counts = torch.zeros(max_r, device=img.device)
    counts.scatter_add_(0, r_int_masked, torch.ones_like(r_int_masked, dtype=torch.float32))
    r_int_masked_batch = r_int_masked.unsqueeze(0).expand(B, -1)
    ps1d.scatter_add_(1, r_int_masked_batch, power_flat)
    ps1d = ps1d / counts.clamp(min=1).unsqueeze(0)
    L_rad = H * pixel_size_arcmin / 60.0 * (torch.pi / 180.0)
    ell = 2 * torch.pi * torch.arange(max_r, device=img.device) / L_rad
    return ell, ps1d
def spectral_loss(pred, true, pixel_size_arcmin=1.17):
    ell, ps_pred = compute_power_spectrum_2d(pred, pixel_size_arcmin)
    _, ps_true = compute_power_spectrum_2d(true, pixel_size_arcmin)
    log_ps_pred = torch.log(ps_pred + 1e-12)
    log_ps_true = torch.log(ps_true + 1e-12)
    weight = (ell ** 2).unsqueeze(0)
    weight = weight / (weight.max() + 1e-12)
    loss = torch.mean(weight[:, 1:] * (log_ps_pred[:, 1:] - log_ps_true[:, 1:])**2)
    return loss
class CompositeLoss(nn.Module):
    def __init__(self, max_lambda_spec=0.1, total_epochs=60):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.max_lambda_spec = max_lambda_spec
        self.total_epochs = total_epochs
    def forward(self, pred, true, epoch):
        l1 = self.l1_loss(pred, true)
        spec = spectral_loss(pred, true)
        lambda_spec = self.max_lambda_spec * (epoch / max(1, self.total_epochs - 1))
        total_loss = l1 + lambda_spec * spec
        return total_loss, l1, spec, lambda_spec
if __name__ == '__main__':
    BATCH_SIZE = 32
    EPOCHS = 60
    LR = 2e-4
    MAX_LAMBDA_SPEC = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    with open('data/splits.json', 'r') as f:
        splits = json.load(f)
    train_idx = splits['train']
    val_idx = splits['val']
    print('Training samples: ' + str(len(train_idx)) + ', Validation samples: ' + str(len(val_idx)))
    train_dataset = CompSepDataset(train_idx, split='train')
    val_dataset = CompSepDataset(val_idx, split='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    model = UNet(in_channels=6, out_channels=1, cond_dim=6, features=[64, 128, 256, 512]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = CompositeLoss(max_lambda_spec=MAX_LAMBDA_SPEC, total_epochs=EPOCHS)
    history = {'train_loss': [], 'val_loss': [], 'train_l1': [], 'val_l1': [], 'train_spec': [], 'val_spec': []}
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_l1, train_spec = 0.0, 0.0, 0.0
        valid_batches = 0
        for x, noise_vars, y in train_loader:
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
            train_l1 += l1.item()
            train_spec += spec.item()
            valid_batches += 1
        if valid_batches > 0:
            train_loss /= valid_batches
            train_l1 /= valid_batches
            train_spec /= valid_batches
        model.eval()
        val_loss, val_l1, val_spec = 0.0, 0.0, 0.0
        val_valid_batches = 0
        with torch.no_grad():
            for x, noise_vars, y in val_loader:
                x = x.to(device)
                noise_vars = torch.log10(noise_vars + 1e-8).to(device)
                y = y.unsqueeze(1).to(device) * 1e6
                pred = model(x, noise_vars)
                loss, l1, spec, _ = criterion(pred, y, epoch)
                if torch.isnan(loss):
                    continue
                val_loss += loss.item()
                val_l1 += l1.item()
                val_spec += spec.item()
                val_valid_batches += 1
        if val_valid_batches > 0:
            val_loss /= val_valid_batches
            val_l1 /= val_valid_batches
            val_spec /= val_valid_batches
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_l1'].append(train_l1)
        history['val_l1'].append(val_l1)
        history['train_spec'].append(train_spec)
        history['val_spec'].append(val_spec)
        print('Epoch ' + str(epoch+1) + '/' + str(EPOCHS) + ' - Train Loss: ' + str(round(train_loss, 4)) + ' (L1: ' + str(round(train_l1, 4)) + ', Spec: ' + str(round(train_spec, 4)) + ') - Val Loss: ' + str(round(val_loss, 4)) + ' (L1: ' + str(round(val_l1, 4)) + ', Spec: ' + str(round(val_spec, 4)) + ') - Lambda: ' + str(round(lambda_spec, 4)))
    torch.save(model.state_dict(), 'data/sr_dae_model.pth')
    print('Model saved to data/sr_dae_model.pth')
    with open('data/training_history.json', 'w') as f:
        json.dump(history, f)
    print('Training history saved to data/training_history.json')