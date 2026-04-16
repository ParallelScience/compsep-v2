# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['OMP_NUM_THREADS'] = '16'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from step_4 import SR_DAE, FocalL1Loss, SpectralLoss
from step_5 import TSZDataset
class LaplacianLoss(nn.Module):
    def __init__(self, levels=3, weight=1.0):
        super().__init__()
        self.levels = levels
        self.weight = weight
    def forward(self, pred, target):
        loss = 0.0
        p, t = pred, target
        for i in range(self.levels):
            p_down = F.avg_pool2d(p, kernel_size=2, stride=2)
            t_down = F.avg_pool2d(t, kernel_size=2, stride=2)
            p_up = F.interpolate(p_down, size=p.shape[-2:], mode='bilinear', align_corners=False)
            t_up = F.interpolate(t_down, size=t.shape[-2:], mode='bilinear', align_corners=False)
            p_detail = p - p_up
            t_detail = t - t_up
            loss += F.l1_loss(p_detail, t_detail)
            p, t = p_down, t_down
        return loss * self.weight
class FineTuneLoss(nn.Module):
    def __init__(self, mask_weight=1e3, gamma=2.0, spectral_weight=0.1, wavelet_weight=1.0):
        super().__init__()
        self.focal_l1 = FocalL1Loss(mask_weight, gamma)
        self.spectral = SpectralLoss(N=256, ps=5.0, ell_n=199)
        self.wavelet = LaplacianLoss(levels=3, weight=wavelet_weight)
        self.spectral_weight = spectral_weight
        self.wavelet_weight = wavelet_weight
    def forward(self, pred, target, mask):
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        l1_loss = self.focal_l1(pred, target, mask)
        pred_clamped = torch.clamp(pred, min=-10.0, max=10.0)
        spec_loss = self.spectral(pred_clamped, target)
        pred_4d = pred.unsqueeze(1)
        target_4d = target.unsqueeze(1)
        wav_loss = self.wavelet(pred_4d, target_4d)
        total_loss = l1_loss + self.spectral_weight * spec_loss + wav_loss
        return total_loss, l1_loss, spec_loss, wav_loss
def finetune_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    train_dataset = TSZDataset('data/train_features.npy', 'data/train_targets.npy', 'data/train_masks.npy')
    val_dataset = TSZDataset('data/val_features.npy', 'data/val_targets.npy', 'data/val_masks.npy')
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print("Initializing model and loading best checkpoint...")
    model = SR_DAE(in_channels=6, out_channels=1, init_features=32).to(device)
    model.load_state_dict(torch.load('data/best_model.pth', map_location=device))
    epochs = 15
    lr = 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    criterion = FineTuneLoss(mask_weight=1e3, gamma=2.0, spectral_weight=0.1, wavelet_weight=1.0).to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    best_epoch = -1
    print("Starting fine-tuning for " + str(epochs) + " epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss, train_l1, train_spec, train_wav = 0.0, 0.0, 0.0, 0.0
        valid_batches = 0
        for batch in train_loader:
            inputs, targets, masks = [b.to(device, non_blocking=True) for b in batch]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss, l1, spec, wav = criterion(outputs, targets, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            train_l1 += l1.item()
            train_spec += spec.item()
            train_wav += wav.item()
            valid_batches += 1
        if valid_batches > 0:
            train_loss /= valid_batches
            train_l1 /= valid_batches
            train_spec /= valid_batches
            train_wav /= valid_batches
        model.eval()
        val_loss, val_l1, val_spec, val_wav = 0.0, 0.0, 0.0, 0.0
        val_valid_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, masks = [b.to(device, non_blocking=True) for b in batch]
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss, l1, spec, wav = criterion(outputs, targets, masks)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_l1 += l1.item()
                    val_spec += spec.item()
                    val_wav += wav.item()
                    val_valid_batches += 1
        if val_valid_batches > 0:
            val_loss /= val_valid_batches
            val_l1 /= val_valid_batches
            val_spec /= val_valid_batches
            val_wav /= val_valid_batches
        if val_loss < best_val_loss and val_valid_batches > 0:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'data/finetuned_model.pth')
        print("Epoch " + str(epoch+1) + "/" + str(epochs) + " - Train Loss: " + str(round(train_loss, 4)) + " (L1: " + str(round(train_l1, 4)) + ", Spec: " + str(round(train_spec, 4)) + ", Wav: " + str(round(train_wav, 4)) + ")" + " - Val Loss: " + str(round(val_loss, 4)) + " (L1: " + str(round(val_l1, 4)) + ", Spec: " + str(round(val_spec, 4)) + ", Wav: " + str(round(val_wav, 4)) + ")")
    print("Fine-tuning completed.")
    print("Best Validation Loss: " + str(round(best_val_loss, 4)) + " achieved at epoch " + str(best_epoch))
    print("Model saved to data/finetuned_model.pth")
if __name__ == '__main__':
    finetune_model()