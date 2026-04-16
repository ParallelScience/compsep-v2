# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['OMP_NUM_THREADS'] = '16'
sys.path.insert(0, os.path.abspath("codebase"))
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from step_4 import SR_DAE, CompositeLoss, configure_training, train_step
class TSZDataset(Dataset):
    def __init__(self, features_path, targets_path, masks_path):
        self.features = torch.from_numpy(np.load(features_path)).float()
        self.targets = torch.from_numpy(np.load(targets_path)).float()
        self.masks = torch.from_numpy(np.load(masks_path)).float()
        self.length = self.features.shape[0]
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.masks[idx]
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    print("Loading datasets into memory...")
    train_dataset = TSZDataset('data/train_features.npy', 'data/train_targets.npy', 'data/train_masks.npy')
    val_dataset = TSZDataset('data/val_features.npy', 'data/val_targets.npy', 'data/val_masks.npy')
    print("Train dataset size: " + str(len(train_dataset)))
    print("Val dataset size: " + str(len(val_dataset)))
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print("Initializing model and training components...")
    model = SR_DAE(in_channels=6, out_channels=1, init_features=32).to(device)
    epochs = 30
    steps_per_epoch = len(train_loader)
    optimizer, scheduler, criterion = configure_training(model, lr=1e-3, weight_decay=1e-4, epochs=epochs, steps_per_epoch=steps_per_epoch)
    criterion = criterion.to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    best_epoch = -1
    print("Starting training for " + str(epochs) + " epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_spec = 0.0
        for batch in train_loader:
            inputs, targets, masks = [b.to(device, non_blocking=True) for b in batch]
            loss, l1, spec = train_step(model, (inputs, targets, masks), criterion, optimizer, scaler=scaler, clip_val=1.0)
            scheduler.step()
            train_loss += loss
            train_l1 += l1
            train_spec += spec
        train_loss /= len(train_loader)
        train_l1 /= len(train_loader)
        train_spec /= len(train_loader)
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        val_spec = 0.0
        val_sig_l1_sum = 0.0
        val_sig_count = 0
        val_null_l1_sum = 0.0
        val_null_count = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, masks = [b.to(device, non_blocking=True) for b in batch]
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss, l1, spec = criterion(outputs, targets, masks)
                val_loss += loss.item()
                val_l1 += l1.item()
                val_spec += spec.item()
                l1_err = torch.abs(outputs.squeeze(1) - targets.squeeze(1))
                m = masks.squeeze(1) > 0.5
                val_sig_l1_sum += l1_err[m].sum().item()
                val_sig_count += m.sum().item()
                val_null_l1_sum += l1_err[~m].sum().item()
                val_null_count += (~m).sum().item()
        val_loss /= len(val_loader)
        val_l1 /= len(val_loader)
        val_spec /= len(val_loader)
        val_sig_l1 = val_sig_l1_sum / val_sig_count if val_sig_count > 0 else 0.0
        val_null_l1 = val_null_l1_sum / val_null_count if val_null_count > 0 else 0.0
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'data/best_model.pth')
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print("Epoch " + str(epoch+1) + "/" + str(epochs) + " - Train Loss: " + str(round(train_loss, 4)) + " (L1: " + str(round(train_l1, 4)) + ", Spec: " + str(round(train_spec, 4)) + ")" + " - Val Loss: " + str(round(val_loss, 4)) + " (L1: " + str(round(val_l1, 4)) + ", Spec: " + str(round(val_spec, 4)) + ")" + " - Val Sig L1: " + str(round(val_sig_l1, 4)) + " - Val Null L1: " + str(round(val_null_l1, 4)))
    print("Training completed.")
    print("Best Validation Loss: " + str(round(best_val_loss, 4)) + " achieved at epoch " + str(best_epoch))
    print("Model saved to data/best_model.pth")