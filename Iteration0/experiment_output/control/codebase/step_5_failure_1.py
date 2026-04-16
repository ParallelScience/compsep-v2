# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.abspath('codebase'))
sys.path.insert(0, '/home/node/data/compsep_data/')
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import utils
from step_1 import FlamingoDataset
from step_2 import DoubleConv, GatedCrossAttention
from step_3 import compute_spectral_loss, compute_edge_loss, pytorch_powers
from step_4 import safe_powers
class SingleBranchUNet(nn.Module):
    def __init__(self, so_in_channels=6, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        in_c = so_in_channels
        for feature in features:
            self.encoder.append(DoubleConv(in_c, feature))
            in_c = feature
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feature in reversed(features[:-1]):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(feature * 2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.mc_dropout = False
    def forward(self, so_inputs, so_vars, cib_inputs=None):
        x = torch.cat([so_inputs, so_vars], dim=1)
        skips = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            skips.append(x)
            if i < len(self.encoder) - 1:
                x = self.pool(x)
        x = nn.functional.dropout2d(x, p=0.2, training=self.training or self.mc_dropout)
        skips = skips[::-1][1:]
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skips[i]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i](x)
            x = nn.functional.dropout2d(x, p=0.2, training=self.training or self.mc_dropout)
        return self.final_conv(x)
class DualBranchUNetMCDropout(nn.Module):
    def __init__(self, so_in_channels=6, cib_in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.so_encoder = nn.ModuleList()
        self.cib_encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        so_in = so_in_channels
        cib_in = cib_in_channels
        for feature in features:
            self.so_encoder.append(DoubleConv(so_in, feature))
            self.cib_encoder.append(DoubleConv(cib_in, feature))
            so_in = feature
            cib_in = feature
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.bottleneck_attn = GatedCrossAttention(features[-1])
        for feature in reversed(features[:-1]):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(feature * 2, feature))
            self.attentions.append(GatedCrossAttention(feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.mc_dropout = False
    def forward(self, so_inputs, so_vars, cib_inputs):
        so_x = torch.cat([so_inputs, so_vars], dim=1)
        cib_x = cib_inputs
        so_skips = []
        cib_skips = []
        for i in range(len(self.so_encoder)):
            so_x = self.so_encoder[i](so_x)
            cib_x = self.cib_encoder[i](cib_x)
            so_skips.append(so_x)
            cib_skips.append(cib_x)
            if i < len(self.so_encoder) - 1:
                so_x = self.pool(so_x)
                cib_x = self.pool(cib_x)
        so_feat = so_skips[-1]
        cib_feat = cib_skips[-1]
        x = self.bottleneck_attn(so_feat, cib_feat)
        x = nn.functional.dropout2d(x, p=0.2, training=self.training or self.mc_dropout)
        so_skips = so_skips[::-1][1:]
        cib_skips = cib_skips[::-1][1:]
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = so_skips[i]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i](x)
            cib_feat = cib_skips[i]
            x = self.attentions[i](x, cib_feat)
            x = nn.functional.dropout2d(x, p=0.2, training=self.training or self.mc_dropout)
        return self.final_conv(x)
def train_ablation_model(model, model_name, lambda_1, lambda_2, lambda_3, train_loader, val_loader, device):
    print('Training ' + model_name + '...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 30
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
    l1_loss_fn = nn.L1Loss()
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
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
            if lambda_2 > 0:
                loss_spec = compute_spectral_loss(pred, tsz_gt, ps=5.0)
            else:
                loss_spec = torch.tensor(0.0, device=device)
            loss_edge = compute_edge_loss(pred, tsz_gt)
            loss = lambda_1 * loss_l1 + lambda_2 * loss_spec + lambda_3 * loss_edge
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                so_inputs = batch['so_inputs'].to(device)
                so_vars = batch['so_vars'].to(device)
                cib_inputs = batch['cib_inputs'].to(device)
                tsz_gt = batch['tsz_gt'].to(device)
                pred = model(so_inputs, so_vars, cib_inputs)
                loss_l1 = l1_loss_fn(pred, tsz_gt)
                if lambda_2 > 0:
                    loss_spec = compute_spectral_loss(pred, tsz_gt, ps=5.0)
                else:
                    loss_spec = torch.tensor(0.0, device=device)
                loss_edge = compute_edge_loss(pred, tsz_gt)
                loss = lambda_1 * loss_l1 + lambda_2 * loss_spec + lambda_3 * loss_edge
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print('Epoch ' + str(epoch+1) + '/' + str(epochs) + ' - Train Loss: ' + str(round(train_loss, 4)) + ' - Val Loss: ' + str(round(val_loss, 4)))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'data/best_' + model_name + '.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered at epoch ' + str(epoch+1))
                break
    model.load_state_dict(torch.load('data/best_' + model_name + '.pth'))
    return model
def evaluate_model(model, test_loader, device, tsz_scale):
    model.eval()
    all_rmse = []
    all_ssim = []
    all_cl_pred = []
    all_cl_gt = []
    all_cl_cross = []
    with torch.no_grad():
        for batch in test_loader:
            so_inputs = batch['so_inputs'].to(device)
            so_vars = batch['so_vars'].to(device)
            cib_inputs = batch['cib_inputs'].to(device)
            tsz_gt_trans = batch['tsz_gt'].cpu().numpy()
            pred_trans = model(so_inputs, so_vars, cib_inputs).cpu().numpy()
            pred_phys = np.sinh(pred_trans) * tsz_scale
            gt_phys = np.sinh(tsz_gt_trans) * tsz_scale
            for i in range(pred_phys.shape[0]):
                p = np.nan_to_num(pred_phys[i, 0])
                g = np.nan_to_num(gt_phys[i, 0])
                p = p - np.mean(p)
                g = g - np.mean(g)
                rmse = np.sqrt(np.mean((p - g)**2))
                all_rmse.append(rmse)
                data_range = g.max() - g.min()
                if data_range == 0: data_range = 1e-9
                s = ssim(g, p, data_range=data_range)
                all_ssim.append(s)
                ell, cl_p = safe_powers(p, p, ps=5.0)
                _, cl_g = safe_powers(g, g, ps=5.0)
                _, cl_cross = safe_powers(p, g, ps=5.0)
                all_cl_pred.append(cl_p)
                all_cl_gt.append(cl_g)
                all_cl_cross.append(cl_cross)
    mean_rmse = np.mean(all_rmse)
    mean_ssim = np.mean(all_ssim)
    mean_cl_pred = np.mean(all_cl_pred, axis=0)
    mean_cl_gt = np.mean(all_cl_gt, axis=0)
    mean_cl_cross = np.mean(all_cl_cross, axis=0)
    r_ell = mean_cl_cross / np.sqrt(np.abs(mean_cl_pred * mean_cl_gt) + 1e-20)
    mean_r_ell = np.mean(r_ell)
    return mean_rmse, mean_ssim, mean_r_ell
def generate_mc_dropout_maps(model, test_loader, device, tsz_scale, num_passes=20):
    model.eval()
    model.mc_dropout = True
    batch = next(iter(test_loader))
    so_inputs = batch['so_inputs'].to(device)
    so_vars = batch['so_vars'].to(device)
    cib_inputs = batch['cib_inputs'].to(device)
    tsz_gt_trans = batch['tsz_gt'].cpu().numpy()
    preds = []
    with torch.no_grad():
        for _ in range(num_passes):
            pred_trans = model(so_inputs, so_vars, cib_inputs).cpu().numpy()
            pred_phys = np.sinh(pred_trans) * tsz_scale
            preds.append(pred_phys[0, 0])
    preds = np.array(preds)
    mean_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0)
    gt_phys = np.sinh(tsz_gt_trans[0, 0]) * tsz_scale
    model.mc_dropout = False
    return gt_phys, mean_pred, std_pred
def plot_uncertainty_maps(gt, mean_pred, std_pred):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    vmax_tsz = max(np.percentile(np.abs(gt), 99), np.percentile(np.abs(mean_pred), 99))
    if vmax_tsz == 0: vmax_tsz = 1e-9
    im0 = axs[0].imshow(gt - np.mean(gt), cmap='RdBu_r', vmin=-vmax_tsz, vmax=vmax_tsz)
    axs[0].set_title('Ground Truth tSZ')
    fig.colorbar(im0, ax=axs[0], label='Compton-y')
    im1 = axs[1].imshow(mean_pred - np.mean(mean_pred), cmap='RdBu_r', vmin=-vmax_tsz, vmax=vmax_tsz)
    axs[1].set_title('MC Dropout Mean Prediction')
    fig.colorbar(im1, ax=axs[1], label='Compton-y')
    vmax_std = np.percentile(std_pred, 99)
    if vmax_std == 0: vmax_std = 1e-9
    im2 = axs[2].imshow(std_pred, cmap='viridis', vmin=0, vmax=vmax_std)
    axs[2].set_title('MC Dropout Uncertainty (Std Dev)')
    fig.colorbar(im2, ax=axs[2], label='Compton-y Uncertainty')
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'data/uncertainty_maps_1_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)
if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    print('Initializing datasets...')
    train_dataset = FlamingoDataset(split='train')
    val_dataset = FlamingoDataset(split='val')
    test_dataset = FlamingoDataset(split='test')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    tsz_scale = test_dataset.tsz_scale
    print('\n--- Training no-CIB Ablation Model ---')
    model_no_cib = SingleBranchUNet().to(device)
    model_no_cib = train_ablation_model(model_no_cib, 'no_cib', 1.0, 0.1, 0.5, train_loader, val_loader, device)
    print('\n--- Training no-spectral-loss Ablation Model ---')
    model_no_spec = DualBranchUNetMCDropout().to(device)
    model_no_spec = train_ablation_model(model_no_spec, 'no_spec', 1.0, 0.0, 0.5, train_loader, val_loader, device)
    print('\n--- Loading Main Model ---')
    model_main = DualBranchUNetMCDropout().to(device)
    model_main.load_state_dict(torch.load('data/best_model.pth', map_location=device))
    print('\n--- Evaluating Models on Test Set ---')
    metrics = {}
    print('Evaluating Main Model...')
    metrics['Main Model'] = evaluate_model(model_main, test_loader, device, tsz_scale)
    print('Evaluating no-CIB Model...')
    metrics['no-CIB'] = evaluate_model(model_no_cib, test_loader, device, tsz_scale)
    print('Evaluating no-spectral-loss Model...')
    metrics['no-spectral-loss'] = evaluate_model(model_no_spec, test_loader, device, tsz_scale)
    print('\n' + '='*60)
    print('Model                | RMSE         | SSIM       | Mean r_ell')
    print('-' * 60)
    for name, (rmse, ssim_val, r_ell) in metrics.items():
        name_str = name.ljust(20)
        rmse_str = str(round(rmse, 6)).ljust(12)
        ssim_str = str(round(ssim_val, 4)).ljust(10)
        rell_str = str(round(r_ell, 4)).ljust(10)
        print(name_str + ' | ' + rmse_str + ' | ' + ssim_str + ' | ' + rell_str)
    print('='*60 + '\n')
    print('Generating MC Dropout Uncertainty Maps (20 passes)...')
    gt, mean_pred, std_pred = generate_mc_dropout_maps(model_main, test_loader, device, tsz_scale, num_passes=20)
    plot_uncertainty_maps(gt, mean_pred, std_pred)
    print('Ablation Studies and Uncertainty Quantification completed successfully.')