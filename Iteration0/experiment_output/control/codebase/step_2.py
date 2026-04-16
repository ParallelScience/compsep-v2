# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from step_1 import FlamingoDataset

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class GatedCrossAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.cib_transform = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, so_feat, cib_feat):
        gate_input = torch.cat([so_feat, cib_feat], dim=1)
        attention_weights = self.gate(gate_input)
        gated_cib = attention_weights * self.cib_transform(cib_feat)
        return so_feat + gated_cib

class DualBranchUNet(nn.Module):
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
        return self.final_conv(x)

if __name__ == '__main__':
    model = DualBranchUNet()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: " + str(total_params))
    print("Trainable parameters: " + str(trainable_params))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    batch_size = 4
    dummy_so_inputs = torch.randn(batch_size, 3, 256, 256).to(device)
    dummy_so_vars = torch.randn(batch_size, 3, 256, 256).to(device)
    dummy_cib_inputs = torch.randn(batch_size, 3, 256, 256).to(device)
    start_time = time.time()
    with torch.no_grad():
        output = model(dummy_so_inputs, dummy_so_vars, dummy_cib_inputs)
    end_time = time.time()
    print("Forward pass completed in " + str(end_time - start_time) + " seconds.")
    try:
        dataset = FlamingoDataset(split='val')
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        so_inputs = batch['so_inputs'].to(device)
        so_vars = batch['so_vars'].to(device)
        cib_inputs = batch['cib_inputs'].to(device)
        tsz_gt = batch['tsz_gt'].to(device)
        with torch.no_grad():
            pred_tsz = model(so_inputs, so_vars, cib_inputs)
        loss_fn = nn.L1Loss()
        initial_loss = loss_fn(pred_tsz, tsz_gt)
        print("Initial L1 Loss: " + str(initial_loss.item()))
    except Exception as e:
        print("Error testing with real data: " + str(e))