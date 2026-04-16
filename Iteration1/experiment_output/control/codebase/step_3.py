# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.cuda.amp import autocast, GradScaler

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

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class GatedCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x_main, x_aux):
        q = self.query(x_main)
        k = self.key(x_aux)
        v = self.value(x_aux)
        attn = torch.sigmoid(q * k)
        attended_aux = attn * v
        g = self.gate(torch.cat([x_main, attended_aux], dim=1))
        return x_main + self.gamma * g * attended_aux

class SR_DAE(nn.Module):
    def __init__(self, main_in_channels=3, aux_in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.main_inc = DoubleConv(main_in_channels, features[0])
        self.main_down1 = Down(features[0], features[1])
        self.main_down2 = Down(features[1], features[2])
        self.main_down3 = Down(features[2], features[3])
        self.main_down4 = Down(features[3], features[3]*2)
        self.aux_inc = DoubleConv(aux_in_channels, features[0])
        self.aux_down1 = Down(features[0], features[1])
        self.aux_down2 = Down(features[1], features[2])
        self.aux_down3 = Down(features[2], features[3])
        self.aux_down4 = Down(features[3], features[3]*2)
        self.attn_inc = GatedCrossAttention(features[0])
        self.attn_down1 = GatedCrossAttention(features[1])
        self.attn_down2 = GatedCrossAttention(features[2])
        self.attn_down3 = GatedCrossAttention(features[3])
        self.attn_bottleneck = GatedCrossAttention(features[3]*2)
        self.up1 = Up(features[3]*2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        x_main = x[:, :3, :, :]
        x_aux = x[:, 3:, :, :]
        m1 = self.main_inc(x_main)
        a1 = self.aux_inc(x_aux)
        m1_attn = self.attn_inc(m1, a1)
        m2 = self.main_down1(m1_attn)
        a2 = self.aux_down1(a1)
        m2_attn = self.attn_down1(m2, a2)
        m3 = self.main_down2(m2_attn)
        a3 = self.aux_down2(a2)
        m3_attn = self.attn_down2(m3, a3)
        m4 = self.main_down3(m3_attn)
        a4 = self.aux_down3(a3)
        m4_attn = self.attn_down3(m4, a4)
        m5 = self.main_down4(m4_attn)
        a5 = self.aux_down4(a4)
        m5_attn = self.attn_bottleneck(m5, a5)
        x_dec = self.up1(m5_attn, m4_attn)
        x_dec = self.up2(x_dec, m3_attn)
        x_dec = self.up3(x_dec, m2_attn)
        x_dec = self.up4(x_dec, m1_attn)
        logits = self.outc(x_dec)
        return logits

if __name__ == '__main__':
    print("Starting Model Architecture Implementation...")
    model = SR_DAE(main_in_channels=3, aux_in_channels=3, out_channels=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters: " + str(total_params))
    print("\nRunning forward pass unit test...")
    batch_size = 2
    channels = 6
    height, width = 256, 256
    dummy_input = torch.randn(batch_size, channels, height, width)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        with autocast(enabled=True):
            output = model(dummy_input)
    end_time = time.time()
    print("Input shape: " + str(dummy_input.shape))
    print("Output shape: " + str(output.shape))
    print("Forward pass time (batch_size=" + str(batch_size) + "): " + ("%.4f" % (end_time - start_time)) + " seconds")
    if output.shape == (batch_size, 1, height, width):
        print("Forward pass unit test passed successfully.")
    else:
        print("Forward pass unit test failed. Unexpected output shape.")
    print("\nRunning backward pass unit test with mixed precision...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    optimizer.zero_grad()
    start_time = time.time()
    with autocast(enabled=True):
        output = model(dummy_input)
        loss = output.mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    end_time = time.time()
    print("Backward pass time (batch_size=" + str(batch_size) + "): " + ("%.4f" % (end_time - start_time)) + " seconds")
    print("Backward pass unit test passed successfully.")
    print("\nModel Architecture Implementation completed successfully.")