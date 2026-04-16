# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class GatedCrossAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class SR_DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p_e1 = ConvBlock(3, 64)
        self.p_e2 = ConvBlock(64, 128)
        self.p_e3 = ConvBlock(128, 256)
        self.p_e4 = ConvBlock(256, 512)
        self.p_b = ConvBlock(512, 1024)
        self.a_e1 = ConvBlock(3, 64)
        self.a_e2 = ConvBlock(64, 128)
        self.a_e3 = ConvBlock(128, 256)
        self.a_e4 = ConvBlock(256, 512)
        self.a_b = ConvBlock(512, 1024)
        self.b_att = GatedCrossAttention(F_g=1024, F_l=1024, F_int=512)
        self.b_conv = ConvBlock(2048, 1024)
        self.up4 = UpConv(1024, 512)
        self.att4 = GatedCrossAttention(F_g=512, F_l=512, F_int=256)
        self.d4_conv = ConvBlock(512 + 512 + 512, 512)
        self.up3 = UpConv(512, 256)
        self.att3 = GatedCrossAttention(F_g=256, F_l=256, F_int=128)
        self.d3_conv = ConvBlock(256 + 256 + 256, 256)
        self.up2 = UpConv(256, 128)
        self.att2 = GatedCrossAttention(F_g=128, F_l=128, F_int=64)
        self.d2_conv = ConvBlock(128 + 128 + 128, 128)
        self.up1 = UpConv(128, 64)
        self.att1 = GatedCrossAttention(F_g=64, F_l=64, F_int=32)
        self.d1_conv = ConvBlock(64 + 64 + 64, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x_primary, x_auxiliary):
        p1 = self.p_e1(x_primary)
        p2 = self.p_e2(self.pool(p1))
        p3 = self.p_e3(self.pool(p2))
        p4 = self.p_e4(self.pool(p3))
        pb = self.p_b(self.pool(p4))
        a1 = self.a_e1(x_auxiliary)
        a2 = self.a_e2(self.pool(a1))
        a3 = self.a_e3(self.pool(a2))
        a4 = self.a_e4(self.pool(a3))
        ab = self.a_b(self.pool(a4))
        ab_att = self.b_att(g=pb, x=ab)
        b = self.b_conv(torch.cat([pb, ab_att], dim=1))
        d4 = self.up4(b)
        a4_att = self.att4(g=p4, x=a4)
        d4 = self.d4_conv(torch.cat([d4, p4, a4_att], dim=1))
        d3 = self.up3(d4)
        a3_att = self.att3(g=p3, x=a3)
        d3 = self.d3_conv(torch.cat([d3, p3, a3_att], dim=1))
        d2 = self.up2(d3)
        a2_att = self.att2(g=p2, x=a2)
        d2 = self.d2_conv(torch.cat([d2, p2, a2_att], dim=1))
        d1 = self.up1(d2)
        a1_att = self.att1(g=p1, x=a1)
        d1 = self.d1_conv(torch.cat([d1, p1, a1_att], dim=1))
        out = self.out_conv(d1)
        return out

if __name__ == '__main__':
    model = SR_DAE()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters: " + str(total_params))
    batch_size = 16
    channels = 3
    height, width = 256, 256
    print("Verifying forward pass with dummy tensors...")
    x_primary = torch.randn(batch_size, channels, height, width)
    x_auxiliary = torch.randn(batch_size, channels, height, width)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x_primary = x_primary.to(device)
    x_auxiliary = x_auxiliary.to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = model(x_primary, x_auxiliary)
    print("Output shape: " + str(out.shape))
    loss = out.sum()
    loss.backward()
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print("Estimated GPU memory footprint (Batch Size 16): " + str(round(max_mem, 2)) + " GB")
    else:
        print("CUDA not available. Could not estimate GPU memory footprint.")