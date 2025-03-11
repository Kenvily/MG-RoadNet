import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.encoder.encoder import build_encoder
from models.MGGCN import build_acfe_gcn, build_gtfe_gcn
from models.decoder import build_decoder

device = torch.device("cuda:0")

class normer(nn.Module):
    def __init__(self, mode=None):
        super(normer, self).__init__()
        self.mode = mode

    def forward(self, num_features):
        if self.mode == 'BN':
            res = nn.BatchNorm2d(num_features=num_features)

        elif self.mode == 'GN':
            res = nn.GroupNorm(num_groups=num_features//4, num_channels=num_features)

        else:
            res = nn.BatchNorm2d(num_features=num_features)

        return res

class MGNet(nn.Module):
    def __init__(self, num_classes, norm_layer=None):
        super(MGNet, self).__init__()
        norm_layer = normer(norm_layer)
        self.backbone = build_encoder(norm_layer)
        self.dsn = nn.Sequential(
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Dropout2d(0.1),
        nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        #  TwofoldGCN # for channel and spatial feature
        self.gtfegcn_out = build_gtfe_gcn(512, 512)

        self.up1 = build_decoder(512, 256)
        self.up2 = build_decoder(256, 128)
        self.up3 = build_decoder(128, 64)

        # Full connection
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        x4 = self.gtfegcn_out(x4)

        x4 = self.up1(x3, x4)
        x4 = self.up2(x2, x4)
        x4 = self.up3(x1, x4)

        final = self.final_conv(x4)
        return final

def build_MGRoad(num_classes=2):
    model = MGNet(num_classes)
    return model



