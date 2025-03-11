import torch
import torch.nn as nn
from models.encoder.ACFE import build_acfe
from models.encoder.ACFE import build_acfe_3
from models.encoder.ACFE import build_gtfe_4

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2,
                               bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class attention_module(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(attention_module, self).__init__()
        self.ca = ChannelAttention(in_channels=in_channels)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        ca = self.ca(x) * x
        out = self.sa(ca) * ca

        return out


class NConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm_layer=None, attention=False, group=1):
        super(NConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1, stride=stride,
                                groups=group)
        self.conv2 = nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel_size=3, padding=1,
                               stride=1, groups=out_channels // 2)

    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out = torch.cat([out_1, out_2], dim=1)
        return out


class L_BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm_layer=None, attention=False, group=1):
        super(L_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.Nconv1 = NConv(in_channels=in_channels, out_channels=out_channels, stride=stride, norm_layer=norm_layer)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.Nconv2 = NConv(in_channels=in_channels, out_channels=out_channels, stride=stride, norm_layer=norm_layer)
        self.bn2 = norm_layer(out_channels)

        self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.Nconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.Nconv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm_layer=None, attention=False, group=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, groups=group)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, groups=group)
        self.bn2 = norm_layer(out_channels)
        self.stride = stride
        if attention:
            self.cbam = attention_module(in_channels=out_channels, kernel_size=7)
        else:
            self.cbam = False

        self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.cbam:
            out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out


class parralle_downsp(nn.Module):
    def __init__(self, dim):
        super(parralle_downsp, self).__init__()
        self.maxp = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        maxp = self.maxp(x)
        conv = self.conv(x)
        return torch.cat([maxp, conv], dim=1)


class Encoder(nn.Module):
    def __init__(self, norm_layer=None):
        super(Encoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            BasicBlock(in_channels=64, out_channels=64, stride=1, norm_layer=norm_layer),
        )

        self.stem1 = nn.Sequential(
            parralle_downsp(dim=64)
        )

        self.rf_attention_pooling_1_1 = nn.Sequential(
            build_acfe(128, [64, 96, 128], 128, norm_layer),

            BasicBlock(in_channels=128, out_channels=128, stride=1, norm_layer=norm_layer),
        )

        self.stem2 = nn.Sequential(
            parralle_downsp(dim=128)
        )

        self.rf_attention_pooling_2_1 = nn.Sequential(
            build_acfe_3(256, [128, 192, 256], 256, norm_layer),

            BasicBlock(in_channels=256, out_channels=256, stride=1, norm_layer=norm_layer),
        )

        self.rf_attention_pooling_2_2 = nn.Sequential(
            build_gtfe_4(256, [128, 192, 256], 256, norm_layer),

            BasicBlock(in_channels=256, out_channels=256, stride=1, norm_layer=norm_layer),

        )

        self.stem3 = nn.Sequential(
            parralle_downsp(dim=256)
        )

        self.rf_attention_pooling_3_1 = nn.Sequential(
            build_acfe(512, [256, 384, 512], 512, norm_layer),

            BasicBlock(in_channels=512, out_channels=512, stride=1, norm_layer=norm_layer),
        )

        self.rf_attention_pooling_3_2 = nn.Sequential(
            build_acfe(512, [256, 384, 512], 512, norm_layer),

            BasicBlock(in_channels=512, out_channels=512, stride=1, norm_layer=norm_layer),
        )

        self.rf_attention_pooling_3_3 = nn.Sequential(
            build_acfe(512, [256, 384, 512], 512, norm_layer),

            BasicBlock(in_channels=512, out_channels=512, stride=1, norm_layer=norm_layer),
        )

        self.rf_attention_pooling_3_4 = nn.Sequential(
            build_acfe(512, [256, 384, 512], 512, norm_layer),

            BasicBlock(in_channels=512, out_channels=512, stride=1, norm_layer=norm_layer),
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out1 = self.inconv(input)

        out2 = self.stem1(out1)
        out2 = self.rf_attention_pooling_1_1(out2)

        out3 = self.stem2(out2)
        out3 = self.rf_attention_pooling_2_1(out3)
        out3 = self.rf_attention_pooling_2_2(out3)

        out4 = self.stem3(out3)
        out4 = self.rf_attention_pooling_3_1(out4)
        out4 = self.rf_attention_pooling_3_2(out4)
        out4 = self.rf_attention_pooling_3_3(out4)
        out4 = self.rf_attention_pooling_3_4(out4)

        return out1, out2, out3, out4


def build_encoder(norm_layer):
    return Encoder(norm_layer=norm_layer)
