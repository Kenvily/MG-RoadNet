import torch
import torch.nn as nn
from models.MGGCN import build_acfe_gcn
from models.MGGCN import build_gtfe_gcn


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


class MG_ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(MG_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.fc = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)
        # )
        self.fc = nn.Sequential(
            build_gtfe_gcn(in_channels, in_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class mrf_attention(nn.Module):
    def __init__(self, kernel_size=7, rf_num=3):
        super(mrf_attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2,
                               bias=False)

        self.attention = nn.Sequential(
            nn.Conv2d(rf_num, rf_num * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(rf_num * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(rf_num * 16, rf_num, kernel_size=3, padding=1),
        )

    def spatial_norm(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x

    def forward(self, x_list):
        out = torch.cat([self.spatial_norm(x) for x in x_list], dim=1)
        out = self.attention(out)
        out = self.softmax(out)
        return out


class ACFE(nn.Module):
    def __init__(self, in_channels=128, m_channels=[32, 48, 64], out_channels=128, norm_layer=nn.BatchNorm2d):
        super(ACFE, self).__init__()
        self.rf1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.rf3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=m_channels[0], kernel_size=1),
            norm_layer(m_channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[0], out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )
        self.rf5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=m_channels[0], kernel_size=1),
            norm_layer(m_channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[0], out_channels=m_channels[1], kernel_size=3, padding=1),
            norm_layer(m_channels[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[1], out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
        )

        self.ca = ChannelAttention(in_channels=out_channels)

        self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.rf_attention = mrf_attention(rf_num=3)

        self.downsample = nn.Identity()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.downsample(x)

        rf1_x = self.rf1(x)
        rf3_x = self.rf3(x)
        rf5_x = self.rf5(x)

        out = self.rf_attention([rf1_x, rf3_x, rf5_x])
        rf1_w, rf3_w, rf5_w = torch.split(out, split_size_or_sections=1, dim=1)
        identity_2 = rf1_x * rf1_w + rf3_x * rf3_w + rf5_x * rf5_w + identity

        out = self.out(identity_2)

        out = self.ca(out) * out

        return self.relu(out + identity_2)


class mrf_attention_3(nn.Module):
    def __init__(self,  rf_num=3):
        super(mrf_attention_3, self).__init__()
        self.softmax = nn.Softmax(dim=1)

        # self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2,
        #                        bias=False)
        self.conv1 = build_acfe_gcn(2, 1)

        self.attention = nn.Sequential(
            nn.Conv2d(rf_num , rf_num * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(rf_num * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(rf_num * 16, rf_num, kernel_size=3, padding=1),
        )

    def spatial_norm(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x

    def forward(self, x_list):
        out = torch.cat([self.spatial_norm(x) for x in x_list], dim=1)
        out = self.attention(out)
        out = self.softmax(out)
        return out


class ACFE_3(nn.Module):
    def __init__(self, in_channels=128, m_channels=[32, 48, 64], out_channels=128, norm_layer=nn.BatchNorm2d):
        super(ACFE_3, self).__init__()
        self.rf1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.rf3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=m_channels[0], kernel_size=1),
            norm_layer(m_channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[0], out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )
        self.rf5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=m_channels[0], kernel_size=1),
            norm_layer(m_channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[0], out_channels=m_channels[1], kernel_size=3, padding=1),
            norm_layer(m_channels[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[1], out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
        )

        self.ca = ChannelAttention(in_channels=out_channels)

        self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.rf_attention = mrf_attention_3(rf_num=3)

        self.downsample = nn.Identity()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.downsample(x)

        rf1_x = self.rf1(x)
        rf3_x = self.rf3(x)
        rf5_x = self.rf5(x)

        out = self.rf_attention([rf1_x, rf3_x, rf5_x])
        rf1_w, rf3_w, rf5_w = torch.split(out, split_size_or_sections=1, dim=1)
        identity_2 = rf1_x * rf1_w + rf3_x * rf3_w + rf5_x * rf5_w + identity

        out = self.out(identity_2)

        out = self.ca(out) * out

        return self.relu(out + identity_2)


class mrf_attention_4(nn.Module):
    def __init__(self, kernel_size=7, rf_num=3):
        super(mrf_attention_4, self).__init__()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2,
                               bias=False)
        # self.conv1 = build_dwm(2, 1)

        self.attention = nn.Sequential(
            nn.Conv2d(rf_num, rf_num * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(rf_num * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(rf_num * 16, rf_num, kernel_size=3, padding=1),
        )

    def spatial_norm(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x

    def forward(self, x_list):
        out = torch.cat([self.spatial_norm(x) for x in x_list], dim=1)
        out = self.attention(out)
        out = self.softmax(out)
        return out


class GTFE_4(nn.Module):
    def __init__(self, in_channels=128, m_channels=[32, 48, 64], out_channels=128, norm_layer=nn.BatchNorm2d):
        super(GTFE_4, self).__init__()
        self.rf1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.rf3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=m_channels[0], kernel_size=1),
            norm_layer(m_channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[0], out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )
        self.rf5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=m_channels[0], kernel_size=1),
            norm_layer(m_channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[0], out_channels=m_channels[1], kernel_size=3, padding=1),
            norm_layer(m_channels[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[1], out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
        )

        self.ca = MG_ChannelAttention(in_channels=out_channels)

        self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.rf_attention = mrf_attention_4(rf_num=3)

        self.downsample = nn.Identity()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.downsample(x)

        rf1_x = self.rf1(x)
        rf3_x = self.rf3(x)
        rf5_x = self.rf5(x)

        out = self.rf_attention([rf1_x, rf3_x, rf5_x])
        rf1_w, rf3_w, rf5_w = torch.split(out, split_size_or_sections=1, dim=1)
        identity_2 = rf1_x * rf1_w + rf3_x * rf3_w + rf5_x * rf5_w + identity

        out = self.out(identity_2)
        out = self.ca(out) * out

        return self.relu(out + identity_2)


def build_acfe(in_channels, m_channels, out_channels, norm_layer):
    return ACFE(in_channels=in_channels, m_channels=m_channels, out_channels=out_channels, norm_layer=norm_layer)


def build_acfe_3(in_channels, m_channels, out_channels, norm_layer):
    return ACFE_3(in_channels=in_channels, m_channels=m_channels, out_channels=out_channels, norm_layer=norm_layer)


def build_gtfe_4(in_channels, m_channels, out_channels, norm_layer):
    return GTFE_4(in_channels=in_channels, m_channels=m_channels, out_channels=out_channels, norm_layer=norm_layer)
