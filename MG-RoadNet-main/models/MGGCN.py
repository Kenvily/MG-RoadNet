import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

class Sobel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Sobel, self).__init__()
        kernel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        kernel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        kernel_x = torch.FloatTensor(kernel_x).expand(out_channel, in_channel, 3, 3)
        kernel_x = kernel_x.type(torch.cuda.FloatTensor)
        kernel_y = torch.cuda.FloatTensor(kernel_y).expand(out_channel, in_channel, 3, 3)
        kernel_y = kernel_y.type(torch.cuda.FloatTensor)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).clone()
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).clone()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        sobel_x = F.conv2d(x, self.weight_x, stride=1, padding=1)
        sobel_x = torch.abs(sobel_x)
        sobel_y = F.conv2d(x, self.weight_y, stride=1, padding=1)
        sobel_y = torch.abs(sobel_y)
        if c == 1:
            sobel_x = sobel_x.view(b, h, -1)
            sobel_y = sobel_y.view(b, h, -1).permute(0, 2, 1)
        else:
            sobel_x = sobel_x.view(b, c, -1)
            sobel_y = sobel_y.view(b, c, -1).permute(0, 2, 1)
        sobel_A = torch.bmm(sobel_x, sobel_y)
        sobel_A = self.softmax(sobel_A)
        return sobel_A

class GCNSpatial(nn.Module):
    def __init__(self, channels):
        super(GCNSpatial, self).__init__()
        self.sobel = Sobel(channels, channels)
        self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def normalize(self, A):
        b, c, im = A.size()
        out = np.array([])
        for i in range(b):
            A1 = A[i].to(device="cpu")
            I = torch.eye(c, im)
            A1 = A1 + I
            d = A1.sum(1)
            D = torch.diag(torch.pow(d, -0.5))
            new_A = D.mm(A1).mm(D).detach().numpy()
            out = np.append(out, new_A)
        out = out.reshape(b, c, im)
        normalize_A = torch.from_numpy(out)
        normalize_A = normalize_A.type(torch.cuda.FloatTensor)
        return normalize_A

    def forward(self, x):
        b, c, h, w = x.size()
        A = self.sobel(x)
        A = self.normalize(A)
        x = x.view(b, c, -1)
        x = F.relu(self.fc1(A.bmm(x)))
        x = F.relu(self.fc2(A.bmm(x)))
        x = self.fc3(A.bmm(x))
        out = x.view(b, c, h, w)
        return out


class GCNChannel(nn.Module):
    def __init__(self, channels):
        super(GCNChannel, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.sobel = Sobel(1, 1)
        self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def pre(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = x.view(b, 1, h * w, c)
        return x

    def normalize(self, A):
        b, c, im = A.size()
        out = np.array([])
        for i in range(b):
            # A = A = I
            A1 = A[i].to(device="cpu")
            I = torch.eye(c, im)
            A1 = A1 + I
            # degree matrix
            d = A1.sum(1)
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            new_A = D.mm(A1).mm(D).detach().numpy()
            out = np.append(out, new_A)
        out = out.reshape(b, c, im)
        normalize_A = torch.from_numpy(out)
        normalize_A = normalize_A.type(torch.cuda.FloatTensor)
        return normalize_A

    def forward(self, x):
        b, c, h, w = x.size()

        x = self.input(x)
        b, c, h1, w1 = x.size()
        x = self.pre(x)
        A = self.sobel(x)
        A = self.normalize(A)
        x = x.view(b, -1, c)
        x = F.relu(self.fc1(A.bmm(x).permute(0, 2, 1))).permute(0, 2, 1)
        x = F.relu(self.fc2(A.bmm(x).permute(0, 2, 1))).permute(0, 2, 1)
        x = self.fc3(A.bmm(x).permute(0, 2, 1))
        out = x.view(b, c, h1, w1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        return out


class ACFE_GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACFE_GCN, self).__init__()
        self.channel_in = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1),
            BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.gcn_c = GCNChannel(in_channels // 2)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1, bias=False),
            BatchNorm2d(in_channels // 2),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(in_channels // 2),
            nn.ReLU(in_channels // 2),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, bias=True)
        )

    def forward(self, x):

        x_channel_in = self.channel_in(x)
        x_channel = self.gcn_c(x_channel_in)
        x_channel = x_channel_in + x_channel

        # x_channel: in_channels/2×original_size/8×original_size/8
        # out = torch.cat((x_de, x_channel), 1) + x

        out = self.output(x_channel)
        # out:in_channels×original_size/8×original_size/8
        return out


class GTFE_GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GTFE_GCN, self).__init__()
        self.spatial_in = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.gcn_s = GCNSpatial(in_channels // 2)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1, bias=False),
            BatchNorm2d(in_channels // 2),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(in_channels // 2),
            nn.ReLU(in_channels // 2),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, bias=True)
        )

    def forward(self, x):

        x_spatial_in = self.spatial_in(x)
        x_spatial = self.gcn_s(x_spatial_in)
        x_spatial = x_spatial_in + x_spatial

        out = self.output(x_spatial)
        return out


def build_acfe_gcn(in_channels, out_channels):
    return ACFE_GCN(in_channels=in_channels, out_channels=out_channels)


def build_gtfe_gcn(in_channels, out_channels):
    return GTFE_GCN(in_channels=in_channels, out_channels=out_channels)
