import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def autopad(self, k, p):
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        x = self.cv1(x)
        x1, x2 = x.chunk(2, 1)
        y = [x1, x2]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class RCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class CBM4(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, 1)
        self.conv2 = Conv(c2, c2, 3, 1)
        self.conv3 = Conv(c2, c2, 3, 1)
        self.conv4 = Conv(c2, c2, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv4(x)

class ChannelAttention(nn.Module):
    def __init__(self, c1, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1 // ratio, c1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class FPA(nn.Module):
    def __init__(self, channels: List[int]):
        super().__init__()
        self.channels = channels
        self.attention_modules = nn.ModuleList([
            ChannelAttention(c) for c in channels
        ])
        
    def forward(self, features: List[torch.Tensor]):
        weighted_features = []
        for i, (feat, attn) in enumerate(zip(features, self.attention_modules)):
            weight = attn(feat)
            weighted_features.append(feat * weight)
        return weighted_features

class CustomYOLOBackbone(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        
        self.stem = Conv(3, channels[0], 6, 2, 2)
        
        self.stage1 = nn.Sequential(
            Conv(channels[0], channels[0], 3, 2),
            RCSP(channels[0], channels[0], 1),
        )
        
        self.stage2 = nn.Sequential(
            Conv(channels[0], channels[1], 3, 2),
            RCSP(channels[1], channels[1], 2),
        )
        
        self.stage3 = nn.Sequential(
            Conv(channels[1], channels[2], 3, 2),
            RCSP(channels[2], channels[2], 2),
        )
        
        self.stage4 = nn.Sequential(
            Conv(channels[2], channels[3], 3, 2),
            RCSP(channels[3], channels[3], 1),
        )
        
        self.sppf = SPPF(channels[3], channels[3])

    def forward(self, x):
        x = self.stem(x)
        
        c1 = self.stage1(x)
        c2 = self.stage2(c1)  
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c4 = self.sppf(c4)
        
        return [c1, c2, c3, c4]

class CustomYOLONeck(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.c2f1 = C2f(channels[3] + channels[2], channels[2], 1)
        self.c2f2 = C2f(channels[2] + channels[1], channels[1], 1)
        
        self.conv1 = Conv(channels[1], channels[1], 3, 2)
        self.c2f3 = C2f(channels[1] + channels[2], channels[2], 1)
        
        self.conv2 = Conv(channels[2], channels[2], 3, 2)
        self.c2f4 = C2f(channels[2] + channels[3], channels[3], 1)
        
        self.fpa = FPA(channels[1:])

    def forward(self, features):
        c1, c2, c3, c4 = features
        
        p4 = self.upsample(c4)
        p4 = torch.cat([p4, c3], 1)
        p4 = self.c2f1(p4)
        
        p3 = self.upsample(p4)
        p3 = torch.cat([p3, c2], 1)
        p3 = self.c2f2(p3)
        
        p4_out = self.conv1(p3)
        p4_out = torch.cat([p4_out, p4], 1)
        p4_out = self.c2f3(p4_out)
        
        p5_out = self.conv2(p4_out)
        p5_out = torch.cat([p5_out, c4], 1)
        p5_out = self.c2f4(p5_out)
        
        outputs = [p3, p4_out, p5_out]
        outputs = self.fpa(outputs)
        
        return outputs

class CustomYOLOHead(nn.Module):
    def __init__(self, nc=6, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            z.append(x[i])
        return z

class CustomYOLO(nn.Module):
    def __init__(self, nc=6, channels=[64, 128, 256, 512]):
        super().__init__()
        self.backbone = CustomYOLOBackbone(channels)
        self.neck = CustomYOLONeck(channels)
        
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.head = CustomYOLOHead(nc, anchors, channels[1:])

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x) 