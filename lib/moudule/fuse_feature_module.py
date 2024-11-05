import torch
import torch.nn as nn
from timm.models.layers.drop import DropPath


class FFN(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, in_channel, out_channel, mid_ratio):
        super(FFN, self).__init__()
        self.mid_channel = in_channel * mid_ratio
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, self.mid_channel, 1),
            nn.Conv2d(self.mid_channel, self.mid_channel, 3, padding=1, groups=self.mid_channel),
            nn.GELU(),
            nn.Conv2d(self.mid_channel, out_channel, 1)
        )

    def forward(self, x):
        return self.layers(x)


class CMA(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, in_channel):
        super(CMA, self).__init__()
        self.conv1_0 = nn.Conv2d(in_channel, in_channel, 1)
        self.act = nn.GELU()

        self.conv0 = nn.Conv2d(in_channel, in_channel, 5, padding=2)
        # self.conv1 = Pconv_MLPBlock(in_channel, n_div, mlp_ratio=2, pconv_fw_type=forward_type)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, padding=0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 1, padding=0, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 5, padding=2, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 5, padding=2, bias=False),
        )
        self.conv4 = nn.Conv2d(in_channel, in_channel, 1)

        self.conv1_1 = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv1_0(x)
        attn = self.conv0(self.act(attn))

        attn1 = self.conv1(attn)
        attn2 = self.conv2(attn)
        attn3 = self.conv3(attn)
        attn = self.conv4(attn + attn1 + attn2 + attn3)

        return self.conv1_1(attn * u)


class CMAM(nn.Module):
    def __init__(self, in_channels, mlp_ratio=2, drop_path_rate=0):
        super(CMAM, self).__init__()
        self.bt0 = nn.BatchNorm2d(in_channels)
        self.attention = CMA(in_channels)
        self.bt1 = nn.BatchNorm2d(in_channels)
        self.FFN = FFN(in_channels, in_channels, mid_ratio=mlp_ratio)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    '''forward'''

    def forward(self, x):
        shortcut = x
        o1 = shortcut + self.attention(self.bt0(x))
        shortcut = o1
        out = shortcut + self.drop_path(self.FFN(self.bt1(o1)))
        return out
