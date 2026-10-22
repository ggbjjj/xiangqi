"""策略-价值网络（ResNet 风格）。

输入 : (B, 15, 10, 9) 的局面张量
策略  : (B, 8100) 的动作 logits（配合 softmax 得策略分布）
价值  : (B, 1) 的局面胜率估计，tanh 到 [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """预激活 ResNet 块（参考 AlphaZero 论文）。

    BN-ReLU-Conv-BN-ReLU-Conv + 残差
    """

    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        return x + out


class PolicyValueNet(nn.Module):
    """策略-价值双头网络。

    Args:
        in_channels: 输入通道数（默认 15）
        channels:    残差主干通道数（默认 64，CPU 训练够用）
        num_blocks:  残差块数（默认 6）
    """

    NUM_ACTIONS = 8100

    def __init__(self, in_channels=15, channels=64, num_blocks=6):
        super().__init__()
        # 初始卷积
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # 残差主干
        self.backbone = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # 策略头
        self.policy_conv = nn.Conv2d(channels, 16, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * 10 * 9, self.NUM_ACTIONS)

        # 价值头
        self.value_conv = nn.Conv2d(channels, 8, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 10 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.backbone(self.stem(x))

        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(h)), inplace=True)
        p = p.reshape(p.size(0), -1)
        p_logits = self.policy_fc(p)

        # 价值头
        v = F.relu(self.value_bn(self.value_conv(h)), inplace=True)
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        v_pred = torch.tanh(self.value_fc2(v))

        return p_logits, v_pred
