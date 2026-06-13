import torch
import torch.nn as nn

class IlluminationGuidedAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            channels, 
            channels, 
            kernel_size=3, 
            padding=1, 
            groups=channels
        )
        self.pw_conv1 = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.relu = nn.ReLU()
        self.pw_conv2 = nn.Conv2d(channels // 4, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        self.residual_layer = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        gamma = self.dw_conv(x)
        gamma = self.pw_conv1(gamma)
        gamma = self.relu(gamma)
        gamma = self.pw_conv2(gamma)
        gamma = self.sigmoid(gamma)
        
        beta = self.residual_layer(x)
        
        return (x * gamma) + beta

class EnvironmentalGating(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gating_network = nn.Sequential(
            nn.Linear(channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        pooled_features = self.global_pool(x).view(batch_size, channels)
        alpha = self.gating_network(pooled_features)
        return alpha.view(batch_size, 1, 1, 1)