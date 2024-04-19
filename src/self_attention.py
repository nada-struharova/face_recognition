import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """ A simple self-attention module """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, width * height)
        value = self.value_conv(x).view(batch, -1, width * height)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return out + x  # Additive skip-connection to help with training deep networks

class OcclusionAwareFaceNet(nn.Module):
    def __init__(self):
        super(OcclusionAwareFaceNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SelfAttention(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 56 * 56, 10)  # Example for 224x224 input

    def forward(self, x, occlusion_map=None):
        x = self.features(x)
        if occlusion_map is not None:
            # Simulating dynamic weighting based on occlusion map
            x = x * (1 - occlusion_map)  # Assuming occlusion_map is pre-calculated
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Example usage
model = OcclusionAwareFaceNet()
print(model)
