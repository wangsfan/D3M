import torch
import torch.nn as nn

class SDMLayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SDMLayer, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out_s = self.sigmod(out)

        return out_s * x