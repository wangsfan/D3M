import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

def get_dct_basis(num_basis):
    all_basis_x = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
                   3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
                   7, 7, 7, 7]
    all_basis_y = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
                   6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3,
                   4, 5, 6, 7]
    mapper_x = all_basis_x[:num_basis]
    mapper_y = all_basis_y[:num_basis]
    return mapper_x, mapper_y

class FDMLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction, num_basis):
        super(FDMLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.num_basis = num_basis
        self.space_size = 8
        mapper_x, mapper_y = get_dct_basis(self.num_basis)
        mapper_x = [temp_x * (self.dct_h // self.space_size) for temp_x in mapper_x]
        mapper_y = [temp_y * (self.dct_w // self.space_size) for temp_y in mapper_y]

        self.fd_layer = FDLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fm_layer = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        out_c = self.fd_layer(x_pooled)

        out_s = self.fm_layer(out_c).view(n, c, 1, 1)
        return x * out_s.expand_as(x)

class FDLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(FDLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_basis = len(mapper_x)

        self.register_parameter('weight', Parameter(self.get_dct_filter(height, width, mapper_x, mapper_y, channel)))

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)
        return dct_filter


    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        x = x * self.weight
        out_c = torch.sum(x, dim=[2, 3])
        return out_c
