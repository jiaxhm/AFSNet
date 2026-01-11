# coding:utf-8
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.kan = FasterKAN(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x)
        y = self.kan(y).view(b, c)
        y = self.sigmoid(y).view(b, c, 1, 1)  # Excitation
        return x * y.expand_as(x)  # Scale

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        super().__init__(in_features, out_features, bias=False, **kw)
        self.init_scale = init_scale
        nn.init.xavier_uniform_(self.weight)


class ReflectionalSwitchFunction(nn.Module):
    def __init__(self, grid_min=-2., grid_max=2., num_grids=8, denominator=0.33):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.inv_denominator = 1 / denominator

    def forward(self, x):
        diff = (x[..., None] - self.grid).mul(self.inv_denominator)
        return 1 - torch.tanh(diff).pow(2)


class FasterKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_grids=8, denominator=0.33, spline_weight_init_scale=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = ReflectionalSwitchFunction(num_grids=num_grids, denominator=denominator)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)

    def forward(self, x):
        spline_basis = self.rbf(self.layernorm(x)).view(x.shape[0], -1)
        return self.spline_linear(spline_basis)


class FasterKAN(nn.Module):
    def __init__(self, channels: int, num_grids=8, denominator=0.33, spline_weight_init_scale=0.667):
        super().__init__()
        self.layers = nn.ModuleList([
            FasterKANLayer(channels, channels * 2, num_grids, denominator, spline_weight_init_scale),
            FasterKANLayer(channels * 2, channels, num_grids, denominator, spline_weight_init_scale)
        ])

    def forward(self, x):
        batch, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x.reshape(batch * h * w, c)

        for layer in self.layers:
            x = layer(x)

        x = x.reshape(batch, h, w, -1)  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = SEBlock(64).to(device)
    x = torch.randn(2, 64, 52, 52).to(device)
    output = model1(x)

    print(f"in shape: {x.shape}, out shape: {output.shape}")
