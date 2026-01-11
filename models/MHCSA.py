# coding:utf-8
import torch
import torch.nn as nn
from einops import rearrange
class MHSA(nn.Module):
    def __init__(self, n_dims):
        super(MHSA, self).__init__()

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.cnn_att_key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.cnn_att = fusionatt()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x2):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)
        content_content = torch.bmm(q.permute(0, 2, 1), k)

        content_position = self.cnn_att_key(self.cnn_att(x2)).view(n_batch, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)
# r wise
        energy = content_content + content_position
        attention = self.softmax(energy)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        return out

class fusionatt(nn.Module):
    def __init__(self):
        super(fusionatt, self).__init__()

        # channel attention
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.softmax = nn.Softmax(1)

        self.conv1 = nn.Conv1d(2, 1, kernel_size=5, padding=2, bias=False)

        self.spatial_conv1 = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, feature):
        b1, c1, h1, w1 = feature.shape
        max_out1 = self.max_pool(feature).squeeze(-1).squeeze(-1)
        avg_out1 = self.avg_pool(feature).squeeze(-1).squeeze(-1)
        stacked1 = torch.stack([avg_out1, max_out1], dim=1)
        out1 = self.conv1(stacked1).squeeze(1)
        out1 = out1.unsqueeze(2).unsqueeze(3)
        channel_out1 = out1.repeat(1, 1, h1, w1)

        spatial_mean_out1 = torch.mean(feature, dim=1, keepdim=True)
        spatial_max_out1, _ = torch.max(feature, dim=1, keepdim=True)
        spatial_out1 = torch.cat([spatial_mean_out1, spatial_max_out1], dim=1)
        spatial_out1 = self.spatial_conv1(spatial_out1)
        spatial_out1 = spatial_out1.repeat(1, c1, 1, 1)

        out = channel_out1 + spatial_out1

        return out

if __name__ == "__main__":
    # 将模块移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建测试输入张量 (batch_size, channels, height, width)
    x = torch.randn(4, 32, 56, 56).to(device)
    x2 = torch.randn(4, 32, 56, 56).to(device)
    # 初始化 GCSA 模块
    mhsa = MHSA(32)
    print(mhsa)
    mhsa = mhsa.to(device)
    # 前向传播
    output = mhsa(x, x2)
    # 打印输入和输出张量的形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)