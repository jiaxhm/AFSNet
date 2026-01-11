import torch
import torch.nn.functional
import torch.nn as nn
from einops import rearrange

class GCSA(nn.Module):
    def __init__(self, dim, bias):
        super(GCSA, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.cnn_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim * 3,
                                    bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x2):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        cnn_k = self.cnn_k(x2)
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        cnn_k = rearrange(cnn_k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        cnn_k = torch.nn.functional.normalize(cnn_k, dim=-1)
        attn1 = (q @ k.transpose(-2, -1))
        attn2 = (q @ cnn_k.transpose(-2, -1)) #* self.temperature
        attn = attn1 + attn2 #+ attn1 * attn2 * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)
        # out1 = x + self.project_out1(out)
        out1 = self.project_out1(out)
        # out2 = x2 + self.project_out2(out)
        out2 = self.project_out2(out)
        # out = torch.cat((out1, out2), dim=1)
        # out = self.out(out)
        ###
        att = torch.stack([out1, out2], dim=1)
        softmax_att = self.softmax(att)
        feature = torch.cat([x.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        out = (feature * softmax_att).sum(dim=1)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 32, 256, 256).to(device)
    x2 = torch.randn(1, 32, 256, 256).to(device)
    gcsa = GCSA(dim=32, bias=True)
    print(gcsa)
    gcsa = gcsa.to(device)
    output = gcsa(x, x2)
    print("X:", x.shape)
    print("OUT", output.shape)