import math

import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels, kernel_size=3, padding=1, stride=1, use_conv=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=kernel_size, padding=padding, stride=1)

        if use_conv:
            self.shortcut = nn.Conv2d(input_channels, num_channels, kernel_size=1, padding=0, stride=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, X):
        out = F.relu(self.conv1(X))
        out = self.conv2(out)
        out += self.shortcut(X)
        out = F.relu(out)
        return out

class TimeStepResBlock(nn.Module):

    def __init__(self, input_channels, num_channels, embedding_size, kernel_size=3, padding=1, stride=1, use_conv=True, dropout=0.1):
        super().__init__()

        self.input_channels = input_channels
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=kernel_size, padding=padding, stride=1)

        if use_conv:
            self.shortcut = nn.Conv2d(input_channels, num_channels, kernel_size=1, padding=0, stride=1)
        else:
            self.shortcut = nn.Identity()

        self.emb_proj = nn.Linear(embedding_size, num_channels)
        
        self.norm1 = nn.GroupNorm(32, input_channels)
        self.norm2 = nn.GroupNorm(32, num_channels)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X, embedding):
        embedding = self.emb_proj(F.relu(embedding))
        out = self.norm1(X)
        out = F.relu(self.conv1(out))

        while len(embedding.shape) < len(out.shape):
            embedding = embedding[..., None]
        
        out = out + embedding
        out = self.norm2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(X)
        out = F.relu(out)
        return out

class PositionalEncoding(nn.Module):
    """Positional Encoding."""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
        self.proj1 = nn.Linear(dim, dim * 4)
        self.proj2 = nn.Linear(dim * 4, dim * 4)

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        
        emb = self.proj1(emb)
        emb = F.relu(emb)
        emb = self.proj2(emb)

        return emb
        

class Attention(nn.Module):
    def __init__(self, channels, num_heads):
        super(Attention, self).__init__()

        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(32, channels)
        
        self.num_heads = num_heads
        self.channels = channels

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        h = torch.einsum("bts,bcs->bct", weight, v)
        
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class MiniUNet(nn.Module):

    def __init__(self, input_channels, num_channels, out_channels, embedding_size, kernel_size=3, padding=1, stride=1, use_conv=True):
        super(MiniUNet, self).__init__()
        
        channel_mults = [1, 1, 2, 2, 4, 4]
        
        self.embeddings = PositionalEncoding(embedding_size)
        
        self.norm = nn.GroupNorm(32, num_channels)

        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=kernel_size, padding=padding)
        self.input_blocks = []
        self.down_attention = []
        self.downsample_blocks = []


        for i in range(len(channel_mults) - 1):
            self.input_blocks.append(TimeStepResBlock(num_channels * channel_mults[i], num_channels * channel_mults[(i + 1)], embedding_size * 4, kernel_size, padding, stride, use_conv))

            self.downsample_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if i == 3 or i == 4:
                self.down_attention.append(Attention(num_channels * channel_mults[(i + 1)], 4))

        self.input_blocks = nn.ModuleList(self.input_blocks)
        self.down_attention = nn.ModuleList(self.down_attention)
        
        self.resize_blocks = []
        for j, i in enumerate(range(len(channel_mults) -1, 0, -1)):
            if j % 2 == 0:
                self.resize_blocks.append(TimeStepResBlock(num_channels * channel_mults[i] * 2 , num_channels * channel_mults[(i - 1)], embedding_size * 4, kernel_size, padding, stride, use_conv))
            else:
                self.resize_blocks.append(TimeStepResBlock(num_channels * channel_mults[i] * 2 , num_channels * channel_mults[(i - 1)] * 2, embedding_size * 4, kernel_size, padding, stride, use_conv))

        self.resize_blocks = nn.ModuleList(self.resize_blocks)

        self.upsample_blocks = []
        self.up_attention = []
        for i in range(len(channel_mults) -1, 0, -1):

            self.upsample_blocks.append(nn.ConvTranspose2d(num_channels * channel_mults[i], num_channels * channel_mults[(i - 1)], kernel_size=2, stride=2))
            if i == 4 or i == 3:
                self.up_attention.append(Attention(num_channels * channel_mults[(i)], 4))

        self.upsample_blocks = nn.ModuleList(self.upsample_blocks)
        self.up_attention = nn.ModuleList(self.up_attention)

        self.conv9 = nn.Conv2d(num_channels, out_channels, 3, padding='same')
        
    def forward(self, x, timesteps):
        timesteps = self.embeddings(timesteps)
        h = self.conv1(x)
        h = F.relu(h)
        hs = [h]
        for i, (block, downsample) in enumerate(zip(self.input_blocks, self.downsample_blocks)):
            h = block(h, timesteps)
            h = downsample(h)
            if i == 3 or i == 4:
                h = self.down_attention[i - 3](h)
            hs.append(h)

        for i, (upsample, resize) in enumerate(zip(self.upsample_blocks, self.resize_blocks)):
            h_b = hs.pop()
            h = resize(torch.cat([h, h_b], dim=1), timesteps)
            h = upsample(h)
            if i == 0 or i == 1:
                h = self.up_attention[i](h)

        final = F.relu(h)
        final = self.norm(final)
        final = self.conv9(final)
        return final


def main():

    timesteps = torch.arange(0, 2)
    x = torch.randn(2, 1, 32, 32)

    model = MiniUNet(1, 64, 1, 64)

    out = model(x, timesteps)

    print(f'{out.size()=}')

if __name__ == '__main__':
    main()