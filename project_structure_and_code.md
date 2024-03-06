📂 Project Structure
├── README.md
├── archs
│   ├── __init__.py
│   ├── dat.py
│   ├── ditn.py
│   ├── omnisr.py
│   ├── realcugan.py
│   ├── rgt.py
│   ├── rrdb.py
│   ├── safmn.py
│   ├── span.py
│   ├── srvgg.py
│   ├── swinir.py
│   ├── types.py
│   └── utils
│       ├── __init__.py
│       ├── block.py
│       ├── drop.py
│       ├── state.py
│       ├── torch_internals.py
│       └── trunc.py
├── ext
│   ├── __init__.py
│   ├── image_adjustments.py
│   └── resize.py
├── requirements.txt
└── utils
    ├── __init__.py
    ├── cuda.py
    ├── file.py
    ├── image.py
    ├── tile.py
    ├── unpickler.py
    └── upscaler.py

📄 Code Files

```python
# main.py

```

```markdown
# project_structure_and_code.md

```

```markdown
# README.md
# Super Upscale
Upscaler that supports multiple Image Super-Resolution architectures.
Example of use with the Upscaler helper class:
```py
from utils.upscaler import Upscaler, UpscalerVideo

upscaler = Upscaler("./4x-esrgan-model.pth", "./inputFolder", "./outputFolder", 256, "png")
upscaler_video = UpscalerVideo("./4x-esrgan-model.pth", "./inputFolder", "./outputFolder", 256, "mp4", "libx264", "aac")

upscaler.run()
upscaler_video.run()

```

# Commits
Before commit, read [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). The repository is held to this standard.
## File Naming
All files are naming in Snake Case, architecture class names are always the same as their names. Example:
```py
# bad:
class omnisr:
  ...

# bad:
class Omnisr:
  ...

# good:
class OmniSR:
  ...
```

# Credits
Repository created for that [Colab Notebook](https://colab.research.google.com/drive/166GftgPwl0pi77mswolxhdnDQJCN2uK2?usp=sharing)

Some of the code was taken from these repositories:
* [muslll/neosr](https://github.com/muslll/neosr)
* [chaiNNer-org/chaiNNer](https://github.com/chaiNNer-org/chaiNNer)
* [sr-core](https://github.com/umzi2/sr-core)

```

```python
# archs\dat.py
# pylint: skip-file
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

from .utils.drop import DropPath
from .utils.trunc import trunc_normal_


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = (
        img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    )
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class SpatialGate(nn.Module):
    """Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
        )  # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = (
            self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W))
            .flatten(2)
            .transpose(-1, -2)
            .contiguous()
        )

        return x1 * x2


class SGFN(nn.Module):
    """Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads),
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Spatial_Attention(nn.Module):
    """Spatial Window Self-Attention.
    It supports rectangle window (containing square window).
    Args:
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """

    def __init__(
        self,
        dim,
        idx,
        split_size=[8, 8],
        dim_out=None,
        num_heads=6,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_scale=None,
        position_bias=True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer("rpe_biases", biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = (
            x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1
            )
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0
            )
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(
            -1, self.H_sp * self.W_sp, C
        )  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x


class Adaptive_Spatial_Attention(nn.Module):
    # The implementation builds on CAT code https://github.com/Zhengchen1999/CAT
    """Adaptive Spatial Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        split_size (tuple(int)): Height and Width of spatial window.
        shift_size (tuple(int)): Shift size for spatial window.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        rg_idx (int): The indentix of Residual Group (RG)
        b_idx (int): The indentix of Block in each RG
    """

    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=[8, 8],
        shift_size=[1, 2],
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        rg_idx=0,
        b_idx=0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert (
            0 <= self.shift_size[0] < self.split_size[0]
        ), "shift_size must in 0-split_size0"
        assert (
            0 <= self.shift_size[1] < self.split_size[1]
        ), "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList(
            [
                Spatial_Attention(
                    dim // 2,
                    idx=i,
                    split_size=split_size,
                    num_heads=num_heads // 2,
                    dim_out=dim // 2,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    position_bias=True,
                )
                for i in range(self.branch_num)
            ]
        )

        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
            self.rg_idx % 2 != 0 and self.b_idx % 4 == 0
        ):
            attn_mask = self.calculate_mask(
                self.patches_resolution, self.patches_resolution
            )
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
        )

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for shift window
        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
        h_slices_0 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        w_slices_0 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )

        h_slices_1 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        w_slices_1 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for window-0
        img_mask_0 = img_mask_0.view(
            1,
            H // self.split_size[0],
            self.split_size[0],
            W // self.split_size[1],
            self.split_size[1],
            1,
        )
        img_mask_0 = (
            img_mask_0.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.split_size[0], self.split_size[1], 1)
        )  # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(
            attn_mask_0 != 0, float(-100.0)
        ).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for window-1
        img_mask_1 = img_mask_1.view(
            1,
            H // self.split_size[1],
            self.split_size[1],
            W // self.split_size[0],
            self.split_size[0],
            1,
        )
        img_mask_1 = (
            img_mask_1.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.split_size[1], self.split_size[0], 1)
        )  # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(
            attn_mask_1 != 0, float(-100.0)
        ).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, HW, C
        # V without partition
        v = qkv[2].transpose(-2, -1).contiguous().view(B, C, H, W)

        # image padding
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)  # 3B C H W
        qkv = (
            F.pad(qkv, (pad_l, pad_r, pad_t, pad_b))
            .reshape(3, B, C, -1)
            .transpose(-2, -1)
        )  # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        # window-0 and window-1 on split channels [C/2, C/2]; for square windows (e.g., 8x8), window-0 and window-1 can be merged
        # shift in block: (0, 4, 8, ...), (2, 6, 10, ...), (0, 4, 8, ...), (2, 6, 10, ...), ...
        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
            self.rg_idx % 2 != 0 and self.b_idx % 4 == 0
        ):
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(
                qkv[:, :, :, :, : C // 2],
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(2, 3),
            )
            qkv_0 = qkv_0.view(3, B, _L, C // 2)
            qkv_1 = torch.roll(
                qkv[:, :, :, :, C // 2 :],
                shifts=(-self.shift_size[1], -self.shift_size[0]),
                dims=(2, 3),
            )
            qkv_1 = qkv_1.view(3, B, _L, C // 2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(
                x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)
            )
            x2 = torch.roll(
                x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2)
            )
            x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
            # attention output
            attened_x = torch.cat([x1, x2], dim=2)

        else:
            x1 = self.attns[0](qkv[:, :, :, : C // 2], _H, _W)[:, :H, :W, :].reshape(
                B, L, C // 2
            )
            x2 = self.attns[1](qkv[:, :, :, C // 2 :], _H, _W)[:, :H, :W, :].reshape(
                B, L, C // 2
            )
            # attention output
            attened_x = torch.cat([x1, x2], dim=2)

        # convolution output
        conv_x = self.dwconv(v)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        channel_map = (
            self.channel_interaction(conv_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(B, 1, C)
        )
        # S-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)

        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map)
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Adaptive_Channel_Attention(nn.Module):
    # The implementation builds on XCiT code https://github.com/facebookresearch/xcit
    """Adaptive Channel Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
        )

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # convolution output
        conv_x = self.dwconv(v_)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = (
            self.spatial_interaction(conv_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(B, N, 1)
        )

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DATB(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=[2, 4],
        shift_size=[1, 2],
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rg_idx=0,
        b_idx=0,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        if b_idx % 2 == 0:
            # DSTB
            self.attn = Adaptive_Spatial_Attention(
                dim,
                num_heads=num_heads,
                reso=reso,
                split_size=split_size,
                shift_size=shift_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                rg_idx=rg_idx,
                b_idx=b_idx,
            )
        else:
            # DCTB
            self.attn = Adaptive_Channel_Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn = SGFN(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))

        return x


class ResidualGroup(nn.Module):
    """ResidualGroup
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (tuple(int)): Height and Width of spatial window.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float): Dropout rate. Default: 0
        attn_drop(float): Attention dropout rate. Default: 0
        drop_paths (float | None): Stochastic depth rate.
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        depth (int): Number of dual aggregation Transformer blocks in residual group.
        use_chk (bool): Whether to use checkpointing to save memory.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        dim,
        reso,
        num_heads,
        split_size=[2, 4],
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_paths=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        depth=2,
        use_chk=False,
        resi_connection="1conv",
        rg_idx=0,
    ):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso

        self.blocks = nn.ModuleList(
            [
                DATB(
                    dim=dim,
                    num_heads=num_heads,
                    reso=reso,
                    split_size=split_size,
                    shift_size=[split_size[0] // 2, split_size[1] // 2],
                    expansion_factor=expansion_factor,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_paths[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    rg_idx=rg_idx,
                    b_idx=i,
                )
                for i in range(depth)
            ]
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        res = x
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = res + x

        return x


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class DAT(nn.Module):
    """Dual Aggregation Transformer
    Args:
        img_size (int): Input image size. Default: 64
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 180
        depths (tuple(int)): Depth of each residual group (number of DATB in each RG).
        split_size (tuple(int)): Height and Width of spatial window.
        num_heads (tuple(int)): Number of attention heads in different residual groups.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        use_chk (bool): Whether to use checkpointing to save memory.
        upscale: Upscale factor. 2/3/4 for image SR
        img_range: Image range. 1. or 255.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, state_dict):
        super().__init__()

        # defaults
        img_size = 64
        in_chans = 3
        embed_dim = 180
        split_size = [2, 4]
        depth = [2, 2, 2, 2]
        num_heads = [2, 2, 2, 2]
        expansion_factor = 4.0
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.1
        act_layer = nn.GELU
        norm_layer = nn.LayerNorm
        use_chk = False
        upscale = 2
        img_range = 1.0
        resi_connection = "1conv"
        upsampler = "pixelshuffle"

        self.model_arch = "DAT"
        self.name = "DAT"
        self.sub_type = "SR"
        self.state = state_dict
        state_keys = state_dict.keys()
        if "conv_before_upsample.0.weight" in state_keys:
            if "conv_up1.weight" in state_keys:
                upsampler = "nearest+conv"
            else:
                upsampler = "pixelshuffle"
                supports_fp16 = False
        elif "upsample.0.weight" in state_keys:
            upsampler = "pixelshuffledirect"
        else:
            upsampler = ""

        num_feat = (
            state_dict.get("conv_before_upsample.0.weight", None).shape[1]
            if state_dict.get("conv_before_upsample.weight", None)
            else 64
        )
   
        num_in_ch = state_dict["conv_first.weight"].shape[1]
        in_chans = num_in_ch
        self.input_channels = in_chans
        if "conv_last.weight" in state_keys:
            num_out_ch = state_dict["conv_last.weight"].shape[0]
        else:
            num_out_ch = num_in_ch

        upscale = 1
        if upsampler == "nearest+conv":
            upsample_keys = [
                x for x in state_keys if "conv_up" in x and "bias" not in x
            ]

            for upsample_key in upsample_keys:
                upscale *= 2
        elif upsampler == "pixelshuffle":
            upsample_keys = [
                x
                for x in state_keys
                if "upsample" in x and "conv" not in x and "bias" not in x
            ]
            for upsample_key in upsample_keys:
                shape = state_dict[upsample_key].shape[0]
                upscale *= math.sqrt(shape // num_feat)
            upscale = int(upscale)
        elif upsampler == "pixelshuffledirect":
            upscale = int(
                math.sqrt(state_dict["upsample.0.bias"].shape[0] // num_out_ch)
            )

        max_layer_num = 0
        max_block_num = 0
        for key in state_keys:
            result = re.match(r"layers.(\d*).blocks.(\d*).norm1.weight", key)
            if result:
                layer_num, block_num = result.groups()
                max_layer_num = max(max_layer_num, int(layer_num))
                max_block_num = max(max_block_num, int(block_num))

        depth = [max_block_num + 1 for _ in range(max_layer_num + 1)]

        if "layers.0.blocks.1.attn.temperature" in state_keys:
            num_heads_num = state_dict["layers.0.blocks.1.attn.temperature"].shape[0]
            num_heads = [num_heads_num for _ in range(max_layer_num + 1)]
        else:
            num_heads = depth

        embed_dim = state_dict["conv_first.weight"].shape[0]
        expansion_factor = float(
            state_dict["layers.0.blocks.0.ffn.fc1.weight"].shape[0] / embed_dim
        )

        # TODO: could actually count the layers, but this should do
        if "layers.0.conv.4.weight" in state_keys:
            resi_connection = "3conv"
        else:
            resi_connection = "1conv"

        if "layers.0.blocks.2.attn.attn_mask_0" in state_keys:
            attn_mask_0_x, attn_mask_0_y, attn_mask_0_z = state_dict[
                "layers.0.blocks.2.attn.attn_mask_0"
            ].shape

            img_size = int(math.sqrt(attn_mask_0_x * attn_mask_0_y))

        if "layers.0.blocks.0.attn.attns.0.rpe_biases" in state_keys:
            split_sizes = (
                state_dict["layers.0.blocks.0.attn.attns.0.rpe_biases"][-1] + 1
            )
            split_size = [int(x) for x in split_sizes]

        self.in_nc = num_in_ch
        self.out_nc = num_out_ch
        self.num_feat = num_feat
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.scale = upscale
        self.upsampler = upsampler
        self.img_size = img_size
        self.img_range = img_range
        self.expansion_factor = expansion_factor
        self.resi_connection = resi_connection
        self.split_size = split_size

        self.supports_fp16 = False  # Too much weirdness to support this at the moment
        self.supports_bfp16 = True
        self.min_size_restriction = 16

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        heads = num_heads

        self.before_RG = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"), nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                split_size=split_size,
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]) : sum(depth[: i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                rg_idx=i,
            )
            self.layers.append(layer)

        self.norm = norm_layer(curr_dim)
        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # ------------------------- 3, Reconstruction ------------------------- #
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch, (img_size, img_size)
            )

        self.apply(self._init_weights)
        self.load_state_dict(state_dict, strict=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(
            m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)
        ):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        """
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for image SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        x = x / self.img_range + self.mean
        return x

```

```python
# archs\ditn.py
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from archs.utils.state import get_seq_len


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ISA(nn.Module):
    def __init__(self, dim, bias):
        super(ISA, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.data.shape
        x = x.view(b, c, -1).transpose(-1, -2)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # flash attention
        out = F.scaled_dot_product_attention(q, k, v)
        # original:
        # out = (attn @ v)
        out = out.view(b, c, h, w)

        out = self.project_out(out)
        return out


class SDA(nn.Module):
    def __init__(self, n_feats, LayerNorm_type="WithBias"):
        super(SDA, self).__init__()
        i_feats = 2 * n_feats
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.DConvs = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 5, 1, 5 // 2, groups=n_feats),
            nn.Conv2d(
                n_feats,
                n_feats,
                7,
                stride=1,
                padding=(7 // 2) * 3,
                groups=n_feats,
                dilation=3,
            ),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
        )

        self.proj_first = nn.Sequential(nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1, 1, 0))
        self.dim = n_feats

    def forward(self, x):
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a = self.DConvs(a)
        x = self.proj_last(x * a) * self.scale

        return x


class ITL(nn.Module):
    def __init__(self, n_feats, ffn_expansion_factor, bias, LayerNorm_type):
        super(ITL, self).__init__()
        self.attn = ISA(n_feats, bias)
        self.act = nn.Tanh()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 1)

        self.ffn = FeedForward(n_feats, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.conv1(self.act(x)))
        x = x + self.ffn(self.conv2(self.act(x)))
        return x


class SAL(nn.Module):
    def __init__(self, n_feats, ffn_expansion_factor, bias, LayerNorm_type):
        super(SAL, self).__init__()
        self.SDA = SDA(n_feats)
        self.ffn = FeedForward(n_feats, ffn_expansion_factor, bias)
        self.act = nn.Tanh()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        x = x + self.SDA(self.conv1(self.act(x)))
        x = x + self.ffn(self.conv2(self.act(x)))
        return x


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))

        super(UpsampleOneStep, self).__init__(*m)


class UFONE(nn.Module):
    def __init__(
        self,
        dim,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        ITL_blocks,
        SAL_blocks,
        patch_size,
    ):
        super(UFONE, self).__init__()
        ITL_body = [
            ITL(dim, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(ITL_blocks)
        ]
        self.ITLs = nn.Sequential(*ITL_body)
        SAL_body = [
            SAL(dim, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(SAL_blocks)
        ]
        self.SALs = nn.Sequential(*SAL_body)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.data.shape
        local_features = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        local_features = (
            local_features.permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(-1, C, self.patch_size, self.patch_size)
        )
        local_features = self.ITLs(local_features)
        local_features = local_features.view(
            B,
            H // self.patch_size,
            W // self.patch_size,
            C,
            self.patch_size,
            self.patch_size,
        )
        local_features = (
            local_features.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        )
        local_features = self.SALs(local_features)
        return local_features


class DITN(nn.Module):
    def __init__(self, state_dict, **kwargs):
        super(DITN, self).__init__()

        inp_channels = state_dict["sft.weight"].shape[1]
        dim = state_dict["sft.weight"].shape[0]
        UFONE_blocks = get_seq_len(state_dict, "UFONE")
        ITL_blocks = get_seq_len(state_dict, "UFONE.0.ITLs")
        SAL_blocks = get_seq_len(state_dict, "UFONE.0.SALs")
        ffn_expansion_factor = (
            state_dict["UFONE.0.ITLs.0.ffn.project_in.weight"].shape[0] / 2 / dim
        )
        bias = "UFONE.0.ITLs.0.attn.project_out.bias" in state_dict
        LayerNorm_type = "WithBias"
        patch_size = 8
        upscale = int(math.sqrt(state_dict["upsample.0.weight"].shape[0] / 3))

        self.patch_size = patch_size
        self.sft = nn.Conv2d(inp_channels, dim, 3, 1, 1)

        ## UFONE Block1
        UFONE_body = [
            UFONE(
                dim,
                ffn_expansion_factor,
                bias,
                LayerNorm_type,
                ITL_blocks,
                SAL_blocks,
                patch_size,
            )
            for _ in range(UFONE_blocks)
        ]
        self.UFONE = nn.Sequential(*UFONE_body)

        self.conv_after_body = nn.Conv2d(dim, dim, 3, 1, 1)
        # drop out
        self.dropout = nn.Dropout2d(p=0.5)
        self.upsample = UpsampleOneStep(upscale, dim, 3)
        self.dim = dim
        self.patch_sizes = [8, 8]
        self.scale = upscale
        self.SAL_blocks = SAL_blocks
        self.ITL_blocks = ITL_blocks
        self.input_channels = inp_channels
        self.name = "DITN"

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.patch_sizes[0]
        for i in range(1, len(self.patch_sizes)):
            wsize = wsize * self.patch_sizes[i] // math.gcd(wsize, self.patch_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward(self, inp_img):
        _, _, old_h, old_w = inp_img.shape
        inp_img = self.check_image_size(inp_img)
        sft = self.sft(inp_img)

        local_features = self.UFONE(sft)

        # stochastic depth
        # stochastic_depth(local_features, p=0.5, mode="batch")
        # dropout
        # local_features = self.dropout(local_features)

        local_features = self.conv_after_body(local_features)

        out_dec_level1 = self.upsample(local_features + sft)

        return out_dec_level1[:, :, 0 : old_h * self.scale, 0 : old_w * self.scale]

```

```python
# archs\omnisr.py
# Code adapted from: https://github.com/Francis0625/Omni-SR

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


def pixelshuffle_block(
    in_channels, out_channels, upscale_factor=2, kernel_size=3, bias=False
):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(
        in_channels,
        out_channels * (upscale_factor**2),
        kernel_size,
        padding=1,
        bias=bias,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    return nn.Sequential(*[conv, pixel_shuffle])


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(
            c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Conv_PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.0):
        super().__init__()

        hidden_features = int(dim * mult)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = (
            torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_()
            > self.prob
        )
        return x * keep_mask / (1 - self.prob)


def MBConv(
    dim_in, dim_out, *, downsample, expansion_rate=4, shrinkage_rate=0.25, dropout=0.0
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(
            hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim
        ),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        # nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0.0, window_size=7, with_pe=True):
        super().__init__()
        assert (
            dim % dim_head
        ) == 0, "dimension should be divisible by dimension per head"

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout)
        )

        # relative positional bias
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
            grid = rearrange(grid, "c i j -> (i j) c")
            rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(
                grid, "j ... -> 1 j ..."
            )
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(
                dim=-1
            )

            self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = (
            *x.shape,
            x.device,
            self.heads,
        )

        # flatten
        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, "b n (h d ) -> b h n d", h=h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, "i j h -> h i j")

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads
        out = rearrange(
            out, "b h (w1 w2) d -> b w1 w2 (h d)", w1=window_height, w2=window_width
        )

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, "(b x y) ... -> b x y ...", x=height, y=width)


class Channel_Attention(nn.Module):
    def __init__(self, dim, heads, bias=False, dropout=0.0, window_size=7):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)",
                ph=self.ps,
                pw=self.ps,
                head=self.heads,
            ),
            qkv,
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out,
            "b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)",
            h=h // self.ps,
            w=w // self.ps,
            ph=self.ps,
            pw=self.ps,
            head=self.heads,
        )

        out = self.project_out(out)

        return out


class Channel_Attention_grid(nn.Module):
    def __init__(self, dim, heads, bias=False, dropout=0.0, window_size=7):
        super(Channel_Attention_grid, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)",
                ph=self.ps,
                pw=self.ps,
                head=self.heads,
            ),
            qkv,
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out,
            "b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)",
            h=h // self.ps,
            w=w // self.ps,
            ph=self.ps,
            pw=self.ps,
            head=self.heads,
        )

        out = self.project_out(out)

        return out


class OSA_Block(nn.Module):
    def __init__(
        self,
        channel_num=64,
        bias=True,
        ffn_bias=True,
        window_size=8,
        with_pe=False,
        dropout=0.0,
    ):
        super(OSA_Block, self).__init__()

        w = window_size

        self.layer = nn.Sequential(
            MBConv(
                channel_num,
                channel_num,
                downsample=False,
                expansion_rate=1,
                shrinkage_rate=0.25,
            ),
            # block-like attention
            Rearrange("b d (x w1) (y w2) -> b x y w1 w2 d", w1=w, w2=w),
            PreNormResidual(
                channel_num,
                Attention(
                    dim=channel_num,
                    dim_head=channel_num // 4,
                    dropout=dropout,
                    window_size=window_size,
                    with_pe=with_pe,
                ),
            ),
            Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
            Conv_PreNormResidual(
                channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)
            ),
            # channel-like attention
            Conv_PreNormResidual(
                channel_num,
                Channel_Attention(
                    dim=channel_num, heads=4, dropout=dropout, window_size=window_size
                ),
            ),
            Conv_PreNormResidual(
                channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)
            ),
            # grid-like attention
            Rearrange("b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=w, w2=w),
            PreNormResidual(
                channel_num,
                Attention(
                    dim=channel_num,
                    dim_head=channel_num // 4,
                    dropout=dropout,
                    window_size=window_size,
                    with_pe=with_pe,
                ),
            ),
            Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"),
            Conv_PreNormResidual(
                channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)
            ),
            # channel-like attention
            Conv_PreNormResidual(
                channel_num,
                Channel_Attention_grid(
                    dim=channel_num, heads=4, dropout=dropout, window_size=window_size
                ),
            ),
            Conv_PreNormResidual(
                channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)
            ),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class OSAG(nn.Module):
    def __init__(
        self,
        channel_num=64,
        bias=True,
        block_num=4,
        ffn_bias=True,
        pe=True,
        window_size=8,
        **kwargs
    ):
        super(OSAG, self).__init__()

        group_list = []
        for _ in range(block_num):
            temp_res = OSA_Block(
                channel_num,
                bias,
                ffn_bias=ffn_bias,
                window_size=window_size,
                with_pe=pe,
            )
            group_list.append(temp_res)
        group_list.append(nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=bias))
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)

    def forward(self, x):
        out = self.residual_layer(x)
        out = out + x
        return self.esa(out)


class OmniSR(nn.Module):
    def __init__(self, state_dict, **kwargs):
        super(OmniSR, self).__init__()

        bias = True
        block_num = 1
        ffn_bias = True
        pe = True

        num_feat = state_dict["input.weight"].shape[0] or 64
        num_in_ch = state_dict["input.weight"].shape[1] or 3
        num_out_ch = num_in_ch

        pixelshuffle_shape = state_dict["up.0.weight"].shape[0]
        up_scale = math.sqrt(pixelshuffle_shape / num_out_ch)
        if up_scale - int(up_scale) > 0:
            print(
                "out_nc is probably different than in_nc, scale calculation might be"
                " wrong"
            )
        up_scale = int(up_scale)
        res_num = 0
        for key in state_dict.keys():
            if "residual_layer" in key:
                temp_res_num = int(key.split(".")[1])
                if temp_res_num > res_num:
                    res_num = temp_res_num
        res_num = res_num + 1  # zero-indexed

        self.res_num = res_num

        if (
            "residual_layer.0.residual_layer.0.layer.2.fn.rel_pos_bias.weight"
            in state_dict.keys()
        ):
            rel_pos_bias_weight = state_dict[
                "residual_layer.0.residual_layer.0.layer.2.fn.rel_pos_bias.weight"
            ].shape[0]
            self.window_size = int((math.sqrt(rel_pos_bias_weight) + 1) / 2)
        else:
            self.window_size = 8

        self.up_scale = up_scale
        residual_layer = []




        for _ in range(res_num):
            temp_res = OSAG(
                channel_num=num_feat,
                bias=bias,
                block_num=block_num,
                window_size=self.window_size,
                pe=pe,
            )
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(
            in_channels=num_in_ch,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.output = nn.Conv2d(
            in_channels=num_feat,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)
        self.input_channels = num_in_ch
        self.name = "OmniSR"

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant", 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)

        out = out[:, :, : H * self.up_scale, : W * self.up_scale]
        return out

```

```python
# archs\realcugan.py

import torch
from torch import nn as nn
from torch.nn import functional as F




class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels // reduction, 1, 1, 0, bias=bias
        )
        self.conv2 = nn.Conv2d(
            in_channels // reduction, in_channels, 1, 1, 0, bias=bias
        )

    def forward(self, x):
        if "Half" in x.type():
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

    def forward_mean(self, x, x0):
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x


class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z


class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x1, x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z


class UNet1x3(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1x3, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 5, 3, 2)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x1, x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z


class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet2, self).__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)

        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4(x2 + x3)
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x2):
        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3.conv(x3)
        return x3

    def forward_c(self, x2, x3):
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def forward_d(self, x1, x4):
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

class cugan(nn.Module):
    def __init__(self, state_dict):
        super(cugan, self).__init__()
        self.name = "cugan"

        if "conv_final.weight" in state_dict:
            # UpCunet4x
            scale = 4
            in_channels = state_dict["unet1.conv1.conv.0.weight"].shape[1]
            out_channels = 3  # hard coded in UpCunet4x
        elif state_dict["unet1.conv_bottom.weight"].shape[2] == 5:
            # UpCunet3x
            scale = 3
            in_channels = state_dict["unet1.conv1.conv.0.weight"].shape[1]
            out_channels = state_dict["unet2.conv_bottom.weight"].shape[0]
        else:
            # UpCunet2x
            scale = 2
            in_channels = state_dict["unet1.conv1.conv.0.weight"].shape[1]
            out_channels = state_dict["unet2.conv_bottom.weight"].shape[0]

        self.input_channels = in_channels
        self.scale = scale
        pro = False
        if list(state_dict.keys())[-1] == "unet2.conv_bottom.bias" or "pro" in state_dict:
            pro = True
        self.pro_mode = pro
        if self.scale == 1:
            raise ValueError(f'1x scale ratio is unsupported. Please use 2x, 3x or 4x.')

        if self.scale == 2:
            self.unet1 = UNet1(in_channels, out_channels, deconv=True)
            self.unet2 = UNet2(in_channels, out_channels, deconv=False)

        if self.scale == 3:
            self.unet1 = UNet1x3(in_channels, out_channels, deconv=True)
            self.unet2 = UNet2(in_channels, out_channels, deconv=False)

        if self.scale == 4:
            self.ps = nn.PixelShuffle(2)
            self.conv_final = nn.Conv2d(64, 12, 3, 1, padding=0, bias=True)
            self.unet1 = UNet1(in_channels, 64, deconv=True)
            self.unet2 = UNet2(64, 64, deconv=False)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)

        if self.pro_mode:
            x = (x * 0.7) + 0.15

        n, c, h0, w0 = x.shape
        x00 = x

        if self.scale == 3:
            ph = ((h0 - 1) // 2 + 1) * 4
            pw = ((w0 - 1) // 2 + 1) * 4 
        else:
            ph = ((h0 - 1) // 2 + 1) * 2
            pw = ((w0 - 1) // 2 + 1) * 2

        if self.scale == 2:
            x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), "reflect")
        if self.scale == 3:
            x = F.pad(x, (14, 14 + pw - w0, 14, 14 + ph - h0), "reflect")
        if self.scale == 4:
            x = F.pad(x, (19, 19 + pw - w0, 19, 19 + ph - h0), "reflect")

        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)

        if self.scale == 4:
            x = self.conv_final(x)
            x = F.pad(x, (-1, -1, -1, -1))
            x = self.ps(x)

        if w0 != pw or h0 != ph:
            x = x[:, :, : h0 * self.scale, : w0 * self.scale]

        if self.scale == 4:
            x += F.interpolate(x00, scale_factor=4, mode="nearest")

        if self.pro_mode:
            x = (x - 0.15) / 0.7

        return x


```

```python
# archs\rgt.py
import math
import re

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torch.utils import checkpoint
from .utils.drop import DropPath


def img2windows(img, H_sp, W_sp):
    """Input: Image (B, C, H, W).

    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = (
        img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    )
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """Input: Window Partition (B', N, C).
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
        )  # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = (
            self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W))
            .flatten(2)
            .transpose(-1, -2)
            .contiguous()
        )

        return x1 * x2


class MLP(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """Dynamic Relative Position Bias.

    Args:
    ----
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads),
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class WindowAttention(nn.Module):
    def __init__(
            self,
            dim,
            idx,
            split_size=[8, 8],
            dim_out=None,
            num_heads=6,
            attn_drop=0.0,
            proj_drop=0.0,
            qk_scale=None,
            position_bias=True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(
                torch.meshgrid([position_bias_h, position_bias_w], indexing="ij")
            )
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer("rpe_biases", biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = (
            x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return x

    def forward(self, qkv, H, W, mask=None):
        """Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1
            )
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0
            )
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(
            -1, self.H_sp * self.W_sp, C
        )  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x


class L_SA(nn.Module):
    # The implementation builds on CAT code https://github.com/zhengchen1999/CAT/blob/main/basicsr/archs/cat_arch.py
    def __init__(
            self,
            dim,
            num_heads,
            split_size=[2, 4],
            shift_size=[1, 2],
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            idx=0,
            reso=64,
            rs_id=0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.idx = idx
        self.rs_id = rs_id
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert (
                0 <= self.shift_size[0] < self.split_size[0]
        ), "shift_size must in 0-split_size0"
        assert (
                0 <= self.shift_size[1] < self.split_size[1]
        ), "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            WindowAttention(
                dim // 2,
                idx=i,
                split_size=split_size,
                num_heads=num_heads // 2,
                dim_out=dim // 2,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                position_bias=True,
            )
            for i in range(self.branch_num)
        ])

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (
                self.rs_id % 2 != 0 and self.idx % 4 == 0
        ):
            attn_mask = self.calculate_mask(
                self.patches_resolution, self.patches_resolution
            )

            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None

            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.get_v = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
        )  # DW Conv

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for Rwin
        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
        h_slices_0 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        w_slices_0 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )

        h_slices_1 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        w_slices_1 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for H-Shift
        img_mask_0 = img_mask_0.view(
            1,
            H // self.split_size[0],
            self.split_size[0],
            W // self.split_size[1],
            self.split_size[1],
            1,
        )
        img_mask_0 = (
            img_mask_0.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.split_size[0], self.split_size[1], 1)
        )  # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(
            attn_mask_0 != 0, float(-100.0)
        ).masked_fill(attn_mask_0 == 0, 0.0)

        # calculate mask for V-Shift
        img_mask_1 = img_mask_1.view(
            1,
            H // self.split_size[1],
            self.split_size[1],
            W // self.split_size[0],
            self.split_size[0],
            1,
        )
        img_mask_1 = (
            img_mask_1.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.split_size[1], self.split_size[0], 1)
        )  # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(
            attn_mask_1 != 0, float(-100.0)
        ).masked_fill(attn_mask_1 == 0, 0.0)

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        """Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, HW, C
        # v without partition
        v = qkv[2].transpose(-2, -1).contiguous().view(B, C, H, W)

        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)  # 3B C H W
        qkv = (
            F.pad(qkv, (pad_l, pad_r, pad_t, pad_b))
            .reshape(3, B, C, -1)
            .transpose(-2, -1)
        )  # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (
                self.rs_id % 2 != 0 and self.idx % 4 == 0
        ):
            qkv = qkv.view(3, B, _H, _W, C)
            # H-Shift
            qkv_0 = torch.roll(
                qkv[:, :, :, :, : C // 2],
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(2, 3),
            )
            qkv_0 = qkv_0.view(3, B, _L, C // 2)
            # V-Shift
            qkv_1 = torch.roll(
                qkv[:, :, :, :, C // 2:],
                shifts=(-self.shift_size[1], -self.shift_size[0]),
                dims=(2, 3),
            )
            qkv_1 = qkv_1.view(3, B, _L, C // 2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))

            else:
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(
                x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)
            )
            x2 = torch.roll(
                x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2)
            )
            x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
            # Concat
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            # V-Rwin
            x1 = self.attns[0](qkv[:, :, :, : C // 2], _H, _W)[:, :H, :W, :].reshape(
                B, L, C // 2
            )
            # H-Rwin
            x2 = self.attns[1](qkv[:, :, :, C // 2:], _H, _W)[:, :H, :W, :].reshape(
                B, L, C // 2
            )
            # Concat
            attened_x = torch.cat([x1, x2], dim=2)

        # mix
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + lcm

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RG_SA(nn.Module):
    """Recursive-Generalization Self-Attention (RG-SA).

    Args:
    ----
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        c_ratio (float): channel adjustment factor.
    """

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            c_ratio=0.5,
    ):
        super(RG_SA, self).__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.cr = int(dim * c_ratio)  # scaled channel dimension

        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        # RGM
        self.reduction1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4, groups=dim)
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
        )
        self.conv = nn.Conv2d(dim, self.cr, kernel_size=1, stride=1)
        self.norm_act = nn.Sequential(nn.LayerNorm(self.cr), nn.GELU())
        # CA
        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
        self.v = nn.Linear(self.cr, dim, bias=qkv_bias)

        # CPE
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        _scale = 1

        # reduction
        _x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        if self.training:
            _time = max(int(math.log(H // 4, 4)), int(math.log(W // 4, 4)))
        else:
            _time = max(int(math.log(H // 16, 4)), int(math.log(W // 16, 4)))
            if _time < 2:
                _time = 2  # testing _time must equal or larger than training _time (2)

        _scale = 4 ** _time

        # Recursion xT
        for _ in range(_time):
            _x = self.reduction1(_x)

        _x = (
            self.conv(self.dwconv(_x))
            .reshape(B, self.cr, -1)
            .permute(0, 2, 1)
            .contiguous()
        )  # shape=(B, N', C')
        _x = self.norm_act(_x)

        # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(B, N', C)
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, int(self.cr / self.num_heads))
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(_x)
            .reshape(B, -1, self.num_heads, int(self.cr / self.num_heads))
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(_x)
            .reshape(B, -1, self.num_heads, int(C / self.num_heads))
            .permute(0, 2, 1, 3)
        )

        # corss-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # CPE
        # v_shape=(B, H, N', C//H)
        v = v + self.cpe(
            v.transpose(1, 2)
            .reshape(B, -1, C)
            .transpose(1, 2)
            .contiguous()
            .view(B, C, H // _scale, W // _scale)
        ).view(B, C, -1).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(
            -1, -2
        )

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            idx=0,
            rs_id=0,
            split_size=[2, 4],
            shift_size=[1, 2],
            reso=64,
            c_ratio=0.5,
            layerscale_value=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if idx % 2 == 0:
            self.attn = L_SA(
                dim,
                split_size=split_size,
                shift_size=shift_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                drop=drop,
                idx=idx,
                reso=reso,
                rs_id=rs_id,
            )
        else:
            self.attn = RG_SA(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                c_ratio=c_ratio,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
        )
        self.norm2 = norm_layer(dim)

        # HAI
        self.gamma = nn.Parameter(
            layerscale_value * torch.ones(dim), requires_grad=True
        )

    def forward(self, x, x_size):
        H, W = x_size

        res = x

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # HAI
        x = x + (res * self.gamma)

        return x


class ResidualGroup(nn.Module):
    def __init__(
            self,
            dim,
            reso,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_paths=None,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            depth=2,
            use_chk=False,
            resi_connection="1conv",
            rs_id=0,
            split_size=[8, 8],
            c_ratio=0.5,
    ):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso

        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                idx=i,
                rs_id=rs_id,
                split_size=split_size,
                shift_size=[split_size[0] // 2, split_size[1] // 2],
                c_ratio=c_ratio,
            )
            for i in range(depth)
        ])

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

    def forward(self, x, x_size):
        """Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        res = x
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = res + x

        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
    ----
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. " "Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)


class RGT(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        img_size = 64
        in_chans = state_dict["conv_first.weight"].shape[1]
        embed_dim = 180
        mlp_ratio = 2
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.1
        act_layer = nn.GELU
        norm_layer = nn.LayerNorm
        use_chk = False
        img_range = 1.0
        split_size = [8, 32]
        c_ratio = 0.5
        num_in_ch = in_chans
        max_layer_num = 0
        max_block_num = 0
        self.name = "RGT"
        self.input_channels = num_in_ch
        self.state = state_dict
        state_keys = state_dict.keys()
        for key in state_keys:
            result = re.match(r"layers.(\d*).blocks.(\d*).norm1.weight", key)
            if result:
                layer_num, block_num = result.groups()
                max_layer_num = max(max_layer_num, int(layer_num))
                max_block_num = max(max_block_num, int(block_num))

        depth = [max_block_num + 1 for _ in range(max_layer_num + 1)]
        num_feat = (
            state_dict.get("conv_before_upsample.0.weight", None).shape[1]
            if state_dict.get("conv_before_upsample.weight", None)
            else 64
        )
        if "layers.0.blocks.1.attn.temperature" in state_keys:
            num_heads_num = state_dict["layers.0.blocks.1.attn.temperature"].shape[0]
            num_heads = [num_heads_num for _ in range(max_layer_num + 1)]
        else:
            num_heads = depth
        if "conv_before_upsample.0.weight" in state_keys:
            if "conv_up1.weight" in state_keys:
                upsampler = "nearest+conv"
            else:
                upsampler = "pixelshuffle"
        elif "upsample.0.weight" in state_keys:
            upsampler = "pixelshuffledirect"
        else:
            upsampler = ""
        if "conv_last.weight" in state_keys:
            num_out_ch = state_dict["conv_last.weight"].shape[0]
        else:
            num_out_ch = num_in_ch

        upscale = 1
        if upsampler == "nearest+conv":
            upsample_keys = [
                x for x in state_keys if "conv_up" in x and "bias" not in x
            ]

            for upsample_key in upsample_keys:
                upscale *= 2
        elif upsampler == "pixelshuffle":
            upsample_keys = [
                x
                for x in state_keys
                if "upsample" in x and "conv" not in x and "bias" not in x
            ]
            for upsample_key in upsample_keys:
                shape = state_dict[upsample_key].shape[0]
                upscale *= math.sqrt(shape // num_feat)
            upscale = int(upscale)
        elif upsampler == "pixelshuffledirect":
            upscale = int(
                math.sqrt(state_dict["upsample.0.bias"].shape[0] // num_out_ch)
            )

        num_feat = 64

        if "layers.0.blocks.2.attn.attn_mask_0" in state_keys:
            attn_mask_0_x, attn_mask_0_y, attn_mask_0_z = state_dict[
                "layers.0.blocks.2.attn.attn_mask_0"
            ].shape

            img_size = int(math.sqrt(attn_mask_0_x * attn_mask_0_y))

        if "layers.0.blocks.0.attn.attns.0.rpe_biases" in state_keys:
            split_sizes = (
                    state_dict["layers.0.blocks.0.attn.attns.0.rpe_biases"][-1] + 1
            )
            split_size = [int(x) for x in split_sizes]
        if "layers.0.conv.4.weight" in state_keys:
            resi_connection = "3conv"
        else:
            resi_connection = "1conv"

        if "layers.0.blocks.2.attn.attn_mask_0" in state_keys:
            attn_mask_0_x, attn_mask_0_y, attn_mask_0_z = state_dict[
                "layers.0.blocks.2.attn.attn_mask_0"
            ].shape

            img_size = int(math.sqrt(attn_mask_0_x * attn_mask_0_y))

        if "layers.0.blocks.0.attn.attns.0.rpe_biases" in state_keys:
            split_sizes = (
                    state_dict["layers.0.blocks.0.attn.attns.0.rpe_biases"][-1] + 1
            )
            split_size = [int(x) for x in split_sizes]
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale

        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        heads = num_heads

        self.before_RG = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"), nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]): sum(depth[: i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                rs_id=i,
                split_size=split_size,
                c_ratio=c_ratio,
            )
            self.layers.append(layer)
        self.norm = norm_layer(curr_dim)
        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # ------------------------- 3, Reconstruction ------------------------- #
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(
                m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)
        ):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x

    def forward(self, x):
        """Input: x: (B, C, H, W)
        """
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x

# @ARCH_REGISTRY.register()
# def rgt_s(**kwargs):
#    return rgt(depth=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6], **kwargs)

```

```python
# archs\rrdb.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import math
import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import block as B


# Borrowed from https://github.com/rlaphoenix/VSGAN/blob/master/vsgan/archs/ESRGAN.py
# Which enhanced stuff that was already here
class RRDBNet(nn.Module):
    def __init__(
        self,
        state_dict,
        norm=None,
        act: str = "leakyrelu",
        upsampler: str = "upconv",
        mode: B.ConvMode = "CNA",
    ) -> None:
        """
        ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
        By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
        and Chen Change Loy.
        This is old-arch Residual in Residual Dense Block Network and is not
        the newest revision that's available at github.com/xinntao/ESRGAN.
        This is on purpose, the newest Network has severely limited the
        potential use of the Network with no benefits.
        This network supports model files from both new and old-arch.
        Args:
            norm: Normalization layer
            act: Activation layer
            upsampler: Upsample layer. upconv, pixel_shuffle
            mode: Convolution mode
        """
        super(RRDBNet, self).__init__()
        self.model_arch = "ESRGAN"
        self.name = "ESRGAN"
        self.sub_type = "SR"

        self.state = state_dict
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode

        self.state_map = {
            # currently supports old, new, and newer RRDBNet arch models
            # ESRGAN, BSRGAN/RealSR, Real-ESRGAN
            "model.0.weight": ("conv_first.weight",),
            "model.0.bias": ("conv_first.bias",),
            "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
            "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
            r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
                r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
                r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)",
            ),
        }
        if "params_ema" in self.state:
            self.state = self.state["params_ema"]
            # self.model_arch = "RealESRGAN"
        self.num_blocks = self.get_num_blocks()
        self.plus = any("conv1x1" in k for k in self.state.keys())
        if self.plus:
            self.model_arch = "ESRGAN+"

        self.state = self.new_to_old_arch(self.state)

        highest_weight_num = max(
            int(re.search(r"model.(\d+)", k).group(1)) for k in self.state
        )

        self.in_nc: int = self.state["model.0.weight"].shape[1]
        self.input_channels = self.in_nc
        self.out_nc: int = self.state[f"model.{highest_weight_num}.bias"].shape[0]

        self.scale: int = self.get_scale()
        self.num_filters: int = self.state["model.0.weight"].shape[0]

        c2x2 = False
        if self.state["model.0.weight"].shape[-2] == 2:
            c2x2 = True
            self.scale = round(math.sqrt(self.scale / 4))
            self.model_arch = "ESRGAN-2c2"

        self.supports_fp16 = True
        self.supports_bfp16 = True
        self.min_size_restriction = None

        # Detect if pixelunshuffle was used (Real-ESRGAN)
        if self.in_nc in (self.out_nc * 4, self.out_nc * 16) and self.out_nc in (
            self.in_nc / 4,
            self.in_nc / 16,
        ):
            self.shuffle_factor = int(math.sqrt(self.in_nc / self.out_nc))
        else:
            self.shuffle_factor = None

        upsample_block = {
            "upconv": B.upconv_block,
            "pixel_shuffle": B.pixelshuffle_block,
        }.get(self.upsampler)
        if upsample_block is None:
            raise NotImplementedError(f"Upsample mode [{self.upsampler}] is not found")

        if self.scale == 3:
            upsample_blocks = upsample_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                upscale_factor=3,
                act_type=self.act,
                c2x2=c2x2,
            )
        else:
            upsample_blocks = [
                upsample_block(
                    in_nc=self.num_filters,
                    out_nc=self.num_filters,
                    act_type=self.act,
                    c2x2=c2x2,
                )
                for _ in range(int(math.log(self.scale, 2)))
            ]

        self.model = B.sequential(
            # fea conv
            B.conv_block(
                in_nc=self.in_nc,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
            B.ShortcutBlock(
                B.sequential(
                    # rrdb blocks
                    *[
                        B.RRDB(
                            nf=self.num_filters,
                            kernel_size=3,
                            gc=32,
                            stride=1,
                            bias=True,
                            pad_type="zero",
                            norm_type=self.norm,
                            act_type=self.act,
                            mode="CNA",
                            plus=self.plus,
                            c2x2=c2x2,
                        )
                        for _ in range(self.num_blocks)
                    ],
                    # lr conv
                    B.conv_block(
                        in_nc=self.num_filters,
                        out_nc=self.num_filters,
                        kernel_size=3,
                        norm_type=self.norm,
                        act_type=None,
                        mode=self.mode,
                        c2x2=c2x2,
                    ),
                )
            ),
            *upsample_blocks,
            # hr_conv0
            B.conv_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=self.act,
                c2x2=c2x2,
            ),
            # hr_conv1
            B.conv_block(
                in_nc=self.num_filters,
                out_nc=self.out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
        )

        # Adjust these properties for calculations outside of the model
        if self.shuffle_factor:
            self.in_nc //= self.shuffle_factor**2
            self.scale //= self.shuffle_factor

        self.load_state_dict(self.state, strict=False)

    def new_to_old_arch(self, state):
        """Convert a new-arch model state dictionary to an old-arch dictionary."""
        if "params_ema" in state:
            state = state["params_ema"]

        if "conv_first.weight" not in state:
            # model is already old arch, this is a loose check, but should be sufficient
            return state

        # add nb to state keys
        for kind in ("weight", "bias"):
            self.state_map[f"model.1.sub.{self.num_blocks}.{kind}"] = self.state_map[
                f"model.1.sub./NB/.{kind}"
            ]
            del self.state_map[f"model.1.sub./NB/.{kind}"]

        old_state = OrderedDict()
        for old_key, new_keys in self.state_map.items():
            for new_key in new_keys:
                if r"\1" in old_key:
                    for k, v in state.items():
                        sub = re.sub(new_key, old_key, k)
                        if sub != k:
                            old_state[sub] = v
                else:
                    if new_key in state:
                        old_state[old_key] = state[new_key]

        # upconv layers
        max_upconv = 0
        for key in state.keys():
            match = re.match(r"(upconv|conv_up)(\d)\.(weight|bias)", key)
            if match is not None:
                _, key_num, key_type = match.groups()
                old_state[f"model.{int(key_num) * 3}.{key_type}"] = state[key]
                max_upconv = max(max_upconv, int(key_num) * 3)

        # final layers
        for key in state.keys():
            if key in ("HRconv.weight", "conv_hr.weight"):
                old_state[f"model.{max_upconv + 2}.weight"] = state[key]
            elif key in ("HRconv.bias", "conv_hr.bias"):
                old_state[f"model.{max_upconv + 2}.bias"] = state[key]
            elif key in ("conv_last.weight",):
                old_state[f"model.{max_upconv + 4}.weight"] = state[key]
            elif key in ("conv_last.bias",):
                old_state[f"model.{max_upconv + 4}.bias"] = state[key]

        # Sort by first numeric value of each layer
        def compare(item1, item2):
            parts1 = item1.split(".")
            parts2 = item2.split(".")
            int1 = int(parts1[1])
            int2 = int(parts2[1])
            return int1 - int2

        sorted_keys = sorted(old_state.keys(), key=functools.cmp_to_key(compare))

        # Rebuild the output dict in the right order
        out_dict = OrderedDict((k, old_state[k]) for k in sorted_keys)

        return out_dict

    def get_scale(self, min_part: int = 6) -> int:
        n = 0
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > min_part and parts[1] == "weight":
                    n += 1
        return 2**n

    def get_num_blocks(self) -> int:
        nbs = []
        state_keys = self.state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"] + (
            r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",
        )
        for state_key in state_keys:
            for k in self.state:
                m = re.search(state_key, k)
                if m:
                    nbs.append(int(m.group(1)))
            if nbs:
                break
        return max(*nbs) + 1

    def forward(self, x):
        if self.shuffle_factor:
            _, _, h, w = x.size()
            mod_pad_h = (
                self.shuffle_factor - h % self.shuffle_factor
            ) % self.shuffle_factor
            mod_pad_w = (
                self.shuffle_factor - w % self.shuffle_factor
            ) % self.shuffle_factor
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
            x = torch.pixel_unshuffle(x, downscale_factor=self.shuffle_factor)
            x = self.model(x)
            return x[:, :, : h * self.scale, : w * self.scale]
        return self.model(x)

```

```python
# archs\safmn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.state import get_seq_len
import math

# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# SE
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


# Channel MLP: Conv1*1 -> Conv1*1
class ChannelMLP(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.mlp(x)


# MBConv: Conv1*1 -> DW Conv3*3 -> [SE] -> Conv1*1
class MBConv(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mbconv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.mbconv(x)


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.ccm(x)


# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [
                nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)
                for _ in range(self.n_levels)
            ]
        )

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()


    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2**i, w // 2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode="nearest")
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.safm = SAFM(dim)
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


class SAFMN(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        dim = state_dict["to_feat.weight"].shape[0]
        n_blocks = get_seq_len(state_dict, "feats")

    # hidden_dim = int(dim * ffn_scale)
        hidden_dim = state_dict["feats.0.ccm.ccm.0.weight"].shape[0]
        ffn_scale = hidden_dim / dim

    # 3 * upscaling_factor**2
        upscaling_factor = int(math.sqrt(state_dict["to_img.0.weight"].shape[0] / 3))
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor),
        )
        self.input_channels = 3 
        self.name = "safmn"

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x

```

```python
# archs\span.py
from collections import OrderedDict
from typing import Literal
import math

import torch
import torch.nn.functional as F
from torch import nn as nn


def _make_pair(value):
    if isinstance(value, int):
        return (value, value)
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(f"activation layer [{act_type:s}] is not found")
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class Conv3XC(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            gain1=1,
            gain2=0,
            s=1,
            bias: Literal[True] = True,
            relu=False,
    ):
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False  # type: ignore
        self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()  # type: ignore
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat  # type: ignore

    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.act2 = activation("lrelu", neg_slope=0.1, inplace=True)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att


class SPAN(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(self, state_dict):
        super().__init__()

        upscale = max(1, int(math.sqrt(state_dict["upsampler.0.weight"].shape[0] / 3)))
        bias = "block_1.c1_r.sk.bias" in state_dict
        img_range = 255.
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.norm = True
        num_in_ch = state_dict["conv_1.sk.weight"].shape[1]
        feature_channels = state_dict["conv_1.sk.weight"].shape[0]
        num_out_ch = num_in_ch
        self.in_channels = num_in_ch
        self.out_channels = num_out_ch
        self.img_range = img_range
        norm = True
        if 'no_norm' in state_dict:
            norm = False
        else:
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.norm = norm
        self.input_channels = num_in_ch
        self.name = "span"
        self.conv_1 = Conv3XC(self.in_channels, feature_channels, gain1=2, s=1)
        self.block_1 = SPAB(feature_channels, bias=bias)
        self.block_2 = SPAB(feature_channels, bias=bias)
        self.block_3 = SPAB(feature_channels, bias=bias)
        self.block_4 = SPAB(feature_channels, bias=bias)
        self.block_5 = SPAB(feature_channels, bias=bias)
        self.block_6 = SPAB(feature_channels, bias=bias)

        self.conv_cat = conv_layer(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = pixelshuffle_block(
            feature_channels, self.out_channels, upscale_factor=upscale
        )

    def forward(self, x):
        if self.norm:
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        out_b1, _, _att1 = self.block_1(out_feature)
        out_b2, _, _att2 = self.block_2(out_b1)
        out_b3, _, _att3 = self.block_3(out_b2)

        out_b4, _, _att4 = self.block_4(out_b3)
        out_b5, _, _att5 = self.block_5(out_b4)
        out_b6, out_b5_2, _att6 = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        output = self.upsampler(out)

        return output

```

```python
# archs\srvgg.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.
    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(
        self,
        state_dict,
        act_type: str = "prelu",
    ):
        super(SRVGGNetCompact, self).__init__()
        self.model_arch = "SRVGG (RealESRGAN)"
        self.name = "Compact"
        self.sub_type = "SR"
        self.input_channels = 3 
        self.act_type = act_type

        self.state = state_dict

        if "params" in self.state:
            self.state = self.state["params"]

        self.weight_keys = [key for key in self.state.keys() if "weight" in key]
        self.highest_num = max(
            [int(key.split(".")[1]) for key in self.weight_keys if "body" in key]
        )

        self.in_nc = self.get_in_nc()
        self.num_feat = self.get_num_feats()
        self.num_conv = self.get_num_conv()
        self.out_nc = self.in_nc  # :(
        self.pixelshuffle_shape = None  # Defined in get_scale()
        self.scale = self.get_scale()

        self.supports_fp16 = True
        self.supports_bfp16 = True
        self.min_size_restriction = None

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(self.in_nc, self.num_feat, 3, 1, 1))
        # the first activation
        if act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=self.num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)  # type: ignore

        # the body structure
        for _ in range(self.num_conv):
            self.body.append(nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1))
            # activation
            if act_type == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_type == "prelu":
                activation = nn.PReLU(num_parameters=self.num_feat)
            elif act_type == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)  # type: ignore

        # the last conv
        self.body.append(nn.Conv2d(self.num_feat, self.pixelshuffle_shape, 3, 1, 1))  # type: ignore
        # upsample
        self.upsampler = nn.PixelShuffle(self.scale)

        self.load_state_dict(self.state, strict=False)

    def get_num_conv(self) -> int:
        return (self.highest_num - 2) // 2

    def get_num_feats(self) -> int:
        return self.state[self.weight_keys[0]].shape[0]

    def get_in_nc(self) -> int:
        return self.state[self.weight_keys[0]].shape[1]

    def get_scale(self) -> int:
        self.pixelshuffle_shape = self.state[f"body.{self.highest_num}.bias"].shape[0]
        # Assume out_nc is the same as in_nc
        # I cant think of a better way to do that
        self.out_nc = self.in_nc
        scale = math.sqrt(self.pixelshuffle_shape / self.out_nc)
        if scale - int(scale) > 0:
            print(
                "out_nc is probably different than in_nc, scale calculation might be"
                " wrong"
            )
        scale = int(scale)
        return scale

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        out += base
        return out

```

```python
# archs\swinir.py
# pylint: skip-file
# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# Originally from the timm package

from .utils.torch_internals import to_2tuple
from .utils.drop import DropPath
from .utils.trunc import trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(  # type: ignore
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)  # type: ignore
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, window_size={self.window_size},"
            f" num_heads={self.num_heads}"
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(
                x_windows, mask=self.calculate_mask(x_size).to(x.device)
            )

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution},"
            f" num_heads={self.num_heads}, window_size={self.window_size},"
            f" shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution},"
            f" depth={self.depth}"
        )

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()  # type: ignore
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
    ):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x, x_size):
        return (
            self.patch_embed(
                self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))
            )
            + x
        )

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],  # type: ignore
            img_size[1] // patch_size[1],  # type: ignore
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim  # type: ignore
        return flops


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],  # type: ignore
            img_size[1] // patch_size[1],  # type: ignore
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution  # type: ignore
        flops = H * W * self.num_feat * 3 * 9
        return flops


class SwinIR(nn.Module):
    r"""SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        state_dict,
        **kwargs,
    ):
        super(SwinIR, self).__init__()

        # Defaults
        img_size = 64
        patch_size = 1
        in_chans = 3
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        ape = False
        patch_norm = True
        use_checkpoint = False
        self.start_unshuffle = 1

        self.model_arch = "SwinIR"
        self.name = "SwinIR"
        self.sub_type = "SR"
        self.state = state_dict
        if "params_ema" in self.state:
            self.state = self.state["params_ema"]
        elif "params" in self.state:
            self.state = self.state["params"]

        state_keys = self.state.keys()

        if "conv_before_upsample.0.weight" in state_keys:
            if "conv_up1.weight" in state_keys:
                upsampler = "nearest+conv"
            else:
                upsampler = "pixelshuffle"
                supports_fp16 = False
        elif "upsample.0.weight" in state_keys:
            upsampler = "pixelshuffledirect"
        else:
            upsampler = ""

        num_feat = (
            self.state.get("conv_before_upsample.0.weight", None).shape[1]
            if self.state.get("conv_before_upsample.weight", None)
            else 64
        )

        if "conv_first.1.weight" in self.state:
            self.state["conv_first.weight"] = self.state.pop("conv_first.1.weight")
            self.state["conv_first.bias"] = self.state.pop("conv_first.1.bias")
            self.start_unshuffle = round(
                math.sqrt(self.state["conv_first.weight"].shape[1] // 3)
            )

        num_in_ch = self.state["conv_first.weight"].shape[1]
        in_chans = num_in_ch
        if "conv_last.weight" in state_keys:
            num_out_ch = self.state["conv_last.weight"].shape[0]
        else:
            num_out_ch = num_in_ch

        upscale = 1
        if upsampler == "nearest+conv":
            upsample_keys = [
                x for x in state_keys if "conv_up" in x and "bias" not in x
            ]

            for upsample_key in upsample_keys:
                upscale *= 2
        elif upsampler == "pixelshuffle":
            upsample_keys = [
                x
                for x in state_keys
                if "upsample" in x and "conv" not in x and "bias" not in x
            ]
            for upsample_key in upsample_keys:
                shape = self.state[upsample_key].shape[0]
                upscale *= math.sqrt(shape // num_feat)
            upscale = int(upscale)
        elif upsampler == "pixelshuffledirect":
            upscale = int(
                math.sqrt(self.state["upsample.0.bias"].shape[0] // num_out_ch)
            )

        max_layer_num = 0
        max_block_num = 0
        for key in state_keys:
            result = re.match(
                r"layers.(\d*).residual_group.blocks.(\d*).norm1.weight", key
            )
            if result:
                layer_num, block_num = result.groups()
                max_layer_num = max(max_layer_num, int(layer_num))
                max_block_num = max(max_block_num, int(block_num))

        depths = [max_block_num + 1 for _ in range(max_layer_num + 1)]

        if (
            "layers.0.residual_group.blocks.0.attn.relative_position_bias_table"
            in state_keys
        ):
            num_heads_num = self.state[
                "layers.0.residual_group.blocks.0.attn.relative_position_bias_table"
            ].shape[-1]
            num_heads = [num_heads_num for _ in range(max_layer_num + 1)]
        else:
            num_heads = depths

        embed_dim = self.state["conv_first.weight"].shape[0]

        mlp_ratio = float(
            self.state["layers.0.residual_group.blocks.0.mlp.fc1.bias"].shape[0]
            / embed_dim
        )

        # TODO: could actually count the layers, but this should do
        if "layers.0.conv.4.weight" in state_keys:
            resi_connection = "3conv"
        else:
            resi_connection = "1conv"

        window_size = int(
            math.sqrt(
                self.state[
                    "layers.0.residual_group.blocks.0.attn.relative_position_index"
                ].shape[0]
            )
        )

        if "layers.0.residual_group.blocks.1.attn_mask" in state_keys:
            img_size = int(
                math.sqrt(
                    self.state["layers.0.residual_group.blocks.1.attn_mask"].shape[0]
                )
                * window_size
            )

        # The JPEG models are the only ones with window-size 7, and they also use this range
        img_range = 255.0 if window_size == 7 else 1.0

        self.in_nc = num_in_ch // self.start_unshuffle**2
        self.out_nc = num_out_ch
        self.num_feat = num_feat
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depths = depths
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.scale = upscale
        self.upsampler = upsampler
        self.img_size = img_size
        self.img_range = img_range
        self.resi_connection = resi_connection
        self.input_channels = num_in_ch
        self.supports_fp16 = False  # Too much weirdness to support this at the moment
        self.supports_bfp16 = True
        self.min_size_restriction = 16

        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(  # type: ignore
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])  # type: ignore
                ],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale,
                embed_dim,
                num_out_ch,
                (patches_resolution[0], patches_resolution[1]),
            )
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            elif self.upscale == 8:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)
        self.load_state_dict(self.state, strict=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore  # type: ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore  # type: ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.start_unshuffle > 1:
            up = torch.nn.Upsample(scale_factor=self.start_unshuffle, mode="bicubic")
            x = up(x)
            x = torch.nn.functional.pixel_unshuffle(x, self.start_unshuffle)

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(
                    torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")  # type: ignore
                )
            )
            if self.upscale == 4:
                x = self.lrelu(
                    self.conv_up2(
                        torch.nn.functional.interpolate(  # type: ignore
                            x, scale_factor=2, mode="nearest"
                        )
                    )
                )
            elif self.upscale == 8:
                x = self.lrelu(
                    self.conv_up2(
                        torch.nn.functional.interpolate(
                            x, scale_factor=2, mode="nearest"
                        )
                    )
                )
                x = self.lrelu(
                    self.conv_up3(
                        torch.nn.functional.interpolate(
                            x, scale_factor=2, mode="nearest"
                        )
                    )
                )
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, : H * self.upscale, : W * self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()  # type: ignore
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()  # type: ignore
        return flops

```

```python
# archs\types.py
from typing import Union

from .span import SPAN
from .ditn import DITN
from .omnisr import OmniSR
from .rrdb import RRDBNet as ESRGAN
from .srvgg import SRVGGNetCompact as RealESRGANv2
from .dat import DAT
from .swinir import SwinIR
from .realcugan import cugan
from .safmn import SAFMN
from .rgt import RGT

PyTorchModel = Union[DITN, OmniSR, ESRGAN, RealESRGANv2, DAT, SwinIR, SPAN, cugan, SAFMN, RGT]

```

```python
# archs\__init__.py
from .span import SPAN
from .types import PyTorchModel
from .ditn import DITN
from .omnisr import OmniSR
from .rrdb import RRDBNet as ESRGAN
from .srvgg import SRVGGNetCompact as RealESRGANv2
from .dat import DAT
from .swinir import SwinIR
from .realcugan import cugan
from .safmn import SAFMN
from .rgt import RGT


def load_model(state_dict) -> PyTorchModel:
    unwrap_keys = ["state_dict", "params_ema", "params-ema", "params", "model", "net"]
    for key in unwrap_keys:
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break

    state_dict_keys = list(state_dict.keys())
    model: PyTorchModel | None = None
    try:
        cugan3x = state_dict["unet1.conv_bottom.weight"].shape[2]
    except:
        cugan3x = 0
    if "UFONE.0.ITLs.0.attn.temperature" in state_dict_keys:
        model = DITN(state_dict)
    elif "residual_layer.0.residual_layer.0.layer.0.fn.0.weight" in state_dict_keys:
        model = OmniSR(state_dict)
    elif "body.0.weight" in state_dict_keys and "body.1.weight" in state_dict_keys:
        model = RealESRGANv2(state_dict)
    elif "layers.0.residual_group.blocks.0.norm1.weight" in state_dict_keys:
        model = SwinIR(state_dict)
    elif "layers.0.blocks.2.attn.attn_mask_0" in state_dict_keys:
        if 'layers.0.blocks.0.gamma' in state_dict_keys:
            model = RGT(state_dict)
        else:
            model = DAT(state_dict)
    elif "block_1.c1_r.sk.weight" in state_dict_keys:
        model = SPAN(state_dict)
    elif 'conv_final.weight' in state_dict_keys or 'unet1.conv1.conv.0.weight' in state_dict_keys or cugan3x == 5:
        model = cugan(state_dict)
    elif 'to_feat.weight' in state_dict_keys:
        model = SAFMN(state_dict)
    else:
        try:
            model = ESRGAN(state_dict)
        except:
            # pylint: disable=raise-missing-from
            print(state_dict_keys)
            raise Exception("UNSUPPORTED_MODEL")

    model.load_state_dict(state_dict, strict=False)

    return model

```

```python
# archs\utils\block.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn

####################
# Basic blocks
####################


def act(act_type: str, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )
    return layer


def norm(norm_type: str, nc: int):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(
            "normalization layer [{:s}] is not found".format(norm_type)
        )
    return layer


def pad(pad_type: str, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(
            "padding layer [{:s}] is not implemented".format(pad_type)
        )
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = "Identity .. \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = "Identity + \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlockSPSR(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlockSPSR, self).__init__()
        self.sub = submodule

    def forward(self, x):
        return x, self.sub

    def __repr__(self):
        tmpstr = "Identity + \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


ConvMode = Literal["CNA", "NAC", "CNAC"]


# 2x2x2 Conv Block
def conv_block_2c2(
    in_nc,
    out_nc,
    act_type="relu",
):
    return sequential(
        nn.Conv2d(in_nc, out_nc, kernel_size=2, padding=1),
        nn.Conv2d(out_nc, out_nc, kernel_size=2, padding=0),
        act(act_type) if act_type else None,
    )


def conv_block(
    in_nc: int,
    out_nc: int,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type: str | None = None,
    act_type: str | None = "relu",
    mode: ConvMode = "CNA",
    c2x2=False,
):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """

    if c2x2:
        return conv_block_2c2(in_nc, out_nc, act_type=act_type)

    assert mode in ("CNA", "NAC", "CNAC"), "Wrong conv mode [{:s}]".format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = act(act_type) if act_type else None
    if mode in ("CNA", "CNAC"):
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == "NAC":
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)
    else:
        assert False, f"Invalid conv mode {mode}"


####################
# Useful blocks
####################


class ResNetBlock(nn.Module):
    """
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    """

    def __init__(
        self,
        in_nc,
        mid_nc,
        out_nc,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        pad_type="zero",
        norm_type=None,
        act_type="relu",
        mode: ConvMode = "CNA",
        res_scale=1,
    ):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(
            in_nc,
            mid_nc,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
        )
        if mode == "CNA":
            act_type = None
        if mode == "CNAC":  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(
            mid_nc,
            out_nc,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
        )
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(
        self,
        nf,
        kernel_size=3,
        gc=32,
        stride=1,
        bias: bool = True,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode: ConvMode = "CNA",
        _convtype="Conv2D",
        _spectral_norm=False,
        plus=False,
        c2x2=False,
    ):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )
        self.RDB2 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )
        self.RDB3 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    Modified options that can be used:
        - "Partial Convolution based Padding" arXiv:1811.11718
        - "Spectral normalization" arXiv:1802.05957
        - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C.
            {Rakotonirina} and A. {Rasoanaivo}

    Args:
        nf (int): Channel number of intermediate features (num_feat).
        gc (int): Channels for each growth (num_grow_ch: growth channel,
            i.e. intermediate channels).
        convtype (str): the type of convolution to use. Default: 'Conv2D'
        gaussian_noise (bool): enable the ESRGAN+ gaussian noise (no new
            trainable parameters)
        plus (bool): enable the additional residual paths from ESRGAN+
            (adds trainable parameters)
    """

    def __init__(
        self,
        nf=64,
        kernel_size=3,
        gc=32,
        stride=1,
        bias: bool = True,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode: ConvMode = "CNA",
        plus=False,
        c2x2=False,
    ):
        super(ResidualDenseBlock_5C, self).__init__()

        ## +
        self.conv1x1 = conv1x1(nf, gc) if plus else None
        ## +

        self.conv1 = conv_block(
            nf,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv2 = conv_block(
            nf + gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv3 = conv_block(
            nf + 2 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv4 = conv_block(
            nf + 3 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        if mode == "CNA":
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(
            nf + 4 * gc,
            nf,
            3,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=last_act,
            mode=mode,
            c2x2=c2x2,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1:
            # pylint: disable=not-callable
            x2 = x2 + self.conv1x1(x)  # +
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1:
            x4 = x4 + x2  # +
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


####################
# Upsampler
####################


def pixelshuffle_block(
    in_nc: int,
    out_nc: int,
    upscale_factor=2,
    kernel_size=3,
    stride=1,
    bias=True,
    pad_type="zero",
    norm_type: str | None = None,
    act_type="relu",
):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv = conv_block(
        in_nc,
        out_nc * (upscale_factor**2),
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=None,
        act_type=None,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_block(
    in_nc: int,
    out_nc: int,
    upscale_factor=2,
    kernel_size=3,
    stride=1,
    bias=True,
    pad_type="zero",
    norm_type: str | None = None,
    act_type="relu",
    mode="nearest",
    c2x2=False,
):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(
        in_nc,
        out_nc,
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=norm_type,
        act_type=act_type,
        c2x2=c2x2,
    )
    return sequential(upsample, conv)

```

```python
# archs\utils\drop.py
import torch
from torch import nn
import math
import warnings
import collections.abc
from itertools import repeat


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

```

```python
# archs\utils\state.py
import math
from typing import Any


def get_first_seq_index(state: dict, key_pattern: str) -> int:
    """
    Returns the maximum index `i` such that `key_pattern.format(str(i))` is in `state`.

    If no such key is in state, then `-1` is returned.

    Example:
        get_first_seq_index(state, "body.{}.weight") -> -1
        get_first_seq_index(state, "body.{}.weight") -> 3
    """
    for i in range(100):
        if key_pattern.format(str(i)) in state:
            return i
    return -1


def get_seq_len(state: dict[str, Any], seq_key: str) -> int:
    """
    Returns the length of a sequence in the state dict.

    The length is detected by finding the maximum index `i` such that
    `{seq_key}.{i}.{suffix}` is in `state` for some suffix.

    Example:
        get_seq_len(state, "body") -> 5
    """
    prefix = seq_key + "."

    keys: set[int] = set()
    for k in state.keys():
        if k.startswith(prefix):
            index = k[len(prefix) :].split(".", maxsplit=1)[0]
            keys.add(int(index))

    if len(keys) == 0:
        return 0
    return max(keys) + 1


def get_scale_and_output_channels(x: int, input_channels: int) -> tuple[int, int]:
    """
    Returns a scale and number of output channels such that `scale**2 * out_nc = x`.

    This is commonly used for pixelshuffel layers.
    """
    # Unfortunately, we do not have enough information to determine both the scale and
    # number output channels correctly *in general*. However, we can make some
    # assumptions to make it good enough.
    #
    # What we know:
    # - x = scale * scale * output_channels
    # - output_channels is likely equal to input_channels
    # - output_channels and input_channels is likely 1, 3, or 4
    # - scale is likely 1, 2, 4, or 8

    def is_square(n: int) -> bool:
        return math.sqrt(n) == int(math.sqrt(n))

    # just try out a few candidates and see which ones fulfill the requirements
    candidates = [input_channels, 3, 4, 1]
    for c in candidates:
        if x % c == 0 and is_square(x // c):
            return int(math.sqrt(x // c)), c

    raise AssertionError(
        f"Expected output channels to be either 1, 3, or 4."
        f" Could not find a pair (scale, out_nc) such that `scale**2 * out_nc = {x}`"
    )

```

```python
# archs\utils\torch_internals.py
import collections.abc
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

```

```python
# archs\utils\trunc.py
import torch
from torch import nn
import math
import warnings


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: torch.Tensor, mean=0.0, std=1.0, a=-2.0, b=2.0
) -> torch.Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

```

```python
# archs\utils\__init__.py

```

```python
# ext\image_adjustments.py
import numpy as np
import cv2
import os


class ImageAdjustments:
    def __init__(self, color_level):
        """
        Initializes the ImageAdjustments class with levels settings provided as a dictionary.
        """
        self.shadows = color_level["shadows"] / 255.0
        self.midtones = color_level["midtones"]
        self.highlights = color_level["highlights"] / 255.0
        self.output_shadows = color_level.get("output_shadows", 0) / 255.0
        self.output_highlights = color_level.get("output_highlights", 255) / 255.0

    def color_level(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the level adjustments to an image.
        
        Parameters:
        - image: The input image as a numpy array.
        
        Returns:
        - The adjusted image as a numpy array.
        """
        # Clip the shadows and highlights
        clipped = np.clip((image - self.shadows) / (self.highlights - self.shadows), 0, 1)
        # Apply gamma correction to midtones
        corrected = np.power(clipped, 1 / self.midtones)
        # Scale to the output range
        scaled = corrected * (self.output_highlights - self.output_shadows) + self.output_shadows
        # Ensure the final values are within byte range
        final_image = np.clip(scaled * 255, 0, 255).astype(np.uint8)
        
        return final_image
    
    def batch_mode(self, input_path, output_path):
        """
        Applies the level adjustments to all images in the input directory and saves
        them to the output directory.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(input_path, filename)
                img = cv2.imread(img_path)

                if img is not None:
                    result_img = self.color_level(img)
                    cv2.imwrite(os.path.join(output_path, filename), result_img)
                else:
                    print(f"Skipping file {filename}, not a valid image.")
```

```python
# ext\resize.py
from chainner_ext import resize, ResizeFilter
import numpy as np
import os
import cv2

class Resize:
    def __init__(self, setting):
        self.size = setting["size"]
        self.width = setting["width"]
        self.interpolation = setting["interpolation"]
        self.percent = setting["percent"] / 100
        self.spread = setting["spread"]
        self.spread_size = setting["spread_size"]
        self.interpolation_map = self._create_interpolation_map(setting["interpolation"])

    def _create_interpolation_map(self, interpolation):
        if interpolation == "auto":
            # Logic to choose interpolation based on upsizing or downsizing
            return {
                'upscale': ResizeFilter.Cubic,
                'downscale': ResizeFilter.Lanczos
            }
        else:
            return {
                interpolation: ResizeFilter[interpolation]
            }

    def _determine_interpolation(self, original_size, new_size):
        if new_size > original_size:
            return self.interpolation_map.get('upscale', ResizeFilter.Cubic)
        else:
            return self.interpolation_map.get('downscale', ResizeFilter.Lanczos)
        
        
    def Factor(self, img):
        """تطبيق النسبة المئوية لتغيير حجم الصورة."""
        height, width = img.shape[:2]
        new_width = int(width * self.percent / 100)
        new_height = int(height * self.percent / 100)
        return (new_width, new_height)
    
    def resize_to_side(self, img):
        """تغيير حجم الصورة إلى جانب محدد باستخدام spread."""
        height, width = img.shape[:2]
        if self.width:
            new_height = self.spread_size
            new_width = int(width / height * self.spread_size)
        else:
            new_width = self.spread_size
            new_height = int(height / width * self.spread_size)
        return (new_width, new_height)
    
    def batch_mode(self, input_path, output_path):
        """تغيير حجم جميع الصور في مجلد وحفظ النتائج في مجلد آخر باستخدام OpenCV."""
        # التأكد من وجود المجلد الناتج
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # قراءة جميع الملفات في المجلد المدخل
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(input_path, filename)
                img = cv2.imread(img_path)  # قراءة الصورة باستخدام OpenCV
                
                if img is not None:
                    # تطبيق تغيير الحجم
                    resized_img = self.run(img)
                    
                    # حفظ الصورة المعدلة باستخدام OpenCV
                    cv2.imwrite(os.path.join(output_path, filename), resized_img)
                else:
                    print(f"Skipping file {filename}, not a valid image or cannot be read.")
        

    def run(self, img):
        original_height, original_width = img.shape[:2]
        if self.spread and ((self.width and original_width > self.spread_size) or (not self.width and original_height > self.spread_size)):
            new_size = self.resize_to_side(img)
        else:
            new_size = self.Factor(img)
        interpolation_method = self._determine_interpolation(original_width if self.width else original_height, new_size[0] if self.width else new_size[1])
        resized_img = resize(img.astype(np.float32), new_size, interpolation_method, gamma_correction=False)
        return resized_img

```

```python
# ext\__init__.py

```

```python
# utils\cuda.py
import torch


def safe_cuda_cache_empty():
    """
    Empties the CUDA cache if CUDA is available. Hopefully without causing any errors.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

```

```python
# utils\file.py
import os
from typing import Tuple


def get_opencv_formats():
    return [
        # Bitmaps
        ".bmp",
        ".dib",
        # JPEG
        ".jpg",
        ".jpeg",
        ".jpe",
        ".jp2",
        # PNG, WebP, Tiff
        ".png",
        ".webp",
        ".tif",
        ".tiff",
        # Portable image format
        ".pbm",
        ".pgm",
        ".ppm",
        ".pxm",
        ".pnm",
        # Sun Rasters
        ".sr",
        ".ras",
        # OpenEXR
        ".exr",
        # Radiance HDR
        ".hdr",
        ".pic",
    ]


def split_file_path(path: str) -> Tuple[str, str, str]:
    """
    Returns the base directory, file name, and extension of the given file path.
    """
    base, ext = os.path.splitext(path)
    dirname, basename = os.path.split(base)
    return dirname, basename, ext


def get_ext(path: str) -> str:
    return split_file_path(path)[2].lower()

```

```python
# utils\image.py
import numpy as np
import cv2
import os
import random
import string
import torch
import math
from torchvision.utils import make_grid
from typing import Tuple, List
from .file import split_file_path, get_ext, get_opencv_formats

MAX_VALUES_BY_DTYPE = {
    np.dtype("int8").name: 127,
    np.dtype("uint8").name: 255,
    np.dtype("int16").name: 32767,
    np.dtype("uint16").name: 65535,
    np.dtype("int32").name: 2147483647,
    np.dtype("uint32").name: 4294967295,
    np.dtype("int64").name: 9223372036854775807,
    np.dtype("uint64").name: 18446744073709551615,
    np.dtype("float32").name: 1.0,
    np.dtype("float64").name: 1.0,
}


def as_3d(img: np.ndarray) -> np.ndarray:
    """Given a grayscale image, this returns an image with 3 dimensions (image.ndim == 3)."""
    if img.ndim == 2:
        return np.expand_dims(img.copy(), axis=2)
    return img


def as_4d(img: np.ndarray) -> np.ndarray:
    """Given a grayscale image, this returns an image with 4 dimensions (image.ndim == 4)."""
    if img.ndim == 3:
        return np.expand_dims(img.copy(), axis=2)
    return img


def get_h_w_c(image: np.ndarray) -> Tuple[int, int, int]:
    """Returns the height, width, and number of channels."""
    h, w = image.shape[:2]
    c = 1 if image.ndim == 2 else image.shape[2]
    return h, w, c


def read_cv(file_path: str, flag='color', float32=True) -> np.ndarray | None:
    """Read an image from bytes.

    Args:
        file_path: Image file path.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """

    if get_ext(file_path) not in get_opencv_formats():
        raise RuntimeError(f'Unsupported image format for file "{file_path}"')

    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }

    try:
        img = cv2.imread(file_path, imread_flags[flag])
        if float32:
            img = img.astype(np.float32) / 255.
        return img
    except Exception as e:
        raise RuntimeError(
            f'Error reading image image from path "{file_path}". Image may be corrupt'
        ) from e


def cv_save_image(path: str, img: np.ndarray, params: List[int]):
    """
    A light wrapper around `cv2.imwrite` to support non-ASCII paths.
    """

    # Write image with opencv if path is ascii, since imwrite doesn't support unicode
    # This saves us from having to keep the image buffer in memory, if possible
    if path.isascii():
        cv2.imwrite(path, img, params)
    else:
        dirname, _, extension = split_file_path(path)
        try:
            temp_filename = f'temp-{"".join(random.choices(string.ascii_letters, k=16))}.{extension}'
            full_temp_path = os.path.join(dirname, temp_filename)
            cv2.imwrite(full_temp_path, img, params)
            os.rename(full_temp_path, path)
        except:
            _, buf_img = cv2.imencode(f".{extension}", img, params)
            with open(path, "wb") as outf:
                outf.write(buf_img)  # type: ignore


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.dtype == "float64":
            img = img.astype("float32")

        if len(img.shape) > 2 and img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(img.shape) == 2:
            img = torch.from_numpy(img[None, ...])
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1))

        if float32:
            img = img.float()

        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (
            torch.is_tensor(tensor)
            or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))
    ):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False
            ).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(
                "Only support 4D, 3D or 2D tensor. But received with dimension:"
                f" {n_dim}"
            )
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

```

```python
# utils\tile.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from utils.image import get_h_w_c

Size = Tuple[int, int]


class Split:
    pass


@dataclass(frozen=True)
class Padding:
    top: int
    right: int
    bottom: int
    left: int

    @staticmethod
    def all(value: int) -> "Padding":
        return Padding(value, value, value, value)

    @staticmethod
    def to(value: Padding | int) -> Padding:
        if isinstance(value, int):
            return Padding.all(value)
        return value

    @property
    def horizontal(self) -> int:
        return self.left + self.right

    @property
    def vertical(self) -> int:
        return self.top + self.bottom

    @property
    def empty(self) -> bool:
        return self.top == 0 and self.right == 0 and self.bottom == 0 and self.left == 0

    def scale(self, factor: int) -> Padding:
        return Padding(
            self.top * factor,
            self.right * factor,
            self.bottom * factor,
            self.left * factor,
        )

    def min(self, other: Padding | int) -> Padding:
        other = Padding.to(other)
        return Padding(
            min(self.top, other.top),
            min(self.right, other.right),
            min(self.bottom, other.bottom),
            min(self.left, other.left),
        )

    def remove_from(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = get_h_w_c(image)

        return image[
            self.top : (h - self.bottom),
            self.left : (w - self.right),
            ...,
        ]


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    @property
    def size(self) -> Size:
        return self.width, self.height

    def scale(self, factor: int) -> Region:
        return Region(
            self.x * factor,
            self.y * factor,
            self.width * factor,
            self.height * factor,
        )

    def intersect(self, other: Region) -> Region:
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        width = min(self.x + self.width, other.x + other.width) - x
        height = min(self.y + self.height, other.y + other.height) - y
        return Region(x, y, width, height)

    def add_padding(self, pad: Padding) -> Region:
        return Region(
            x=self.x - pad.left,
            y=self.y - pad.top,
            width=self.width + pad.horizontal,
            height=self.height + pad.vertical,
        )

    def remove_padding(self, pad: Padding) -> Region:
        return self.add_padding(pad.scale(-1))

    def child_padding(self, child: Region) -> Padding:
        """
        Returns the padding `p` such that `child.add_padding(p) == self`.
        """
        left = child.x - self.x
        top = child.y - self.y
        right = self.width - child.width - left
        bottom = self.height - child.height - top
        return Padding(top, right, bottom, left)

    def read_from(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = get_h_w_c(image)
        if (w, h) == self.size:
            return image

        return image[
            self.y : (self.y + self.height),
            self.x : (self.x + self.width),
            ...,
        ]

    def write_into(self, lhs: np.ndarray, rhs: np.ndarray):
        h, w, c = get_h_w_c(rhs)
        assert (w, h) == self.size
        assert c == get_h_w_c(lhs)[2]

        if c == 1:
            if lhs.ndim == 2 and rhs.ndim == 3:
                rhs = rhs[:, :, 0]
            if lhs.ndim == 3 and rhs.ndim == 2:
                rhs = np.expand_dims(rhs, axis=2)

        lhs[
            self.y : (self.y + self.height),
            self.x : (self.x + self.width),
            ...,
        ] = rhs


def split_tile_size(tile_size: Size) -> Size:
    w, h = tile_size
    assert w >= 16 and h >= 16
    return max(16, w // 2), max(16, h // 2)


def auto_split(
    img: np.ndarray,
    tile_max_size,
    upscale,
    overlap: int = 16,
) -> np.ndarray:
    h, w, c = get_h_w_c(img)

    img_region = Region(0, 0, w, h)

    max_tile_size = (tile_max_size, tile_max_size)
    #print(f"Auto split image ({w}x{h}px @ {c}) with initial tile size {max_tile_size}.")

    if w <= max_tile_size[0] and h <= max_tile_size[1]:
        upscale_result = upscale(img)
        if not isinstance(upscale_result, Split):
            return upscale_result

        max_tile_size = split_tile_size(max_tile_size)

        print(
            "Unable to upscale the whole image at once. Reduced tile size to"
            f" {max_tile_size}."
        )

    start_x = 0
    start_y = 0
    result: Optional[np.ndarray] = None
    scale: int = 0

    restart = True
    while restart:
        restart = False

        tile_count_x = math.ceil(w / max_tile_size[0])
        tile_count_y = math.ceil(h / max_tile_size[1])
        tile_size_x = math.ceil(w / tile_count_x)
        tile_size_y = math.ceil(h / tile_count_y)

        for y in range(0, tile_count_y):
            if restart:
                break
            if y < start_y:
                continue

            for x in range(0, tile_count_x):
                if y == start_y and x < start_x:
                    continue

                tile = Region(
                    x * tile_size_x, y * tile_size_y, tile_size_x, tile_size_y
                ).intersect(img_region)
                pad = img_region.child_padding(tile).min(overlap)
                padded_tile = tile.add_padding(pad)

                upscale_result = upscale(padded_tile.read_from(img))

                if isinstance(upscale_result, Split):
                    max_tile_size = split_tile_size(max_tile_size)

                    new_tile_count_x = math.ceil(w / max_tile_size[0])
                    new_tile_count_y = math.ceil(h / max_tile_size[1])
                    new_tile_size_x = math.ceil(w / new_tile_count_x)
                    new_tile_size_y = math.ceil(h / new_tile_count_y)
                    start_x = (x * tile_size_x) // new_tile_size_x
                    start_y = (y * tile_size_x) // new_tile_size_y

                    print(
                        f"Split occurred. New tile size is {max_tile_size}. Starting at"
                        f" {start_x},{start_y}."
                    )

                    restart = True
                    break

                up_h, up_w, _ = get_h_w_c(upscale_result)
                current_scale = up_h // padded_tile.height
                assert current_scale > 0
                assert padded_tile.height * current_scale == up_h
                assert padded_tile.width * current_scale == up_w

                if result is None:
                    scale = current_scale
                    result = np.zeros((h * scale, w * scale, c), dtype=np.float32)

                assert current_scale == scale

                upscale_result = pad.scale(scale).remove_from(upscale_result)

                tile.scale(scale).write_into(result, upscale_result)

    assert result is not None
    return result

```

```python
# utils\unpickler.py
# Safe unpickler to prevent arbitrary code execution
# https://github.com/chaiNNer-org/spandrel/blob/ff7f11467db20a4e6ccd25280e233c1295857f5c/src/spandrel/__helpers/unpickler.py
import pickle
from types import SimpleNamespace

safe_list = {
    ("collections", "OrderedDict"),
    ("typing", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch", "BFloat16Storage"),
    ("torch", "FloatStorage"),
    ("torch", "HalfStorage"),
    ("torch", "IntStorage"),
    ("torch", "LongStorage"),
    ("torch", "DoubleStorage"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # Only allow required classes to load state dict
        if (module, name) not in safe_list:
            raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")
        return super().find_class(module, name)


RestrictedUnpickle = SimpleNamespace(
    Unpickler=RestrictedUnpickler,
    __name__="pickle",
    load=lambda *args, **kwargs: RestrictedUnpickler(*args, **kwargs).load(),
)
```

```python
# utils\upscaler.py
import os
import numpy as np
import torch
from archs import load_model
from utils.cuda import safe_cuda_cache_empty
from utils.image import cv_save_image, img2tensor, tensor2img, read_cv
from utils.tile import auto_split
from tqdm import tqdm
from utils.unpickler import RestrictedUnpickle
from moviepy.editor import VideoFileClip


class Upscaler:
    def __init__(self, model_path, input_folder, output_folder, tile_size=256, form="png"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(
            model_path, map_location="cpu", pickle_module=RestrictedUnpickle
        )

        model = load_model(state_dict)
        model.eval()
        model = model.to(device)

        self.model = model
        self.device = device
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.tile_max_size = tile_size
        self.format_image = form
        if self.model.input_channels == 1:
            self.channels = "grayscale"
        else:
            self.channels = "color"

    def __upscale(self, img: np.ndarray) -> np.ndarray:
        tensor = img2tensor(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            tensor = self.model(tensor)

        return tensor2img(tensor)

    def run(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        list_files = [
            file
            for file in os.listdir(self.input_folder)
            if os.path.isfile(os.path.join(self.input_folder, file))
        ]
        for filename in tqdm(list_files, desc=self.model.name, leave=True):
            input_image_path = os.path.join(self.input_folder, filename)
            try:
                img = read_cv(input_image_path, self.channels)
                if img is None:
                    raise RuntimeError(f"Unsupported image type: {filename}")

                result = auto_split(img, self.tile_max_size, self.__upscale)
                output_image_path = os.path.join(self.output_folder, "".join(filename.split(".")[:-1]))
                output_image_path_format = f"{ output_image_path }.{self.format_image}"
                cv_save_image(output_image_path_format, result, [])

            except RuntimeError as e:
                print(f"[FAILED] {filename} : {e}")

        safe_cuda_cache_empty()


class UpscalerVideo:
    def __init__(self, model_path, input_folder, output_folder, tile_size=256, form_video="mp4", codec_video="libx264",
                 codec_audio="aac"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(
            model_path, map_location="cpu", pickle_module=RestrictedUnpickle
        )

        model = load_model(state_dict)
        model.eval()
        model = model.to(device)

        self.model = model
        self.device = device
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.tile_max_size = tile_size
        self.format_video = form_video
        self.codec_video = codec_video
        self.codec_audio = codec_audio
        if self.model.input_channels == 1:
            self.channels = "grayscale"
        else:
            self.channels = "color"

    def __upscale(self, img: np.ndarray) -> np.ndarray:
        tensor = img2tensor(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            tensor = self.model(tensor)

        return tensor2img(tensor)

    def process_frame(self, frame):

        frame_np = np.array(frame) / 255
        return auto_split(frame_np, self.tile_max_size, self.__upscale)

    def run(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        list_files = [
            file
            for file in os.listdir(self.input_folder)
            if os.path.isfile(os.path.join(self.input_folder, file))
        ]
        for filename in list_files:
            input_video_path = os.path.join(self.input_folder, filename)
            try:
                video_clip = VideoFileClip(input_video_path)
                processed_clip = video_clip.fl_image(self.process_frame)
                output_video_path = os.path.join(self.output_folder, "".join(filename.split(".")[:-1]))
                output_video_path_format = f"{output_video_path}.{self.format_video}"

                processed_clip.write_videofile(output_video_path_format, codec=self.codec_video,
                                               audio_codec=self.codec_audio)
            except RuntimeError as e:
                return print(f"[FAILED] {e}")
        safe_cuda_cache_empty()


```

```python
# utils\__init__.py

```
