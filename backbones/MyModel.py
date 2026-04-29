import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .WaveletOperator import LearnableWavelet1D

def iq_to_complex(x_iq: torch.Tensor) -> torch.Tensor:
    if x_iq.dim() != 3 or x_iq.size(1) != 2:
        raise ValueError("Expected input shape [B, 2, L] for IQ.")
    return torch.complex(x_iq[:, 0:1, :], x_iq[:, 1:2, :])


def complex_to_iq(z: torch.Tensor) -> torch.Tensor:
    if z.dim() != 3 or z.size(1) != 1:
        raise ValueError("Expected input shape [B, 1, L] for complex.")
    return torch.cat([z.real, z.imag], dim=1).to(dtype=z.real.dtype)

class WaveletDecompReconstruct_DWT(nn.Module):
    """
    Fixed levels learnable wavelet decomposition and reconstruction.
    Input: complex [B, 1, L]
    Output: reconstructed complex [B, 1, L]
    """
    def __init__(
        self,
        seq_len: int,
        levels: int = 3,
        kernel_size: int = 2,
        eps: float = 1e-8,
        wavelet_init: str = "haar",
        wavelet_learnable: bool = True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.levels = int(levels)
        self.kernel_size = int(kernel_size)
        self.eps = float(eps)

        if self.levels < 1:
            raise ValueError("levels must be >= 1")

        self.waves = nn.ModuleList([
            LearnableWavelet1D(
                channels=1,
                kernel_size=self.kernel_size,
                init=wavelet_init,
                learnable=wavelet_learnable
            )
            for _ in range(self.levels)
        ])

    @staticmethod
    def _pad_to_even(x: torch.Tensor):
        L = x.size(-1)
        if (L % 2) == 0:
            return x, 0
        return F.pad(x, (0, 1)), 1

    def forward(self, x: torch.Tensor):
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError("Expected x shape [B, 1, L] complex")

        _, _, L = x.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {L}")

        x_mean = x.mean(dim=-1, keepdim=True)
        x0 = x - x_mean

        x_var = (
            x0.real.var(dim=-1, keepdim=True, unbiased=False)
            + x0.imag.var(dim=-1, keepdim=True, unbiased=False)
        )
        x_var = 0.5 * x_var + 1e-5
        x0 = x0 / torch.sqrt(x_var)

        x0, pad_right = self._pad_to_even(x0)
        L0 = x0.size(-1)

        low = x0
        highs = []
        shapes = []
        for j in range(self.levels):
            low, high = self.waves[j].analysis(low)
            highs.append(high)
            shapes.append((low.size(-1), high.size(-1)))

        for j in reversed(range(self.levels)):
            high_cur = highs[j]
            out_len = high_cur.size(-1) * 2
            low = self.waves[j].synthesis(low, high_cur, out_len=out_len)

        x_rec0 = low
        if x_rec0.size(-1) > L0:
            x_rec0 = x_rec0[..., :L0]
        if pad_right == 1:
            x_rec0 = x_rec0[..., : L0 - 1]

        x_rec = x_rec0 * torch.sqrt(x_var) + x_mean

        info = {
            "input_len_after_pad": int(L0),
            "levels": int(self.levels),
            "per_level_low_high_len": shapes,
            "pad_right": int(pad_right),
        }
        return x_rec, info


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size, in_chans, embed_dim, stride=None):
        super().__init__()
        if stride is None:
            stride = patch_size // 2
        if seq_len < patch_size:
            raise ValueError(f"seq_len({seq_len}) must be >= patch_size({patch_size}).")
        if stride <= 0:
            raise ValueError(f"stride({stride}) must be > 0.")
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x)
        x_out = x_out.transpose(1, 2)
        return x_out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dr):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dr),
        )

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dr):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, dr)

    def forward(self, x):
        x = x + self.ffn(self.norm(x))
        return x

class MyModel(nn.Module):
    def __init__(
        self,
        seq_len=256,
        in_chans=2,
        patch_size=64,
        embed_dim=128,
        num_classes=6,
        mlp_ratio=2.0,
        dr=0.5,
        stride=None,
        use_wavelet: bool = True,
        wavelet_levels: int = 3,
        wavelet_learnable: bool = True,
        wavelet_init: str = "randn",
        wavelet_kernel_size: int = 8,
    ):
        super().__init__()

        self.use_wavelet = bool(use_wavelet)
        if self.use_wavelet:
            self.wavelet = WaveletDecompReconstruct_DWT(
                seq_len=int(seq_len),
                levels=int(wavelet_levels),
                kernel_size=int(wavelet_kernel_size),
                wavelet_init=str(wavelet_init),
                wavelet_learnable=bool(wavelet_learnable),
            )

        self.patch_embedding = PatchEmbed(
            int(seq_len),
            int(patch_size),
            int(in_chans),
            int(embed_dim),
            stride=stride,
        )

        self.bottleneck = MLP(int(embed_dim), mlp_ratio=mlp_ratio, dr=dr)
        self.cls_head = nn.Linear(int(embed_dim), int(num_classes))

    def forward_features(self, x):
        if self.use_wavelet:
            z = iq_to_complex(x)
            z_rec, _ = self.wavelet(z)
            x_in = complex_to_iq(z_rec)
        else:
            x_in = x

        x_patch = self.patch_embedding(x_in)
        x_neck = self.bottleneck(x_patch)
        x_mean = x_neck.mean(dim=1)
        return x_mean

    def forward(self, x):
        x_mean = self.forward_features(x)
        logit = self.cls_head(x_mean)
        return logit, x, x_mean