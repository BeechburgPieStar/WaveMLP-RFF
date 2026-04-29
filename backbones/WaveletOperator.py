import math
import torch
import torch.nn as nn
from .ComplexOperator import ComplexConv1d, ComplexConvTranspose1d

def _place_diag_filter(weight: torch.Tensor, filt_1d: torch.Tensor):
    weight.zero_()
    C = weight.size(0)
    for c in range(C):
        weight[c, c, :] = filt_1d

def _pad_or_crop_to_kernel(f: torch.Tensor, kernel_size: int) -> torch.Tensor:
    K = int(kernel_size)
    L = int(f.numel())
    if L == K:
        return f
    if L < K:
        out = torch.zeros((K,), dtype=f.dtype, device=f.device)
        start = (K - L) // 2
        out[start:start + L] = f
        return out
    start = (L - K) // 2
    return f[start:start + K]


def _wavelet_filters_1d(init: str, kernel_size: int, dtype, device):
    init = str(init).lower()

    if init == "randn":
        raise RuntimeError("randn is handled outside this function")

    if init == "haar":
        base = torch.tensor([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=dtype, device=device)
        h0 = base
        h1 = torch.tensor([1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)], dtype=dtype, device=device)
        return _pad_or_crop_to_kernel(h0, kernel_size), _pad_or_crop_to_kernel(h1, kernel_size)

    if init == "db2":
        sqrt3 = math.sqrt(3.0)
        denom = 4.0 * math.sqrt(2.0)
        h0 = torch.tensor(
            [(1 + sqrt3) / denom, (3 + sqrt3) / denom, (3 - sqrt3) / denom, (1 - sqrt3) / denom],
            dtype=dtype, device=device
        )
        h1 = torch.tensor(
            [(1 - sqrt3) / denom, -(3 - sqrt3) / denom, (3 + sqrt3) / denom, -(1 + sqrt3) / denom],
            dtype=dtype, device=device
        )
        if kernel_size < 4:
            raise ValueError("db2 requires kernel_size >= 4")
        return _pad_or_crop_to_kernel(h0, kernel_size), _pad_or_crop_to_kernel(h1, kernel_size)

    if init == "sym2":
        h0 = torch.tensor(
            [-0.12940952255126034, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416],
            dtype=dtype, device=device
        )
        h1 = torch.tensor(
            [-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126034],
            dtype=dtype, device=device
        )
        if kernel_size < 4:
            raise ValueError("sym2 requires kernel_size >= 4")
        return _pad_or_crop_to_kernel(h0, kernel_size), _pad_or_crop_to_kernel(h1, kernel_size)

    if init == "db4":
        h0 = torch.tensor(
            [
                -0.010597401784997278,
                0.032883011666982945,
                0.030841381835986965,
                -0.18703481171888114,
                -0.02798376941698385,
                0.6308807679298587,
                0.7148465705529154,
                0.2303778133088965
            ],
            dtype=dtype, device=device
        )
        h1 = torch.tensor(
            [
                -0.2303778133088965,
                0.7148465705529154,
                -0.6308807679298587,
                -0.02798376941698385,
                0.18703481171888114,
                0.030841381835986965,
                -0.032883011666982945,
                -0.010597401784997278
            ],
            dtype=dtype, device=device
        )
        if kernel_size < 8:
            raise ValueError("db4 requires kernel_size >= 8")
        return _pad_or_crop_to_kernel(h0, kernel_size), _pad_or_crop_to_kernel(h1, kernel_size)

    if init == "sym4":
        h0 = torch.tensor(
            [
                -0.07576571478927333,
                -0.02963552764599851,
                0.49761866763201545,
                0.8037387518059161,
                0.29785779560527736,
                -0.09921954357684722,
                -0.012603967262037833,
                0.0322231006040427
            ],
            dtype=dtype, device=device
        )
        h1 = torch.tensor(
            [
                -0.0322231006040427,
                0.012603967262037833,
                0.09921954357684722,
                0.29785779560527736,
                -0.8037387518059161,
                0.49761866763201545,
                0.02963552764599851,
                -0.07576571478927333
            ],
            dtype=dtype, device=device
        )
        if kernel_size < 8:
            raise ValueError("sym4 requires kernel_size >= 8")
        return _pad_or_crop_to_kernel(h0, kernel_size), _pad_or_crop_to_kernel(h1, kernel_size)

    if init == "sinc":
        if kernel_size < 8:
            raise ValueError("sinc init suggests kernel_size >= 8")
        n = torch.arange(kernel_size, device=device, dtype=dtype)
        m = (kernel_size - 1) / 2.0
        t = n - m

        fc = 0.25
        x = 2.0 * fc * t
        sinc = torch.where(torch.abs(x) < 1e-8, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))

        win = 0.54 - 0.46 * torch.cos(2.0 * math.pi * n / (kernel_size - 1))
        h0 = sinc * win
        h0 = h0 / (h0.sum() + 1e-12)

        delta = torch.zeros_like(h0)
        delta[int(m)] = 1.0
        h1 = delta - h0
        h1 = h1 - h1.mean()
        h1 = h1 / (h1.abs().sum() + 1e-12)
        return h0, h1

    raise ValueError("init must be one of: haar, randn, db2, sym2, db4, sym4, sinc")


class LearnableWavelet1D(nn.Module):
    def __init__(self, channels: int = 1, kernel_size: int = 2, init: str = "haar", learnable: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")

        pad = (self.kernel_size - 1) // 2
        self.pad = int(pad)

        self.analysis_low = ComplexConv1d(self.channels, self.channels, self.kernel_size, stride=2, padding=self.pad, bias=False)
        self.analysis_high = ComplexConv1d(self.channels, self.channels, self.kernel_size, stride=2, padding=self.pad, bias=False)

        self.synthesis_low = ComplexConvTranspose1d(self.channels, self.channels, self.kernel_size, stride=2, padding=self.pad, output_padding=0, bias=False)
        self.synthesis_high = ComplexConvTranspose1d(self.channels, self.channels, self.kernel_size, stride=2, padding=self.pad, output_padding=0, bias=False)

        self._init_filters(init)

        if not learnable:
            for p in self.parameters():
                p.requires_grad = False

    def _init_filters(self, init: str):
        init = str(init).lower()
        with torch.no_grad():
            if init == "randn":
                for m in [self.analysis_low, self.analysis_high]:
                    nn.init.normal_(m.conv_r.weight, mean=0.0, std=0.02)
                    nn.init.normal_(m.conv_i.weight, mean=0.0, std=0.02)
                for m in [self.synthesis_low, self.synthesis_high]:
                    nn.init.normal_(m.deconv_r.weight, mean=0.0, std=0.02)
                    nn.init.normal_(m.deconv_i.weight, mean=0.0, std=0.02)
                return

            h0, h1 = _wavelet_filters_1d(
                init=init,
                kernel_size=self.kernel_size,
                dtype=self.analysis_low.conv_r.weight.dtype,
                device=self.analysis_low.conv_r.weight.device
            )

            _place_diag_filter(self.analysis_low.conv_r.weight, h0)
            self.analysis_low.conv_i.weight.zero_()
            _place_diag_filter(self.analysis_high.conv_r.weight, h1)
            self.analysis_high.conv_i.weight.zero_()

            _place_diag_filter(self.synthesis_low.deconv_r.weight, h0)
            self.synthesis_low.deconv_i.weight.zero_()
            _place_diag_filter(self.synthesis_high.deconv_r.weight, h1)
            self.synthesis_high.deconv_i.weight.zero_()

    def analysis(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError("Expected x shape [B, C, L] complex.")
        return self.analysis_low(x), self.analysis_high(x)

    def synthesis(self, low: torch.Tensor, high: torch.Tensor, out_len: int):
        x = self.synthesis_low(low) + self.synthesis_high(high)
        if x.size(-1) > out_len:
            x = x[..., :out_len]
        elif x.size(-1) < out_len:
            x = F.pad(x, (0, out_len - x.size(-1)))
        return x