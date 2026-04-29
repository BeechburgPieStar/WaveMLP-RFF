import torch
import torch.nn as nn

class ComplexConv1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.conv_r = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_i = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.real, x.imag
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return torch.complex(yr, yi)


class ComplexConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.deconv_r = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        self.deconv_i = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.real, x.imag
        yr = self.deconv_r(xr) - self.deconv_i(xi)
        yi = self.deconv_r(xi) + self.deconv_i(xr)
        return torch.complex(yr, yi)