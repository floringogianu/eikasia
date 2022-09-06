import torch
import torch.nn as nn
import torch.nn.functional as F


def _infer_phi_dims(f, inp_ch, inp_size):
    x = torch.randn((1, inp_ch, inp_size, inp_size))
    return f(x).flatten().detach().numel()


class ResNetBlock(nn.Module):
    def __init__(self, ch=64) -> None:
        super().__init__()
        self.cnv0 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.cnv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_ = self.cnv1(F.relu(self.cnv0(F.relu(x))))
        return x + x_


class ResNetStack(nn.Module):
    def __init__(self, inp_ch, out_ch, grp_norm, k=3, s=1) -> None:
        super().__init__()
        self.stack = nn.ModuleList(
            [
                nn.Conv2d(inp_ch, out_ch, kernel_size=k, stride=s, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResNetBlock(out_ch),
                ResNetBlock(out_ch),
            ]
        )
        if grp_norm:
            self.stack.append(nn.GroupNorm(num_groups=1, num_channels=out_ch))

    def forward(self, x):
        for m in self.stack:
            x = m(x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(
        self,
        inp_ch=4,
        ch_mult=[1, 2, 2],
        base_width=32,
        grp_norm=False,
        z_ch=8,
        **kwargs,
    ):
        """A ResNet-based Encoder

        Args:
            inp_ch (int, optional): Number of input channels. Defaults to 4.
            ch_mult (list, optional): Channel multipliers per stack. The resulting width is `base_width x multiplier`. Defaults to [1, 2, 2].
            grp_norm (bool, optional): Whether the stacks use GroupNorm. Defaults to False.
            base_width (int, optional): The base width of each stack. Defaults to 16.
        """
        super().__init__()
        stack_widths = [base_width * m for m in ch_mult]
        self.stacks = nn.ModuleList()
        for out_ch in stack_widths:
            self.stacks.append(ResNetStack(inp_ch, out_ch, grp_norm))
            inp_ch = out_ch
        self.cnv1x1 = nn.Conv2d(out_ch, z_ch, kernel_size=1)

    def forward(self, x):
        for stack in self.stacks:
            x = stack(x)
        return self.cnv1x1(x)
