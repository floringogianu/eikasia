import torch
import torch.nn as nn
from .sd_enc_dec import Upsample, Downsample


class ResNetBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, bias=True) -> None:
        super().__init__()
        self.cnv0 = nn.Conv2d(
            inp_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.act0 = nn.SiLU(inplace=True)
        self.cnv1 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x_ = self.cnv1(self.act0(self.cnv0(x)))
        return x + x_


class ResNetStack(nn.Module):
    def __init__(self, inp_ch, out_ch, grp_norm, stack_sz=1) -> None:
        super().__init__()

        self.inp = (
            nn.Identity()
            if inp_ch == out_ch
            else nn.Conv2d(inp_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )
        self.act = nn.SiLU(inplace=True)
        self.stack = nn.ModuleList()

        for _ in range(stack_sz):
            self.stack.append(ResNetBlock(out_ch, out_ch, not grp_norm))

        if grp_norm:
            self.stack.append(nn.GroupNorm(num_groups=32, num_channels=out_ch))
        self.stack.append(nn.SiLU(inplace=True))

    def forward(self, x):
        x = self.act(self.inp(x))
        for m in self.stack:
            x = m(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        inp_ch=3,
        ch_mult=[1, 2, 4],
        base_width=64,
        downsample_res=[96, 48],
        z_ch=8,
        grp_norm=True,
        stack_sz=1,
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

        self.stacks = nn.ModuleList()

        res = downsample_res[0]
        stack_widths = [base_width * m for m in ch_mult]
        for i, out_ch in enumerate(stack_widths):
            grp_norm_ = grp_norm and i == len(stack_widths) - 1
            self.stacks.append(
                ResNetStack(inp_ch, out_ch, grp_norm_, stack_sz=stack_sz)
            )
            if res in downsample_res:
                self.stacks.append(Downsample(inp_ch, False))
                res = res // 2
            inp_ch = out_ch
        self.cnv1x1 = nn.Conv2d(out_ch, z_ch, kernel_size=1, bias=False)

    def forward(self, x):
        for stack in self.stacks:
            x = stack(x)
        return self.cnv1x1(x)


class Decoder(nn.Module):
    def __init__(
        self,
        rgb_ch=3,
        ch_mult=[2, 2, 1],
        base_width=64,
        upsample_res=[24, 48],
        z_ch=4,
        grp_norm=True,
        stack_sz=1,
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

        inp_ch = stack_widths[0]

        self.cnv1x1 = nn.Conv2d(z_ch, inp_ch, kernel_size=1)
        self.stacks = nn.ModuleList()

        res = upsample_res[0]
        for out_ch in stack_widths:
            self.stacks.append(ResNetStack(inp_ch, out_ch, grp_norm, stack_sz=stack_sz))
            if res in upsample_res:
                self.stacks.append(Upsample(out_ch, False))
                res = res * 2
            inp_ch = out_ch

        self.out = nn.Conv2d(inp_ch, rgb_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.cnv1x1(x)
        for stack in self.stacks:
            x = stack(x)
        return self.out(x)
    
    @property
    def last_layer(self):
        return self.out.weight


def main():
    C, D, Z = 3, 96, 4
    encoder = Encoder(C, z_ch=Z * 2, grp_norm=True)
    decoder = Decoder(C, z_ch=Z, grp_norm=True)

    print(encoder, "\n")
    print(decoder)

    print(f"encoder:  {sum([p.numel() for p in encoder.parameters()]):,}")
    print(f"decoder:  {sum([p.numel() for p in decoder.parameters()]):,}")

    x = torch.randn((1, C, D, D))
    z = encoder(x)
    x_ = decoder(z[:, :4, :, :])
    print("\nz:  ", z.shape)
    print("out:", x_.shape, "\n")


if __name__ == "__main__":
    main()
