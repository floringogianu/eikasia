import torch
import torch.nn as nn


class Encoder(nn.Module):
    """WorldModels/Dreamer/DreamerV2 -ish encoder."""

    def __init__(
        self,
        inp_ch,
        num_layers=5,
        base_width=48,
        z_ch=4,
        max_width=512,
        group_norm=True,
        **kwargs,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            out_ch = min(2**i * base_width, max_width)
            bias = not (group_norm and i == (num_layers - 1))
            self.layers.append(
                nn.Conv2d(
                    inp_ch,
                    out_ch,
                    kernel_size=4 if i <= 1 else 3,
                    stride=2 if i <= 1 else 1,
                    bias=bias,
                )
            )
            if bias:
                self.layers.append(nn.SiLU(inplace=True))
            else:
                self.layers.append(
                    nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
                )
                self.layers.append(nn.SiLU(inplace=True))
            inp_ch = out_ch
        self.layers.append(nn.Conv2d(inp_ch, z_ch, 1, 1))

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


class Decoder(nn.Module):
    """WorldModels/Dreamer/DreamerV2 -ish decoder."""

    def __init__(
        self,
        out_ch,
        num_layers=5,
        base_width=32,
        z_ch=4,
        max_width=256,
        group_norm=True,
        **kwargs,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        out_ch_ = min(2 ** (num_layers - 1) * base_width, max_width)
        self.layers.append(nn.Conv2d(z_ch, out_ch_, 1, 1))
        self.layers.append(nn.GroupNorm(num_groups=32, num_channels=out_ch_))
        self.layers.append(nn.SiLU(inplace=True))
        z_ch = out_ch_

        ks = [(8, 1), (8, 1), (7, 1), (6, 2), (6, 2)]
        # ks = [(7, 1), (7, 1), (6, 2), (6, 2)]
        for i in range(num_layers - 1, -1, -1):

            k, s = ks[i]
            out_ch_ = min(2 ** (i - 1) * base_width, max_width)

            self.layers.append(
                nn.ConvTranspose2d(
                    z_ch,
                    out_ch if i == 0 else out_ch_,
                    kernel_size=k,
                    stride=s,
                    bias=not group_norm,
                )
            )
            if group_norm and (out_ch_ >= 64):
                self.layers.append(nn.GroupNorm(num_groups=32, num_channels=out_ch_))
            if i != 0:
                self.layers.append(nn.SiLU(inplace=True))
            z_ch = out_ch_

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

    def mode(self):
        return self.mean
    
    @property
    def last_layer(self):
        return self.layers[-1].weight


def main():
    C, D, Z = 3, 96, 4
    encoder = Encoder(C, z_ch=Z * 2)
    decoder = Decoder(C, z_ch=Z)

    print(f"encoder:  {sum([p.numel() for p in encoder.parameters()]):,}")
    print(f"decoder:  {sum([p.numel() for p in decoder.parameters()]):,}")

    x = torch.randn((1, C, D, D))
    z = encoder(x)
    x_ = decoder(z[:, :4, :, :])
    print("\nz:  ", z.shape)
    print("out:", x_.shape, "\n")
    print(encoder)
    print(decoder)


if __name__ == "__main__":
    main()
