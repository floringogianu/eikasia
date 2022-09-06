import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..priors import DiagonalGaussianDistribution


class Encoder(nn.Module):
    """WorldModels/Dreamer/DreamerV2 -ish encoder."""

    def __init__(
        self, inp_ch, num_layers=5, base_width=48, z_ch=4, max_width=512, **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            out_ch = min(2**i * base_width, max_width)
            self.layers.append(
                nn.Conv2d(
                    inp_ch,
                    out_ch,
                    kernel_size=4 if i <= 1 else 3,
                    stride=2 if i <= 1 else 1,
                )
            )
            inp_ch = out_ch
        self.layers.append(nn.Conv2d(inp_ch, z_ch, 1, 1))

    def forward(self, x):
        for m in self.layers[:-1]:
            x = F.relu(m(x))
        return self.layers[-1](x)


class Decoder(nn.Module):
    """WorldModels/Dreamer/DreamerV2 -ish decoder."""

    def __init__(
        self, out_ch, num_layers=5, base_width=32, z_ch=4, max_width=256, **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        out_ch_ = min(2 ** (num_layers - 1) * base_width, max_width)
        self.layers.append(nn.Conv2d(z_ch, out_ch_, 1, 1))
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
                )
            )
            z_ch = out_ch_

    def forward(self, x):
        for m in self.layers[:-1]:
            x = F.relu(m(x))
        return self.layers[-1](x)

    def mode(self):
        return self.mean


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

    return

    optim = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()),
        # [{"params": encoder.parameters(), "params": decoder.parameters()}],
        lr=0.0001,
    )

    x = torch.rand((1, C, D, D))

    for i in range(1000):
        h = encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        z = posterior.sample()
        x_ = decoder(z)

        optim.zero_grad()
        rec_loss = F.mse_loss(x_, x)
        kl_loss = posterior.kl().sum()
        loss = rec_loss + 0.5 * kl_loss
        loss.backward()
        optim.step()

        if i % 100 == 0:
            enc_norm = nn.utils.parameters_to_vector(encoder.parameters()).norm()
            dec_norm = nn.utils.parameters_to_vector(decoder.parameters()).norm()
            print(
                "{:05d}  loss={:2.6f},  |z|={:9.5f}, |h|={:9.5f}, |E|={:9.5f}, |G|={:9.5f}".format(
                    i,
                    loss.detach().item(),
                    z.flatten().norm(),
                    h.flatten().norm(),
                    enc_norm,
                    dec_norm,
                )
            )


if __name__ == "__main__":
    main()
