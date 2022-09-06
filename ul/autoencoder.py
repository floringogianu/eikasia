from functools import partial
from itertools import chain

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector as param2vec

from ul.nets import WMDecoder, WMEncoder
from ul.priors import DiagonalGaussianDistribution
import ul.losses as losses

__all__ = ["AutoEncoderKL"]


class AutoEncoderKL(nn.Module):
    def __init__(self, inp_ch, z_dim=4, loss=None, optims=None) -> None:
        super().__init__()
        self.encoder = WMEncoder(inp_ch, z_ch=2 * z_dim)
        self.decoder = WMDecoder(inp_ch, z_ch=z_dim)
        self.ante_emb = nn.Identity()#nn.Conv2d(2 * z_dim, 2 * z_dim, 1, 1)
        self.post_emb = nn.Identity()#nn.Conv2d(z_dim, z_dim, 1, 1)
        self.loss = loss
        self.optim_g = optims["g"](
            chain(
                *[
                    m.parameters()
                    for m in [self.encoder, self.decoder, self.ante_emb, self.post_emb]
                ]
            )
        )
        self.optim_d = optims["d"](loss.discriminator.parameters())
        self.step = 0

    @classmethod
    def from_opt(cls, opt):
        optim = partial(getattr(torch.optim, opt.optim.name), **opt.optim.args)
        loss = getattr(losses, opt.loss.name)(**opt.loss.args)
        return cls(
            opt.args["inp_ch"],
            z_dim=opt.args["z_dim"],
            loss=loss,
            optims={"g": optim, "d": optim},  # in this case they share settings
        )

    def forward(self, x):
        h = self.ante_emb(self.encoder(x))
        pzx = DiagonalGaussianDistribution(h)
        z = pzx.sample()
        x_ = self.decoder(self.post_emb(z))
        return x_, pzx, (h, z)

    def train(self, x):
        logs = []

        for optim_idx, optim in enumerate([self.optim_g, self.optim_d]):

            # forward
            x_, pz_x, (h, z) = self(x)

            # get the loss
            loss, log = self.loss(
                x,
                x_,
                pz_x,
                optim_idx,  # either vae or discriminator loss
                self.step,
                last_layer=self._get_last_layer(),
                split="train",
            )

            # optimize
            optim.zero_grad()
            loss.backward()
            optim.step()

            # logging
            logs.append(log)

            if self.step % 100 == 0:
                if optim_idx == 0:
                    print(
                        (
                            "{:8d}   G_loss={:12.5f}, "
                            + "|z|={:8.3f}, |h|={:6.3f}, |E|={:6.3f}, |G|={:6.3f}"
                        ).format(
                            self.step,
                            loss.detach().item(),
                            z.flatten().norm(),
                            h.flatten().norm(),
                            param2vec(self.encoder.parameters()).norm(),
                            param2vec(self.decoder.parameters()).norm(),
                        )
                    )
                else:
                    print(
                        "{:8d}   D_loss={:12.5f}, |D|={:8.3f}".format(
                            self.step,
                            loss.detach().item(),
                            param2vec(self.loss.discriminator.parameters()).norm(),
                        )
                    )

        self.step += 1

        return losses, logs

    def _get_last_layer(self):
        return self.decoder.layers[-1].weight
