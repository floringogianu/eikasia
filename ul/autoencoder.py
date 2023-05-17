from functools import partial
from itertools import chain

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector as param2vec

from ul import losses, nets
from ul.priors import DiagonalGaussianDistribution

__all__ = ["AutoEncoderKL"]


class AutoEncoderKL(nn.Module):
    def __init__(
        self, encoder, decoder, loss=None, optims=None, single_pass=True
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ante_emb = nn.Identity()  # nn.Conv2d(2 * z_dim, 2 * z_dim, 1, 1)
        self.post_emb = nn.Identity()  # nn.Conv2d(z_dim, z_dim, 1, 1)
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
        self.single_pass = single_pass
        self._step = 0

    @classmethod
    def from_opt(cls, opt):
        opt.encoder.args["z_ch"] = 2 * opt.z_dim
        opt.decoder.args["z_ch"] = opt.z_dim
        encoder = getattr(nets, opt.encoder.name)(**opt.encoder.args)
        decoder = getattr(nets, opt.decoder.name)(**opt.decoder.args)
        optim = partial(getattr(torch.optim, opt.optim.name), **opt.optim.args)
        loss = getattr(losses, opt.loss.name)(**opt.loss.args)
        return cls(
            encoder,
            decoder,
            loss=loss,
            optims={"g": optim, "d": optim},  # in this case they share settings
            single_pass=opt.single_pass
        )

    def forward(self, x):
        h = self.ante_emb(self.encoder(x))
        pzx = DiagonalGaussianDistribution(h)
        z = pzx.sample()
        x_ = self.decoder(self.post_emb(z))
        return x_, pzx, (h, z)

    def train(self, x, tgt=None):
        logs = []

        if self.single_pass:
            x_, pz_x, (h, z) = self(x)

        for optim_idx, optim in enumerate([self.optim_g, self.optim_d]):

            if not self.single_pass:
                x_, pz_x, (h, z) = self(x)

            # get the loss
            loss, log = self.loss(
                tgt if tgt is not None else x,
                x_,
                pz_x,
                optim_idx,  # either vae or discriminator loss
                self._step,
                last_layer=self._get_last_layer(),
                split="train",
            )

            # optimize
            optim.zero_grad()
            loss.backward()
            optim.step()

            # logging
            logs.append(log)

            if self._step % 100 == 0:
                if optim_idx == 0:
                    print(
                        (
                            "{:8d}   G_loss={:12.5f}, "
                            + "|z|={:8.3f}, |h|={:6.3f}, |E|={:6.3f}, |G|={:6.3f}"
                        ).format(
                            self._step,
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
                            self._step,
                            loss.detach().item(),
                            param2vec(self.loss.discriminator.parameters()).norm(),
                        )
                    )

        self._step += 1

        return losses, logs
    
    @property
    def step(self):
        return self._step

    def _get_last_layer(self):
        return self.decoder.last_layer

    def extra_repr(self) -> str:
        return f"single_pass={self.single_pass}"
