from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector as param2vec

from .autoencoder import AutoEncoderKL


class RewardModel(nn.Module):
    def __init__(self, inp_size, optimizer) -> None:
        super().__init__()
        self.estimator = nn.Linear(inp_size, 1, bias=False)
        self.optimizer = optimizer(self.estimator.parameters())

    def forward(self, x):
        return self.estimator(x.flatten(1))


class ObservationRewardModel(nn.Module):
    def __init__(self, observation_model, reward_model) -> None:
        super().__init__()
        self.observation_model = observation_model
        self.reward_model = reward_model
        self._step = 0

    @classmethod
    def init_from_opts(cls, opt):
        autoencoder = AutoEncoderKL.from_opt(opt.model.observation_model)
        z_dim = cls._get_latent_dims(autoencoder.encoder, opt)
        optim = (partial(getattr(torch.optim, opt.optim.name), **opt.optim.args),)
        reward_model = RewardModel(np.prod(z_dim), optim)
        return cls(autoencoder, reward_model)

    def train(self, x):
        obs, reward = x
        obs_, pz_obs, (h, z) = self.observation_model(obs)
        reward_ = self.reward_model(h[:, :4])

        # alias the optimizers:
        optim_g = self.observation_model.optim_g
        optim_d = self.observation_model.optim_d
        optim_r = self.reward_model.optim

        # compute reconstruction loss
        reconstruction_loss, log = self.observation_model.loss(
            obs,
            obs_,
            pz_obs,
            0,  # signal we want the reconstruction loss
            self.step,
            last_layer=self.observation_model._get_last_layer(),
            split="train",
        )

        # reward loss
        reward_loss = nn.functional.mse_loss(reward_, reward)
        loss = reconstruction_loss + reward_loss

        optim_g.zero_grad()
        optim_r.zero_grad()
        loss.backward()  # compute gradients
        optim_g.step()
        optim_r.step()

        # now train the discriminator
        # compute reconstruction loss
        discriminator_loss, log = self.observation_model.loss(
            obs,
            obs_,
            pz_obs,
            1,  # signal we want the discriminator loss
            self.step,
            split="train",
        )

        optim_d.zero_grad()
        discriminator_loss.bacward()
        optim_d.step()

        if self.step % 100 == 0:
            print(
                (
                    "{:8d}  G_loss={:12.5f}, "
                    + "|z|={:8.3f}, |h|={:6.3f}, |E|={:6.3f}, |G|={:6.3f}\n"
                    + " " * 10
                    + "D_loss={:12.5f}, |D|={:8.3f}\n"
                    + " " * 10
                    + "R_loss={:6.3f}, |R|={:8.3f}"
                ).format(
                    self.step,
                    reconstruction_loss.detach().item(),
                    z.flatten().norm(),
                    h.flatten().norm(),
                    param2vec(self.observation_model.encoder.parameters()).norm(),
                    param2vec(self.observation_model.decoder.parameters()).norm(),
                    discriminator_loss.detach().item(),
                    param2vec(
                        self.observation_model.loss.discriminator.parameters()
                    ).norm(),
                    reward_loss.detach().item(),
                    param2vec(self.reward_model.parameters()).norm(),
                )
            )

    @staticmethod
    def _get_latent_dims(f, opt):
        inp_ch = opt.estimator.encoder.args["inp_ch"]
        inp_dim = (1, 1, inp_ch, *opt.env.args["obs_dims"])
        x = torch.ones(
            *inp_dim, device=list(f.parameters())[0].device, dtype=torch.uint8
        )
        z = f(x).detach()[:, :4]
        return z.shape[1:]
