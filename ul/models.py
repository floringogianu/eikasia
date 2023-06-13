from functools import partial
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector as param2vec

from .autoencoder import AutoEncoderKL


__all__ = ["AutoEncoderKL", "TimeConsistencyAE"]


def batch_randperm(x, dim=1):
    """Takes a batch and permutes it with a different random permutation on the
    specified dimension.
    """
    idxs = torch.argsort(torch.rand(x.shape[: dim + 1], device=x.device), dim=-1)
    for _ in range(x.ndim - dim - 1):
        idxs.unsqueeze_(-1)
    _idxs = idxs.repeat(*[1 for _ in range(dim + 1)], *(x.shape[dim + 1 :]))
    return x.gather(dim, _idxs), idxs.squeeze()


class TimeConsistencyAE(AutoEncoderKL):
    def __init__(self, encoder, decoder, loss=None, optims=None, single_pass=True):
        super().__init__(encoder, decoder, loss, optims, single_pass)

    def train(self, x):
        xperm, _ = batch_randperm(x)
        super().train(xperm, x)


class PairWiseTCModel(nn.Module):
    """Kind of a container class containing an MLP, an optimizer and the loss function."""

    def __init__(self, inp_dim, optimizer, out_dim=12):
        super().__init__()
        self.estimator = mlp(inp_dim, out_dim, [512])
        self.optimizer = optimizer(self.estimator.parameters())
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.estimator(x.flatten(1))


class PairWiseTimeConsistencyAE(nn.Module):
    def __init__(self, observation_model, pairwise_tc_model) -> None:
        super().__init__()
        self.observation_model = observation_model
        self.pairwise_tc_model = pairwise_tc_model
        # all the permutations of 2 elements in a set of 4 channel indices.
        self.register_buffer(
            "_permutations", torch.tensor(list(permutations(range(4), 2)))
        )
        self._step = 0

    def forward(self, obs):
        obs_, pz_obs, (h, z) = self.observation_model(obs)
        return obs_, pz_obs, (h, z)

    def train(self, x):
        x_perm, idx_perm = batch_randperm(x)
        x_, pz_x, (h, z) = self.observation_model(x_perm)

        # alias the optimizers:
        optim_g = self.observation_model.optim_g
        optim_d = self.observation_model.optim_d
        optim_t = self.pairwise_tc_model.optimizer

        # compute reconstruction loss
        reconstruction_loss, log = self.observation_model.loss(
            x,
            x_,
            pz_x,
            0,  # signal we want the reconstruction loss
            self._step,
            last_layer=self.observation_model._get_last_layer(),
            split="train",
        )

        # tc loss
        tc = self.pairwise_tc_model(h[:, :4])
        tc_tgt = self._get_pairwise_target(idx_perm)
        tc_loss = self.pairwise_tc_model.loss(tc, tc_tgt)

        # aggregate
        loss = reconstruction_loss + tc_loss

        optim_g.zero_grad()
        optim_t.zero_grad()
        loss.backward()  # compute gradients
        optim_g.step()
        optim_t.step()

        # now train the discriminator
        discriminator_loss, log = self.observation_model.loss(
            x,
            x_,
            pz_x,
            1,  # signal we want the discriminator loss
            self._step,
            split="train",
        )

        optim_d.zero_grad()
        discriminator_loss.backward()
        optim_d.step()

        if self._step % 100 == 0:
            print(
                (
                    "{:8d}  G_loss={:12.5f}, "
                    + "|z|={:8.3f}, |h|={:6.3f}, |E|={:6.3f}, |G|={:6.3f}\n"
                    + " " * 10
                    + "D_loss={:12.5f}, |D|={:8.3f}\n"
                    + " " * 10
                    + "T_loss={:6.3f}, |T|={:8.3f}"
                ).format(
                    self._step,
                    reconstruction_loss.detach().item(),
                    z.flatten().norm(),
                    h.flatten().norm(),
                    param2vec(self.observation_model.encoder.parameters()).norm(),
                    param2vec(self.observation_model.decoder.parameters()).norm(),
                    discriminator_loss.detach().item(),
                    param2vec(
                        self.observation_model.loss.discriminator.parameters()
                    ).norm(),
                    tc_loss.detach().item(),
                    param2vec(self.pairwise_tc_model.parameters()).norm(),
                )
            )

        self._step += 1

    def _get_pairwise_target(self, idx_perm):
        """Computes the target for the pairwise time consistency model given the
        permutation indexes.
        """
        pairs = idx_perm[:, self._permutations]
        return (pairs[:, :, 0] < pairs[:, :, 1]).float()

    @classmethod
    def from_opt(cls, opt):
        autoencoder = AutoEncoderKL.from_opt(opt.observation_model)
        pairwise_tc = PairWiseTCModel(
            4 * 16 * 16,
            partial(
                getattr(torch.optim, opt.pairwise_tc_model.optim.name),
                **opt.pairwise_tc_model.optim.args
            ),
        )
        return cls(autoencoder, pairwise_tc)

    def decoder(self, z):
        return self.observation_model.decoder(z)

    @property
    def step(self):
        return self._step

    # @staticmethod
    # def _get_latent_dims(f):
    #     x = torch.ones(1, 3, 96, 96, device=list(f.parameters())[0].device)
    #     z = f(x).detach()[:, :4]
    #     return z.shape[1:]


# Observation-Reward models


def mlp(inp: int, out: int, hid: list, act_fn="ReLU", link_fn=None) -> nn.Module:
    layers = []
    act_fn = getattr(nn, act_fn)
    for i, h in enumerate(hid):
        if i == 0:
            layers.append(nn.Linear(inp, hid[0]))
        layers.append(act_fn())
        layers.append(nn.Linear(h, h))
        if i == (len(hid) - 1):
            layers.append(act_fn())
            layers.append(nn.Linear(hid[-1], out))
    if link_fn is not None:
        layers.append(link_fn())
    return nn.Sequential(*layers)


class RewardModel(nn.Module):
    def __init__(self, layers, optimizer) -> None:
        super().__init__()
        inp, out = layers.pop(0), layers.pop()
        self.estimator = mlp(inp, out, layers)
        self.optimizer = optimizer(self.estimator.parameters())

    def forward(self, x):
        return self.estimator(x)


class ObservationRewardModel(nn.Module):
    def __init__(self, observation_model, reward_model) -> None:
        super().__init__()
        self.observation_model = observation_model
        self.reward_model = reward_model
        self._step = 0

    @classmethod
    def from_opt(cls, opt):
        autoencoder = AutoEncoderKL.from_opt(opt.observation_model)
        reward_model = RewardModel(
            [
                np.prod(cls._get_latent_dims(autoencoder.encoder)),
                *opt.reward_model.args["layers"],
                1,
            ],
            partial(
                getattr(torch.optim, opt.reward_model.optim.name),
                **opt.reward_model.optim.args
            ),
        )
        return cls(autoencoder, reward_model)

    def forward(self, obs):
        obs_, pz_obs, (h, z) = self.observation_model(obs)
        reward_ = self.reward_model(h[:, :4])
        return obs_, pz_obs, (h, z, reward_)

    def decoder(self, z):
        return self.observation_model.decoder(z)

    def train(self, x):
        obs, ard = x
        obs = obs.squeeze()
        reward = torch.stack([el["reward"] for el in ard]).float().to(obs.device)

        obs_, pz_obs, (h, z) = self.observation_model(obs)
        reward_ = self.reward_model(h[:, :4])

        # alias the optimizers:
        optim_g = self.observation_model.optim_g
        optim_d = self.observation_model.optim_d
        optim_r = self.reward_model.optimizer

        # compute reconstruction loss
        reconstruction_loss, log = self.observation_model.loss(
            obs,
            obs_,
            pz_obs,
            0,  # signal we want the reconstruction loss
            self._step,
            last_layer=self.observation_model._get_last_layer(),
            split="train",
        )

        # reward loss
        reward_loss = nn.functional.mse_loss(reward_, reward.clamp(-1, 1))
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
            self._step,
            split="train",
        )

        optim_d.zero_grad()
        discriminator_loss.backward()
        optim_d.step()

        if self._step % 100 == 0:
            print(
                (
                    "{:8d}  G_loss={:12.5f}, "
                    + "|z|={:8.3f}, |h|={:6.3f}, |E|={:6.3f}, |G|={:6.3f}\n"
                    + " " * 10
                    + "D_loss={:12.5f}, |D|={:8.3f}\n"
                    + " " * 10
                    + "R_loss={:6.3f}, |R|={:8.3f}"
                ).format(
                    self._step,
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

        self._step += 1

    @property
    def step(self):
        return self._step

    @staticmethod
    def _get_latent_dims(f):
        x = torch.ones(1, 3, 96, 96, device=list(f.parameters())[0].device)
        z = f(x).detach()[:, :4]
        return z.shape[1:]
