""" Entry point. """
from copy import deepcopy

import rlog
import torch
import torch.nn as nn
import torch.utils.benchmark as bench
from liftoff import parse_opts

import common.io_utils as ioutil
from rl.encoders import ImpalaEncoder
from ul.data_loading import get_seq_loader
from ul.models import mlp


class WorldModel(nn.Module):
    def __init__(
        self,
        encoder,
        K=8,
        N=512,
        M=256,
        pfx_steps=40,
        act_no=18,
        act_emb_sz=32,
        act_emb_train=False,
        hw_mlp=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.K = K
        self.pfx_steps = pfx_steps
        # modules
        self.encoder = encoder
        self.cls_loop = nn.GRU(N, M)
        self.opn_loop = nn.GRU(act_emb_sz, M)
        self.g = mlp(M, N, hw_mlp)
        self.act_emb = nn.Embedding(act_no, act_emb_sz)
        self.act_emb.weight.requires_grad = act_emb_train

    def forward(self, obs_seq, act_seq):
        T, B, C, W, H = obs_seq.shape

        z = obs_seq.view(B * T, C, W, H)
        z = self.encoder(z)
        z = z.view(T, B, -1)

        # split the sequences
        wrmup_seq = z[: self.pfx_steps]
        train_seq = z[self.pfx_steps :]
        act_seq = act_seq[self.pfx_steps :]

        # closed loop integration of past timesteps
        # first we warmup the GRU
        if self.pfx_steps:
            with torch.no_grad():
                _, warm_hn = self.cls_loop(wrmup_seq)
        else:
            warm_hn = None
        # then we compute the current state at each time t, given b_{0:t-1}
        # and get T x B x 1*M
        bts, _ = self.cls_loop(train_seq, warm_hn)

        # open loop prediction of the future given actions
        # we will vectorize the open loop prediction
        # since it only depends on b_t and a_{t+k}
        bts = bts[: -self.K]  # that means we don't need the last K states
        T, B, M = bts.shape  # WARN: T,B change here
        bts = bts.view(1, T * B, M)  # reshape in order to vectorize the time
        # we don't need the last action
        act_seq = act_seq[:-1, :]
        # take sliding windows of length K, these will be the inputs to the open loop
        act_k = act_seq.unfold(0, self.K, 1)
        act_k = act_k.view(T * B, self.K)
        # compute action embeddings
        act_embs = self.act_emb(act_k)
        # the horizon K is now the time dimension
        act_embs = act_embs.permute(1, 0, 2)

        # the important bit
        out, _ = self.opn_loop(act_embs, bts)
        # final projection
        out = out.view(self.K * T * B, M)  # batch everything :)
        out = self.g(out)
        return out.view(self.K, T, B, -1).permute(1, 0, 2, 3)


class BYOL(nn.Module):
    def __init__(self, dynamics_net, optim, alpha=None) -> None:
        """Bootstrap Your Own Latent Model.

        Args:
            alpha (float): Exponentiated moving average coefficient of the target net.
            dynamics_net (nn.Module): Dynamics prediction network.
            optim (nn.optim.Optimizer): Optimizer for the `dynamics_net`.
        """
        super().__init__()
        self.alpha = alpha
        self.dynamics = dynamics_net
        self.optim = optim
        self.target_encoder = deepcopy(self.dynamics.encoder)

    @classmethod
    def from_opt(cls, opt):
        encoder = ImpalaEncoder(**opt.encoder.args)
        dynet = m = WorldModel(encoder, **opt.dynamics_net.args)
        optim = getattr(torch.optim, opt.optim.name)(m.parameters(), **opt.optim.args)
        return cls(dynet, optim, **opt.args)

    def train(self, obs_seq, act_seq):
        # cache some shapes
        P, K = self.dynamics.pfx_steps, self.dynamics.K

        # unroll the world model
        predictions = self.dynamics(obs_seq, act_seq)

        # get the targets
        with torch.no_grad():
            # ignore the warmup prefix. Targets are Wt+1,
            # that's why we also ignore the first observation.
            tgt_seq = obs_seq[P + 1 :]
            T, B, C, W, H = tgt_seq.shape
            # reshape for efficient encoding
            ys = tgt_seq.view(B * T, C, W, H)
            ys = self.target_encoder(ys)
            ys = ys.view(T, B, -1)

        losses = []
        for t, z_t in enumerate(predictions):
            y = ys[t : t + K]
            # print(f"{t:03}:{(t+K):03}", z_t.shape, y.shape)
            losses.append(2 - 2 * nn.functional.cosine_similarity(z_t, y, dim=-1))
            # losses.append(nn.functional.mse_loss(z_t, y))
        loss = torch.stack(losses).mean()

        # optimize
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # ema
        onl_params = self.dynamics.encoder.parameters()
        tgt_params = self.target_encoder.parameters()
        for wo, wt in zip(onl_params, tgt_params):
            wt.data = self.alpha * wt.data + (1 - self.alpha) * wo.data

        return loss.detach().cpu().item()


def runtime_opt_(opt):
    """Use this method to add any runtime opt fields are necessary."""
    opt.device = torch.device(opt.device)
    opt.save_freq = int(opt.base_save_freq / opt.model.dynamics_net.args["seq_steps"])
    return opt


def run(opt):
    if __debug__:
        print("Code might have assertions. Use -O in liftoff.")
    torch.backends.cudnn.benchmark = True

    # runtime options
    runtime_opt_(opt)

    ioutil.create_paths(opt)
    # configure rlog
    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True, relative_time=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_loss", metargs=["trn_loss", 1]),
        rlog.FPSMetric("trn_sps", metargs=["trn_seq"]),
    )

    # get data and model
    ldr, dset = get_seq_loader(opt)
    model = BYOL.from_opt(opt.model).to(opt.device)

    # some logging
    rlog.info(ioutil.config_to_string(opt))
    rlog.info(model)
    rlog.info(dset)

    # save the resulting config
    ioutil.save_config(opt)

    #
    # Let's have some space, shall we?
    #

    step = 0
    for epoch in range(1, opt.epochs + 1):
        for obs_seq, act_seq in ldr:

            # fix the input shapes a bit
            act_seq = act_seq.transpose(0, 1).contiguous().to(opt.device)
            obs_seq = obs_seq.transpose(0, 1).contiguous().to(opt.device)
            obs_seq = obs_seq.float().div(255.0)

            # one training step
            loss = model.train(obs_seq, act_seq)
            rlog.put(trn_loss=loss, trn_seq=obs_seq.shape[1])

            if step % 1000 == 0 and step != 0:
                rlog.traceAndLog(step)

            if step % opt.save_freq == 0 and step != 0:
                torch.save(
                    {"model": model.state_dict()},
                    f"{opt.out_dir}/model_{step:08d}.pkl",
                )
                rlog.info("Saved model.")

            step += 1
        rlog.info(f"Done epoch {epoch}.")


def main():
    """Liftoff"""
    run(parse_opts())


if __name__ == "__main__":
    main()
