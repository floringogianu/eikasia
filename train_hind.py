""" Implementation of Curiosity in Hindsight on top of BYOL-Explore.
    Training is offlin only.
"""
import itertools
from copy import deepcopy

import rlog
import torch
from torch import nn

import common.io_utils as ioutil
from rl.encoders import ImpalaEncoder
from ul.data_loading import get_seq_loader
from ul.models import mlp


class WorldModel(nn.Module):
    def __init__(
        self,
        encoder,
        K=8,
        M=256,
        N=512,
        pfx_steps=40,
        act_no=18,
        act_emb_sz=32,
        act_emb_train=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.K, self.M, self.N, self.P = K, M, N, pfx_steps
        # modules
        self.encoder = encoder
        self.cls_loop = nn.GRU(N, M)
        self.opn_loop = nn.GRU(act_emb_sz, M)
        self.act_emb = nn.Embedding(act_no, act_emb_sz)
        self.act_emb.weight.requires_grad = act_emb_train

    def forward(self, obs_seq, act_seq):
        T, B, C, W, H = obs_seq.shape

        z = obs_seq.view(B * T, C, W, H)
        z = self.encoder(z)
        z = z.view(T, B, -1)

        # split the sequences
        wrmup_seq = z[: self.P]
        train_seq = z[self.P :]
        act_seq = act_seq[self.P :]

        # closed loop integration of past timesteps
        # first we warmup the GRU
        if self.P:
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
        T, B, M = bts.shape
        bts = bts.view(1, T * B, M)  # reshape in order to vectorize the time
        # we don't need the last action
        act_seq = act_seq[:-1, :]

        # take sliding windows of length K, these will be the inputs to the open loop
        act_k = act_seq.unfold(0, self.K, 1)
        act_k = act_k.view(T * B, self.K)
        # compute action embeddings
        act_embs = self.act_emb(act_k)
        # the horizon K is now the time dimension: K,T,act_emb_sz
        act_embs = act_embs.permute(1, 0, 2)

        # the important bit, check the paper
        # we unroll K times into the future, **starting from each timestep**
        out, _ = self.opn_loop(act_embs, bts)

        # out will be T,K+1,B,M for bts and T, K, B, M for actions
        out = out.view(self.K, T, B, M).permute(1, 0, 2, 3).contiguous()
        bts = bts.view(T, B, M).unsqueeze(1)
        out = torch.cat((bts, out), dim=1)

        act_embs = act_embs.view(self.K, T, B, -1).permute(1, 0, 2, 3).contiguous()
        return out, act_embs


class HindsightBYOL(nn.Module):
    """Manages training of the WorldModel."""

    def __init__(
        self,
        dyn_net,
        gen_net,
        rec_net,
        critic,
        optim,
        alpha=None,
        eps_dim=None,
        hin_dim=None,
        **kwargs,
    ) -> None:
        """Bootstrap Your Own Latent Model with Hindsight.

        Args:
            dyn_net (nn.Module): Dynamics prediction network, eg. WorldModel.
            gen_net (nn.Module): generator MLP from which we sample hindsight vectors Z
            rec_net (nn.Module): reconstruction MLP for following the targets
            critic (nn.Module): critic MLP
            alpha (float): Exponentiated moving average coefficient of the target net.
            optim (nn.optim.Optimizer): Optimizer for the `dynamics_net`.
        """
        super().__init__()
        self.dyn_net = dyn_net
        self.gen_net = gen_net
        self.rec_net = rec_net
        self.critic = critic
        self.optim = optim
        self.alpha = alpha
        self.eps_dim = eps_dim
        self.hin_dim = hin_dim
        self.target_encoder = deepcopy(self.dyn_net.encoder)

    @classmethod
    def from_opt(cls, opt):
        """Factory method for BYOL."""
        encoder = ImpalaEncoder(**opt.encoder.args)
        dyn_net = WorldModel(encoder, **opt.dynamics_net.args, **opt.shared)
        # figure out the rest
        M, N, act_emb_sz = [opt.shared[k] for k in ["M", "N", "act_emb_sz"]]

        # predictor:
        # prd_net = mlp(M, N, opt.args["hw_prd"])

        # generator: Z_(t,i) ~ p(. | b_(t,i-1), a_t+i-1, w_t+1)
        inp = M + act_emb_sz + N + opt.args["eps_dim"]
        gen_net = mlp(inp, opt.args["hin_dim"], opt.args["hw_gen"])

        # reconstructor: f(b_(t,i-1), a_t+i-1, Z_(t,i))
        inp = M + act_emb_sz + opt.args["hin_dim"]
        rec_net = mlp(inp, N, opt.args["hw_rec"])

        # critic: g(b_(t,i-1), a_t+i-1, Z_(t,i))
        inp = M + act_emb_sz + opt.args["hin_dim"]
        critic = mlp(inp, 1, opt.args["hw_critic"], link_fn=nn.Sigmoid)

        # register parameters in the optimizer
        optim = getattr(torch.optim, opt.optim.name)(
            itertools.chain(
                *[m.parameters() for m in [dyn_net, gen_net, rec_net, critic]]
            ),
            **opt.optim.args,
        )

        # construct
        return cls(dyn_net, gen_net, rec_net, critic, optim, **opt.args)

    def train(self, obs_seq, act_seq):
        """Training routine."""

        # cache some shapes
        P, K = self.dyn_net.P, self.dyn_net.K

        # get the targets
        with torch.no_grad():
            # ignore the warmup prefix. Targets are Wt+1,
            # that's why we also ignore the first observation.
            tgt_seq = obs_seq[P + 1 :]
            T, B, C, W, H = tgt_seq.shape
            # reshape for efficient encoding
            ys = tgt_seq.view(T * B, C, W, H)
            ys = self.target_encoder(ys)
            ys = ys.view(T, B, -1)

        # unroll the world model: btk = T,K+1,B,M, actk = T,K,B,emb_size
        btk, actk = self.dyn_net(obs_seq, act_seq)
        T, _, B, _ = btk.shape

        # here's the weirdness, for the generator we need b_t, b_t,1, b_t,2, ...
        _btk = btk[:, :K, :, :]
        # for the reconstruction and critic we need the open-loop activations instead
        btk_ = btk[:, 1:, :, :]

        # sample hindsight vectors Z
        # eps = torch.randn(T + K - 1, B, self.eps_dim, device=btk.device)
        epsk = torch.randn(T, K, B, self.eps_dim, device=btk.device)

        # unfold noise and targets (Wt+i in the paper)
        # epsk = eps.unfold(0, K, 1).permute(0, 3, 1, 2).contiguous()
        ysk = ys.unfold(0, K, 1).permute(0, 3, 1, 2)

        # sample hindsight vector
        # input is reshaped to T * K * B
        gen_net_inp = torch.cat([epsk, _btk, actk, ysk], dim=-1).view(T * K * B, -1)
        ztk = self.gen_net(gen_net_inp).view(T, K, B, -1)

        # compute reconstructions (Wt+i hat in the paper)
        rec_net_inp = torch.cat([btk_, actk, ztk], dim=-1).view(T * K * B, -1)
        wtk = self.rec_net(rec_net_inp).view(T, K, B, -1)

        # d = zip(["btk", "epsk", "actk", "ysk", "ztk"], [btk, epsk, actk, ysk, ztk])
        # for k, x in d:
        #     print(f"{k:<8s}", x.shape)

        # objective 1. Reconstruction
        rec_loss = nn.functional.mse_loss(wtk, ysk)

        # compute positive samples
        pos_inp = torch.cat([btk_, actk, ztk], dim=-1).view(T * K * B, -1)
        pos = torch.exp(self.critic(pos_inp)).view(T, K, B, -1)

        neg = []
        for i in range(B):
            bfix = btk_[:, :, i, :].unsqueeze(-2).expand(T, K, B - 1, btk_.shape[-1])
            afix = actk[:, :, i, :].unsqueeze(-2).expand(T, K, B - 1, actk.shape[-1])
            idxs = torch.tensor([x for x in range(B) if x != i])
            ztk_ = ztk[:, :, idxs, :]
            neg_inp = torch.cat([bfix, afix, ztk_], dim=-1).view(T * K * (B - 1), -1)
            neg_ = torch.exp(self.critic(neg_inp).view(T, K, (B - 1), -1))
            neg.append(neg_.sum(dim=2, keepdim=True))
        neg = torch.cat(neg, dim=2)

        # objective 2. Invariance
        inv_loss = torch.log(pos / (pos + neg).div(B)).mean()

        print("pos:{:8.3f}   neg:{:8.3f}".format(pos.data.mean().item(), neg.data.div(B-1).mean().item()))
        print("pos:{:8.3f}   neg:{:8.3f}".format(pos.data.max().item(), neg.data.div(B-1).max().item()))
        print("den:{:8.3f}".format((pos.data + neg.data).div(B).mean().item()))
        print("inv:{:8.5f}".format(inv_loss.data.item()))
        print("rec:{:8.5f}".format(rec_loss.data.item()))
        print("---")

        # total loss
        loss = rec_loss + inv_loss

        # optimize
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # EMA on the target network
        onl_params = self.dyn_net.encoder.parameters()
        tgt_params = self.target_encoder.parameters()
        for wo, wt in zip(onl_params, tgt_params):
            wt.data = self.alpha * wt.data + (1 - self.alpha) * wo.data

        return (x.detach().cpu().item() for x in [loss, rec_loss, inv_loss])


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
        rlog.AvgMetric("trn_rec_loss", metargs=["trn_rec_loss", 1]),
        rlog.AvgMetric("trn_inv_loss", metargs=["trn_inv_loss", 1]),
        rlog.AvgMetric("trn_ppos", metargs=["trn_ppos", 1]),
        rlog.AvgMetric("trn_pneg", metargs=["trn_pneg", 1]),
        rlog.FPSMetric("trn_sps", metargs=["trn_seq"]),
    )

    # get data and model
    ldr, dset = get_seq_loader(opt)
    model = HindsightBYOL.from_opt(opt.model).to(opt.device)

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
            loss, rec_loss, inv_loss = model.train(obs_seq, act_seq)
            rlog.put(
                trn_loss=loss,
                trn_rec_loss=rec_loss,
                trn_inv_loss=inv_loss,
                trn_seq=obs_seq.shape[1],
            )

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
    from liftoff import parse_opts
    run(parse_opts())

    # import sys
    # from common import io_utils as ioutil

    # opt = ioutil.read_config(sys.argv[1], info=False)
    # device = torch.device(opt.device)
    # print(ioutil.config_to_string(opt))

    # B = 16
    # P, T = [opt.model.dynamics_net.args[k] for k in ["pfx_steps", "seq_steps"]]

    # obs_seq = torch.randn((T + P, B, 1, 96, 96), device=device)
    # act_seq = torch.randint(18, (T + P, B), device=device)

    # model = HindsightBYOL.from_opt(opt.model).to(device)
    # loss, rec_loss, inv_loss = model.train(obs_seq, act_seq)
    # print(loss, rec_loss, inv_loss)


if __name__ == "__main__":
    main()
