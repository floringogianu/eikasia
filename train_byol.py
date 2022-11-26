""" Entry point. """
from collections import deque
from copy import deepcopy
from functools import partial
from itertools import product
import random

import cv2
import numpy as np
import rlog
import torch
import torch.nn as nn
import torch.utils.benchmark as bench
import torchvision.transforms as T
import webdataset as wds
from liftoff import parse_opts

import common.io_utils as ioutil
from rl.encoders import ImpalaEncoder


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
        **kwargs,
    ) -> None:
        super().__init__()
        self.K = K
        self.pfx_steps = pfx_steps
        # modules
        self.encoder = encoder
        self.cls_loop = nn.GRU(N, M)
        self.opn_loop = nn.GRU(act_emb_sz, M)
        self.g = nn.Sequential(nn.Linear(M, M), nn.ReLU(), nn.Linear(M, N))
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
    def __init__(self, alpha, dynamics_net, optim) -> None:
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
        return cls(opt.alpha, dynet, optim)

    def train(self, obs_seq, act_seq):
        # cache some shapes
        P, K = self.dynamics.pfx_steps, self.dynamics.K

        # unroll the world model
        predictions = self.dynamics(obs_seq, act_seq)

        # get the targets
        with torch.no_grad():
            # ignore the warmup prefix and the first observation
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


def is_valid(seq, seq_steps):
    """Checks the sequence is not crossing shard boundaries.
    We do this by checking the idxs are consistent.
    """
    diff = int(seq[-1].split(":")[-1]) - int(seq[0].split(":")[-1])
    return diff == (seq_steps - 1)


def sliding_window(data, seq_steps, subsample):
    """Used by webdataset to compose sliding windows of samples.
    A sample is usually a tuple (image, dict, __key__). The key can be used to ID the sample.

    Args:
        data (generator): sequence of samples from webdataset.
        seq_steps (int): length of the sequence used for training.

    Yields:
        tuple: Sequences of frames, actions, rewards, etc...
    """
    # concate in a deque, then yield if conditions apply
    list_of_tuples = deque(maxlen=seq_steps)
    for i, d in enumerate(data):
        list_of_tuples.append(d)
        if len(list_of_tuples) == seq_steps:  # deque reached required size

            # we want to avoid overfitting so we only sample every other
            # subsample / seq_steps
            if subsample and (random.random() > (subsample / seq_steps)):
                continue

            tuple_of_lists = tuple(zip(*list_of_tuples))
            keys = tuple_of_lists[2]
            if is_valid(keys, seq_steps):  # and the sequence is valid
                # arrange the data in tensors the size of sequence lenght
                state_seq = (
                    torch.from_numpy(np.stack(tuple_of_lists[0], axis=0))
                    .unsqueeze_(1)
                    .contiguous()
                )
                # actions, rewards and done are in a dictionary at this point
                action_seq, reward_seq, done_seq = list(
                    zip(*[el.values() for el in tuple_of_lists[1]])
                )
                action_seq = torch.tensor(action_seq, dtype=torch.int64)
                # reward_seq = torch.tensor(reward_seq, dtype=torch.float32)
                yield (state_seq, action_seq)


def get_dloader(opt):
    prep = T.Lambda(
        lambda x: cv2.resize(
            cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (96, 96), interpolation=cv2.INTER_AREA
        )
    )

    # we use a sliding window to get sequences of total length given by
    # the the training length + the warm-up lenght required by the RNN
    seq_length = sum([opt.dynamics_net.args[k] for k in ("seq_steps", "pfx_steps")])
    sequencer = partial(
        sliding_window, seq_steps=seq_length, subsample=opt.dset.subsample
    )

    dset = (
        wds.WebDataset(opt.dset.path, shardshuffle=True)
        .decode("rgb8")
        .to_tuple("state.png", "ard.msg", "__key__")
        .map_tuple(prep, None, None)
        .compose(sequencer)
        .shuffle(opt.dset.shuffle)
        .batched(opt.dset.batch_size)
    )
    ldr = wds.WebLoader(dset, **opt.loader.args)
    ldr = ldr.unbatched().shuffle(opt.loader.shuffle).batched(opt.loader.batch_size)
    return ldr, dset


def runtime_opt_(opt):
    """Use this method to add any runtime opt fields are necessary."""
    opt.device = torch.device(opt.device)
    opt.save_freq = int(opt.base_save_freq / opt.dynamics_net.args["seq_steps"])
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
    ldr, dset = get_dloader(opt)
    model = BYOL.from_opt(opt).to(opt.device)

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


def train_mockup(model, data):
    model.train(*data)


def check_perf():
    device = torch.device("cuda")

    results = []
    for (B, T, L) in product((16, 32), (60, 80), (0, 20, 40)):
        data = (
            torch.randn((T + L, B, 3, 96, 96), device=device),
            torch.randint(18, (T + L, B), device=device),
        )

        models = {
            "BYOL(K=2)": partial(BYOL, pfx_steps=L, K=2),
            "BYOL(K=4)": partial(BYOL, pfx_steps=L, K=4),
            "BYOL(K=8)": partial(BYOL, pfx_steps=L, K=8),
        }

        for model_name, model_fn in models.items():
            model = model_fn().to(device)

            results.append(
                bench.Timer(
                    stmt="train_mockup(model, data)",
                    setup="from __main__ import train_mockup",
                    globals={"model": model, "data": data},
                    label="TRAIN one sequence",
                    sub_label=f"B={B:3d},T={T:3d},L={L:3d}",
                    description=model_name,
                ).blocked_autorange(min_run_time=2)
            )
    compare = bench.Compare(results)
    compare.colorize()
    compare.print()


def main():
    """Liftoff"""
    # check_perf()
    run(parse_opts())


if __name__ == "__main__":
    main()
