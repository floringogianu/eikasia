""" Entry point. """
from collections import deque
from copy import deepcopy
from functools import partial
from itertools import product

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


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnv0 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.act0 = nn.ReLU()
        self.cnv1 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.act1 = nn.ReLU()
        self.cnv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.act2 = nn.ReLU()
        self.lin0 = nn.Linear(8 * 8 * 32, 512)
        self.act3 = nn.ReLU()

    def forward(self, x):
        x = self.act0(self.cnv0(x))
        x = self.act1(self.cnv1(x))
        x = self.act2(self.cnv2(x))
        x = self.act3(self.lin0(x.view(x.shape[0], -1)))
        return x


class WorldModelLoop(nn.Module):
    def __init__(
        self, encoder, K=8, N=512, M=256, wrmup_steps=40, act_no=18, act_emb_sz=32
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.cls_loop = nn.GRU(N, M)
        self.opn_loop = nn.GRUCell(act_emb_sz, M)
        self.g = nn.Sequential(nn.Linear(M, M), nn.ReLU(), nn.Linear(M, N))
        self.act_emb = nn.Embedding(act_no, act_emb_sz)
        self.K = K
        self.wrmup_steps = wrmup_steps

    def forward(self, obs_seq, act_seq):
        T, B, C, W, H = obs_seq.shape

        z = obs_seq.view(B * T, C, W, H)
        z = self.encoder(z)
        z = z.view(T, B, -1)

        # split the sequences
        wrmup_seq = z[: self.wrmup_steps]
        train_seq = z[self.wrmup_steps :]
        act_seq = act_seq[self.wrmup_steps :]

        # closed loop integration of past timesteps
        # first we warmup the GRU
        if self.wrmup_steps:
            with torch.no_grad():
                _, warm_hn = self.cls_loop(wrmup_seq)
        else:
            warm_hn = None
        # then we compute the current state at each time t, given b_{0:t-1}
        bts, _ = self.cls_loop(train_seq, warm_hn)

        # open loop prediction of the future given actions
        output = []
        for t, bt in enumerate(bts[: -self.K]):
            bt_k = bt.clone()
            horizon = []
            for k in range(self.K):
                # get the action embeddings at time t+k
                act_idxs = act_seq[t + k, :]
                act_embs = self.act_emb(act_idxs)
                # increment the open loop gru
                bt_k = self.opn_loop(act_embs, bt_k)
                zt_k = self.g(bt_k)
                horizon.append(zt_k)
            output.append(torch.stack(horizon, dim=0))
        return torch.stack(output)


class WorldModel(nn.Module):
    def __init__(
        self, encoder, K=8, N=512, M=256, wrmup_steps=40, act_no=18, act_emb_sz=32
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.cls_loop = nn.GRU(N, M)
        self.opn_loop = nn.GRU(act_emb_sz, M)
        self.g = nn.Sequential(nn.Linear(M, M), nn.ReLU(), nn.Linear(M, N))
        self.act_emb = nn.Embedding(act_no, act_emb_sz)
        self.K = K
        self.wrmup_steps = wrmup_steps

    def forward(self, obs_seq, act_seq):
        T, B, C, W, H = obs_seq.shape

        z = obs_seq.view(B * T, C, W, H)
        z = self.encoder(z)
        z = z.view(T, B, -1)

        # split the sequences
        wrmup_seq = z[: self.wrmup_steps]
        train_seq = z[self.wrmup_steps :]
        act_seq = act_seq[self.wrmup_steps :]

        # closed loop integration of past timesteps
        # first we warmup the GRU
        if self.wrmup_steps:
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
    def __init__(
        self,
        alpha=0.99,
        K=8,
        N=512,
        M=256,
        wrmup_steps=40,
        act_no=18,
        act_emb_sz=32,
        wm=None,
    ) -> None:
        super().__init__()
        self.dynamics = wm(
            Encoder(),
            K=K,
            N=N,
            M=M,
            wrmup_steps=wrmup_steps,
            act_no=act_no,
            act_emb_sz=act_emb_sz,
        )
        self.target_encoder = deepcopy(self.dynamics.encoder)
        self.optim = torch.optim.Adam(self.dynamics.parameters(), lr=0.0001)
        self.alpha = alpha

    def train(self, obs_seq, act_seq):
        # cache some shapes
        P, K = self.dynamics.wrmup_steps, self.dynamics.K

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


def sample_seq(data, seq_len=80, burn_in=40):
    seq_len = seq_len + burn_in
    # concate in a deque, then yield.
    list_of_tuples = deque(maxlen=seq_len)
    for d in data:
        list_of_tuples.append(d)
        tuple_of_lists = tuple(zip(*list_of_tuples))
        if len(list_of_tuples) == seq_len:
            state_seq = torch.stack(tuple_of_lists[0], dim=0)
            # state_seq = torch.from_numpy(np.stack(tuple_of_lists[0], axis=0))
            action_seq, reward_seq, done_seq = list(
                zip(*[el.values() for el in tuple_of_lists[1]])
            )
            action_seq = torch.tensor(action_seq, dtype=torch.int64)
            # reward_seq = torch.tensor(reward_seq, dtype=torch.float32)
            yield (
                state_seq,
                action_seq,
                # reward_seq,
                # tuple_of_lists[2],
            )


def get_dloader():
    # ckpts = (
    #     "{00250000,33000000,44250000,48750000,15250000,"
    #     "38250000,46250000,50000000,25750000,41750000,47500000}"
    # )
    ckpts = "{00250000,33000000,44250000,48750000}"
    seeds = "1"
    path = f"./data/MDQN_rgb/Breakout/{seeds}/{ckpts}.tar"
    prep = T.Compose(
        [
            T.Lambda(lambda x: cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA)),
            T.ToTensor(),
        ]
    )
    dset = (
        wds.WebDataset(path, shardshuffle=True)
        .decode("rgb8")
        .to_tuple("state.png", "ard.msg", "__key__")
        .map_tuple(prep)
        .compose(sample_seq)
    )
    ldr = wds.WebLoader(dset, num_workers=16, pin_memory=True)
    ldr = ldr.shuffle(1000).batched(32)
    return ldr


def flatten(a):
    out = []
    for sublist in a:
        out.extend(sublist)
    return out


def run(opt):
    if __debug__:
        print("Code might have assertions. Use -O in liftoff.")

    opt.device = torch.device(opt.device)
    ioutil.create_paths(opt)

    # configure rlog
    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True, relative_time=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_loss", metargs=["trn_loss", 1]),
        rlog.FPSMetric("trn_tps", metargs=["trn_steps"]),
    )
    rlog.info(ioutil.config_to_string(opt))

    # get data and model
    ldr = get_dloader()
    net = BYOL(wrmup_steps=40).to(opt.device)

    def fix_dims(x):
        """Remove extra dims and make it time first, batch second."""
        return x.squeeze().transpose(0, 1).contiguous()

    #
    # Let's have some space, shall we?
    #

    for step, payload in enumerate(ldr):

        # fix the input shapes a bit
        # obs_seq, act_seq, _ = [fix_dims(x).to(opt.device) for x in payload[:-1]]
        obs_seq, act_seq = [fix_dims(x).to(opt.device) for x in payload]

        # one training step
        loss = net.train(obs_seq, act_seq)
        rlog.put(trn_loss=loss, trn_steps=1)

        if step % 100 == 0:
            rlog.traceAndLog(step)

        if step % 50_000 == 0 and step != 0:
            torch.save(
                {"model": net.state_dict()},
                f"{opt.out_dir}/model_{step:08d}.pkl",
            )


def train_mockup(model, data):
    model.train(*data)


def check_perf():
    device = torch.device("cuda")

    results = []
    for (B, T, L, K) in product((16, 32), (60, 80), (0, 20, 40), (2, 4, 8)):
        data = (
            torch.randn((T + L, B, 3, 96, 96), device=device),
            torch.randint(18, (T + L, B), device=device),
        )

        models = {
            "BYOL(loooop)": partial(BYOL, wrmup_steps=L, K=K, wm=WorldModelLoop),
            "BYOL(NOloop)": partial(BYOL, wrmup_steps=L, K=K, wm=WorldModel),
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
    check_perf()
    # run(parse_opts())


if __name__ == "__main__":
    main()
