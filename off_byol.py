import functools
import operator
from collections import deque
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import webdataset as wds


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


class WorldModel(nn.Module):
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
        with torch.no_grad():
            _, warm_hn = self.cls_loop(wrmup_seq)
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


class BYOL(nn.Module):
    def __init__(
        self, alpha=0.99, K=8, N=512, M=256, wrmup_steps=40, act_no=18, act_emb_sz=32
    ) -> None:
        super().__init__()
        self.dynamics = WorldModel(
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


def sample_seq(data):
    seq_len = 64 + 10
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
            reward_seq = torch.tensor(reward_seq, dtype=torch.float32)
            yield (
                state_seq,
                action_seq,
                reward_seq,
                tuple_of_lists[2],
            )


def get_dloader():
    path = "./data/Breakout/0/{00250000,38250000,48750000}.tar"
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
        .shuffle(1000)
    )
    ldr = wds.WebLoader(dset, num_workers=8)
    ldr = ldr.shuffle(1000).batched(32)
    return ldr


def flatten(a):
    out = []
    for sublist in a:
        out.extend(sublist)
    return out


def main():
    device = torch.device("cuda")
    ldr = get_dloader()
    net = BYOL(wrmup_steps=10).to(device)

    i = 0
    for payload in ldr:
        obs_seq, act_seq, _ = [
            x.squeeze().transpose(0, 1).contiguous().to(device) for x in payload[:-1]
        ]
        print(obs_seq.shape, act_seq.shape)

        loss = net.train(obs_seq, act_seq)
        print(f"{i:03}  loss={loss:08.5f}")
        i += 1

        # print([flatten(seq) for seq in payload[-1]])
        # print(state_seq.shape, action_seq.shape)


if __name__ == "__main__":
    main()
