from functools import partial
from itertools import product

import torch
import torch.nn as nn
import torch.utils.benchmark as bench


def loop(rnn, emb, bts, act_seq, K):
    # open loop prediction of the future given actions
    output = []
    for t, bt in enumerate(bts[:-K]):
        bt_k = bt.clone()
        horizon = []
        for k in range(K):
            # get the action embeddings at time t+k
            act_idxs = act_seq[t + k, :]
            act_embs = emb(act_idxs)
            # increment the open loop gru
            bt_k = rnn(act_embs, bt_k)
            horizon.append(bt_k)
        output.append(torch.stack(horizon, dim=0))
    return torch.stack(output).squeeze()


def ufld(rnn, emb, bts, act_seq, K):
    bts = bts[:-K]
    T, B, M = bts.shape
    bts = bts.view(T * B, M)

    act_k = act_seq.unfold(0, K, 1)
    act_k = act_k.view(T * B, K)

    horizon = []
    for k in range(K):
        act_idxs = act_k[:, k]
        act_embs = emb(act_idxs)
        bts = rnn(act_embs, bts)
        horizon.append(bts.view(T, B, -1))
    return torch.stack(horizon).permute(1, 0, 2, 3)


def cdnn(rnn, emb, bts, act_seq, K):
    bts = bts[:-K]
    T, B, M = bts.shape
    bts = bts.view(1, T * B, M)  # cudnn lstm needs 1 x B x H

    act_k = act_seq.unfold(0, K, 1)
    act_k = act_k.view(T * B, K)
    act_embs = emb(act_k).permute(1, 0, 2)
    out, _ = rnn(act_embs, bts)
    out = out.view(K, T, B, M).permute(1, 0, 2, 3)
    return out


def test(fn):
    fn()


def compare():
    M = 256
    A, anum = 2, 5  # action embedding size, number of actions
    device = torch.device("cuda")

    results = []
    for (T, B, K) in product((64, 128), (32, 64), (2, 4, 8)):
        rnn = nn.GRUCell(A, M).to(device)
        gru = nn.GRU(A, M).to(device)
        emb = nn.Embedding(anum, A).to(device)

        bts = torch.randn((T, B, M), device=device)
        act_seq = torch.randint(1, anum, (T - 1, B), device=device)

        fns = {
            "loop": partial(loop, rnn, emb, bts, act_seq, K),
            "ufld": partial(ufld, rnn, emb, bts, act_seq, K),
            "cdnn": partial(cdnn, gru, emb, bts, act_seq, K),
        }

        for fn_name, fn in fns.items():
            results.append(
                bench.Timer(
                    stmt="test(fn)",
                    setup="from __main__ import test",
                    globals={"fn": fn},
                    label="TRAIN one sequence",
                    sub_label=f"T={T:3d},B={B:3d},K={K:3d}",
                    description=fn_name,
                ).blocked_autorange(min_run_time=2)
            )
    compare = bench.Compare(results)
    compare.colorize(rowwise=True)
    compare.print()


def main():
    B, T = 12, 9
    M, K = 4, 3  # hidden size, horizon
    A, anum = 2, 5  # action embedding size, number of actions

    rnn = nn.GRUCell(A, M)
    gru = nn.GRU(A, M)
    for p0, p1 in zip(rnn.parameters(), gru.parameters()):
        p1.data = p0.data.clone()
    emb = nn.Embedding(anum, A)

    bts = torch.randn(T, B, M)
    act_seq = torch.randint(1, anum, (T - 1, B))

    res_loop = loop(rnn, emb, bts, act_seq, K)
    res_ufld = ufld(rnn, emb, bts, act_seq, K)
    res_cdnn = cdnn(gru, emb, bts, act_seq, K)

    print("\n MSE(unfold - loop):", nn.MSELoss()(res_ufld, res_loop).item())
    print("\n MSE(loooop - cdnn):", nn.MSELoss()(res_loop, res_cdnn).item())


if __name__ == "__main__":
    main()
    compare()
