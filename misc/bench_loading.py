import time
from collections import deque

import cv2
import numpy as np
import torch
import webdataset as wds
from torchvision import transforms as T


def sample_seq(data, seq_len=80, burn_in=40):
    seq_len = seq_len + burn_in
    # concate in a deque, then yield.
    list_of_tuples = deque(maxlen=seq_len)
    for d in data:
        list_of_tuples.append(d)
        tuple_of_lists = tuple(zip(*list_of_tuples))
        if len(list_of_tuples) == seq_len:
            state_seq = (
                torch.from_numpy(np.stack(tuple_of_lists[0], axis=0))
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            action_seq, reward_seq, done_seq = list(
                zip(*[el.values() for el in tuple_of_lists[1]])
            )
            action_seq = torch.tensor(action_seq, dtype=torch.int64)
            # reward_seq = torch.tensor(reward_seq, dtype=torch.float32)
            yield (
                state_seq,
                action_seq,
                # reward_seq,
                tuple_of_lists[2],
            )


def check_ids(a, b):
    print(a, b)
    a, b = [int(x.split(":")[-1]) for x in (a, b)]
    return b - a


def main():
    ckpts = "{00250000,33000000,44250000,48750000}"
    seeds = "1"
    path = f"./data/MDQN_rgb/Breakout/{seeds}/{ckpts}.tar"
    prep = T.Compose(
        [
            T.Lambda(lambda x: cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA)),
            # T.ToTensor(),
        ]
    )
    dset = (
        wds.WebDataset(path, shardshuffle=True)
        .decode("rgb8")
        .to_tuple("state.png", "ard.msg", "__key__")
        .map_tuple(prep)
        .compose(sample_seq)
        .shuffle(1000)
        .batched(32)
    )
    # dset_it = iter(dset)
    # s, a, k = next(dset_it)
    # print(s.shape, s.max())

    ldr = wds.WebLoader(dset, num_workers=8, pin_memory=False, batch_size=None)
    # ldr = ldr.unbatched()
    ldr = ldr.unbatched().shuffle(1000).batched(32)

    ldr_it = iter(ldr)
    s, a, k = next(ldr_it)
    print("done first batch", s.shape)
    print("done first batch", s.shape, check_ids(k[0][0], k[0][-1]))

    tot = 0
    s0 = time.time()
    deltas, start = [], time.time()
    for i in range(1, 301):
        s, a, k = next(ldr_it)
        # print(s.shape, a.shape, check_ids(k[0][0], k[0][-1]), len(k[0]))
        tot += s.shape[0]
        if i % 10 == 0:
            deltas.append(time.time() - start)
            start = time.time()
    print("Done:", time.time() - s0, tot)
    deltas = torch.tensor(deltas)
    print(f"mean={deltas.mean().item():2.3f}, std={deltas.std().item():2.3f}", deltas.shape)


if __name__ == "__main__":
    main()
