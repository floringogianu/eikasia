""" Datasets and DataLoaders based on webdataset."""
import random
from collections import deque
from functools import partial
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import webdataset as wds  # pylint: disable=import-error
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CelebA


# configure sequence loaders


def is_valid(seq, seq_steps):
    """Checks the sequence is not crossing shard boundaries.
    We do this by checking the key.step(s) are consistent.
    """
    return (seq[-1].step - seq[0].step) == (seq_steps - 1)


def _preprocess(tuple_of_lists):
    # arrange the data in tensors the size of sequence lenght
    state_seq = (
        torch.from_numpy(np.stack(tuple_of_lists[0], axis=0)).unsqueeze_(1).contiguous()
    )
    # actions, rewards and done are in a dictionary at this point
    action_seq, reward_seq, _ = list(zip(*[el.values() for el in tuple_of_lists[1]]))
    action_seq = torch.tensor(action_seq, dtype=torch.int64)
    reward_seq = torch.tensor(reward_seq, dtype=torch.float32)
    # game index
    gid = torch.tensor(
        [[k.game, k.seed, k.ckpt, k.step] for k in tuple_of_lists[-1]],
        dtype=torch.int64,
    )
    return state_seq, action_seq, reward_seq, gid


def sliding_window(data, seq_steps, subsample):
    """Used by webdataset to compose sliding windows of samples.  A sample is
    usually a tuple (image, dict, __key__). The key can be used to ID the
    sample.

    Args:
        data (generator): sequence of samples from webdataset.
        seq_steps (int): length of the sequence used for training.

    Yields:
        tuple: Sequences of frames, actions, rewards, etc...
    """
    # add in a deque, then yield if conditions apply
    list_of_tuples = deque(maxlen=seq_steps)

    was_done = False
    for i, sample in enumerate(data):
        list_of_tuples.append(sample)

        # check if ready
        deq_is_ready = len(list_of_tuples) == seq_steps
        if not deq_is_ready:
            continue

        # check if valid
        tuple_of_lists = tuple(zip(*list_of_tuples))
        keys = tuple_of_lists[2]
        if not is_valid(keys, seq_steps):  # and the sequence is valid
            continue

        # immediately return the sequence containing the terminal transition.
        if sample[1]["done"]:
            # and flush the deque so that we never sample sequences
            # containing "done" elsewhere than at the end of the sequence
            list_of_tuples.clear()
            # also signal we just returned a "terminal sequence"
            was_done = True
            yield _preprocess(tuple_of_lists)

        # if we just returned the "terminal sequence" wait for the deque to fill
        # and return the first valid sequence of the episode -- "starting sequence"
        # this mirrors the previous condition and unbiases the sampling.
        elif was_done:
            # also switch back the flag, return to normal operation
            was_done = False
            yield _preprocess(tuple_of_lists)

        # we want to avoid overfitting so we only sample every other
        # subsample / seq_steps
        elif subsample and (random.random() > (subsample / seq_steps)):
            continue

        # normal operation
        else:
            yield _preprocess(tuple_of_lists)


def obs_prep(x):
    """Grayscale and downsample."""
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA)


# key preprocessing
SampleKey = NamedTuple("SampleKey", game=int, seed=int, ckpt=int, step=int)


GAMES = ["Asterix", "Breakout", "Enduro", "MsPacman", "Seaquest", "SpaceInvaders"]
I2G = dict(enumerate(GAMES))
G2I = {g: i for i, g in I2G.items()}


def key_prep(key):
    """Convert keys of the form "Asterix_s:2_c:00250000_sid:024797"
    to something frendlier and leaner.
    """
    parts = [x.split(":")[-1] for x in key.split("_")]
    return SampleKey(G2I[parts[0]], *[int(x) for x in parts[1:]])


def get_seq_loader(opt, split="trn"):
    """Returns a DataLoader for sequences of transitions."""
    # we use a sliding window to get sequences of total length given by
    # the the training length + the warm-up lenght required by the RNN
    try:
        seq_steps = opt.dset.args["seq_steps"]
    except AttributeError:
        dyn_args = opt.model.dynamics_net.args
        seq_steps = dyn_args["seq_steps"] + dyn_args["pfx_steps"]

    sequencer = partial(
        sliding_window,
        seq_steps=seq_steps,
        subsample=opt.dset.args["subsample"],
    )

    obs_prep = T.Compose(
        [
            T.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)),
            T.Lambda(lambda x: cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA)),
            T.ToTensor(),
        ]
    )

    dset = (
        wds.WebDataset(opt.dset.args[f"{split}_path"], shardshuffle=True)
        .decode("rgb8")
        .to_tuple("state.png", "ard.msg", "__key__")
        .map_tuple(obs_prep, None, key_prep)
        .compose(sequencer)
        .shuffle(opt.dset.args["shuffle"])
        .batched(opt.dset.args["batch_size"])
    )
    if split == "trn":
        ldr = wds.WebLoader(dset, **opt.loader.args)
        ldr = ldr.unbatched().shuffle(opt.loader.shuffle).batched(opt.loader.batch_size)
    else:
        ldr = wds.WebLoader(dset, batch_size=None, num_workers=6)
        ldr = ldr.unbatched().shuffle(1000).batched(1)
    return ldr, dset


# configure one sample loaders


def get_loader(dset, **kwargs):
    if isinstance(dset, wds.compat.WebDataset):
        shuffle, num_workers, batch_size = (
            kwargs.get(k) for k in ["shuffle", "num_workers", "batch_size"]
        )
        loader = wds.WebLoader(dset, num_workers=num_workers)
        loader = loader.shuffle(shuffle).batched(batch_size)
        return loader
    return DataLoader(dset, **kwargs)


# configure datasets


def get_dset(name, split="trn", **kwargs):
    assert split in ("trn", "val"), f"Unknown split {split}. Accepted: `trn`,`val`"
    path = kwargs.get(f"{split}_path")

    if name == "CelebA":
        return get_celeba(split, path)
    if name == "Atari":
        shuffle = kwargs["shuffle"]
        return get_atari(path, shuffle)


def get_celeba(split, path):
    prep = T.Compose([T.Resize(96), T.CenterCrop(96), T.ToTensor()])
    _CelebA = partial(CelebA, root=path, download=True, transform=prep)
    if split == "trn":
        return ConcatDataset([_CelebA(split=split) for split in ["train", "valid"]])
    elif split == "val":
        return _CelebA(split="test")


def get_atari(path, shuffle=1000):
    prep = T.Compose(
        [
            T.Lambda(lambda x: cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA)),
            T.ToTensor(),
        ]
    )
    return (
        wds.WebDataset(path, shardshuffle=True)
        .shuffle(shuffle)
        .decode("rgb8")
        .to_tuple("state.png", "ard.msg", "__key__")
        .map_tuple(prep)
    )


def _dev():
    sequencer = partial(sliding_window, seq_steps=4, subsample=0)
    obs_prep = T.Compose(
        [
            T.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)),
            T.Lambda(lambda x: cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA)),
            T.ToTensor(),
        ]
    )

    dset = (
        wds.WebDataset(
            "./data/6games_rnd_new/{asterix,breakout,enduro,mspacman,seaquest,spaceinvaders}_{0164..0165}.tar",
            shardshuffle=True,
            # resampled=True,
        )
        .decode("rgb8")
        .to_tuple("state.png", "ard.msg", "__key__")
        .map_tuple(obs_prep, None, key_prep)
        .compose(sequencer)
        .shuffle(8000)
    )

    # for i, (obs, act, rew, gid) in enumerate(dset):
    #     print(i, obs.shape, act.shape, rew.shape, gid.shape)
    #     print(gid.squeeze())
    #     if i == 100:
    #         break

    ldr = wds.WebLoader(dset, batch_size=None, num_workers=16)
    ldr = ldr.unbatched().shuffle(1000).batched(32)

    # for i, (obs_seq, act_seq, rew_seq, gid_seq) in enumerate(ldr):
    #     print(i, obs_seq.shape, act_seq.shape, rew_seq.shape, gid_seq.shape)
    #     print(gid_seq.squeeze())
    #     if i == 100:
    #         break

    return ldr, dset


if __name__ == "__main__":
    _dev()
