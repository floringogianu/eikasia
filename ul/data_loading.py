import random
from collections import deque
from functools import partial

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
    We do this by checking the idxs are consistent.
    """
    diff = int(seq[-1].split(":")[-1]) - int(seq[0].split(":")[-1])
    return diff == (seq_steps - 1)


def _preprocess(tuple_of_lists):
    # arrange the data in tensors the size of sequence lenght
    state_seq = (
        torch.from_numpy(np.stack(tuple_of_lists[0], axis=0)).unsqueeze_(1).contiguous()
    )
    # actions, rewards and done are in a dictionary at this point
    action_seq, reward_seq, done_seq = list(
        zip(*[el.values() for el in tuple_of_lists[1]])
    )
    # reward_seq = torch.tensor(reward_seq, dtype=torch.float32)
    action_seq = torch.tensor(action_seq, dtype=torch.int64)
    return state_seq, action_seq


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
    for i, d in enumerate(data):
        list_of_tuples.append(d)

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
        if d[1]["done"]:
            state_seq, action_seq = _preprocess(tuple_of_lists)
            # and flush the deque so that we never sample sequences
            # containing "done" elsewhere than at the end of the sequence
            list_of_tuples.clear()
            # also signal we just returned a "terminal sequence"
            was_done = True
            yield (state_seq, action_seq)

        # if we just returned the "terminal sequence" wait for the deque to fill
        # and return the first valid sequence of the episode -- "starting sequence"
        # this mirrors the previous condition and unbiases the sampling.
        elif was_done:
            state_seq, action_seq = _preprocess(tuple_of_lists)
            # also switch back the flag, return to normal operation
            was_done = False
            yield (state_seq, action_seq)

        # we want to avoid overfitting so we only sample every other
        # subsample / seq_steps
        elif subsample and (random.random() > (subsample / seq_steps)):
            continue

        # normal operation
        else:
            state_seq, action_seq = _preprocess(tuple_of_lists)
            yield (state_seq, action_seq)


def get_seq_loader(opt):
    prep = T.Lambda(
        lambda x: cv2.resize(
            cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (96, 96), interpolation=cv2.INTER_AREA
        )
    )

    # we use a sliding window to get sequences of total length given by
    # the the training length + the warm-up lenght required by the RNN
    dyn_args = opt.model.dynamics_net.args
    sequencer = partial(
        sliding_window,
        seq_steps=dyn_args["seq_steps"] + dyn_args["pfx_steps"],
        subsample=opt.dset.subsample,
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
