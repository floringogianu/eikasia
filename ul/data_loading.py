from functools import partial

import cv2
import torchvision.transforms as T
import webdataset as wds  #pylint: disable=import-error
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CelebA


# configure loaders


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
