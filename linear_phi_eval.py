"""Simple Linear Probe on the empirical return."""
import gc
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import webdataset as wds
from tqdm import tqdm

from common import io_utils as ioutil
from train_byol import ImpalaEncoder


prep = T.Lambda(
    lambda x: cv2.resize(
        cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (96, 96), interpolation=cv2.INTER_AREA
    )
)


def get_dset(path):
    dset = (
        wds.WebDataset(path)
        .decode("rgb8")
        .to_tuple("state.png", "ard.msg", "extra.msg", "__key__")
        .map_tuple(prep, None, None, None)
    )
    obs, Gt_clip_disc, Gt, dones = [], [], [], []
    for o, ard, xtr, _ in tqdm(dset):
        obs.append(o)
        Gt_clip_disc.append(xtr["Gts_clip_disc"])
        Gt.append(xtr["Gts"])
        dones.append(int(ard["done"]))
        # if ard["done"]:
        #     print(i, len(obs))
    return torch.utils.data.TensorDataset(
        torch.from_numpy(np.stack(obs, axis=0)),
        torch.tensor(Gt_clip_disc),
        torch.tensor(Gt),
        torch.tensor(dones, dtype=torch.uint8),
    )


class SamplingWorldModel(nn.Module):
    def __init__(self, encoder, N=512, M=256, max_seq_size=2000) -> None:
        super().__init__()
        self.encoder = encoder
        self.cls_loop = nn.GRU(N, M)
        self.M = M
        self.N = N
        self.max_seq_size = max_seq_size

    def forward(self, x):
        T, B, C, W, H = x.shape
        # sometimes T is very large so we chunk it in order to fit it in GPU RAM.
        z_chunks = []
        for _x in torch.split(x, self.max_seq_size):
            # collapse time and batch and back again
            z_chunks.append(
                self.encoder(_x.view(_x.shape[0] * B, C, W, H))
                .detach()
                .view(_x.shape[0], B, self.N)
            )
        z = torch.cat(z_chunks)
        # unroll
        out, _ = self.cls_loop(z)
        return out.view(T * B, self.M)

    def from_checkpoint(self, path):
        keys = self.state_dict().keys()
        state = torch.load(path)["model"]
        state = {".".join(k.split(".")[1:]): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if k in keys}
        self.load_state_dict(state)
        return self


class SamplingEncoder(ImpalaEncoder):
    def __init__(self, inp_ch, grp_norm=1, stack_sz=2, max_seq_size=2000) -> None:
        super().__init__(inp_ch, grp_norm, stack_sz)
        self.max_seq_size = max_seq_size

    def forward(self, x):
        T, B, C, W, H = x.shape
        # sometimes T is very large so we chunk it in order to fit it in GPU RAM.
        z_chunks = []
        for _x in torch.split(x, self.max_seq_size):
            # collapse time and batch and back again
            z_chunks.append(super().forward(_x.view(_x.shape[0] * B, C, W, H)).detach())
        return torch.cat(z_chunks).view(T * B, 512)

    def from_checkpoint(self, path):
        keys = self.state_dict().keys()
        state = torch.load(path)["model"]
        state = {".".join(k.split(".")[2:]): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if k in keys}
        self.load_state_dict(state)
        return self


def cache_dset(dset):
    for t, k in zip(dset.tensors, ["obs", "Gt_clip_disc", "Gt", "done"]):
        print(k, t.shape, t.dtype)
        torch.save(t, f"./data/linear_probe/{k}.pt")


class EpisodeDataset(torch.utils.data.Dataset):
    """Split tensors accordingly to a "done episode" mask,
    by convention the last in the list.
    """

    def __init__(self, *tensors) -> None:
        super().__init__()
        # done signal is in the last tensor
        idxs = ((tensors[-1] == 1).nonzero().squeeze() + 1)[:-1]
        self.data = list(zip(*[torch.tensor_split(t, idxs) for t in tensors]))

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


def from_cache(path):
    print(f"Loading dset from cache @ {path}")
    keys = ["obs", "Gt_clip_disc", "Gt", "done"]
    data = []
    for k in keys:
        shard_paths = [d["url"] for d in wds.SimpleShardList(f"{path}_{k}.pt")]
        data.append(torch.cat([torch.load(p) for p in shard_paths]))
    return EpisodeDataset(*data)


def featurize_dset(dset, net, device):
    """Warning, this changes the dataset."""
    net.eval()
    data = [[] for _ in range(4)]
    with torch.no_grad():
        for obs_seq, Gtcd_seq, Gt_seq, done_seq in tqdm(dset, total=len(dset)):
            if obs_seq.shape[0] > 13_000:
                continue
            # cast to float, normalize to [0,1] and add batch and channel dimensions
            obs_seq = obs_seq.float().div(255).unsqueeze(1).unsqueeze(1).to(device)
            z_seq = net(obs_seq).cpu()
            # append to results
            for i, seq in enumerate([z_seq, Gtcd_seq, Gt_seq, done_seq]):
                data[i].append(seq)

    data = [torch.cat(el) for el in data]
    return torch.utils.data.TensorDataset(*data)


def get_size(t):
    return (t.nelement() * t.element_size()) / 1_073_741_824


def main(opt):
    if opt.dev:
        print("Devel mode -------------------------------")
    root = Path(opt.root)
    cfg = ioutil.read_config(root / "cfg.yaml", info=False)
    device = torch.device("cuda")

    # load from tarfile and cache tensors for faster loading
    # dset = get_dset(
    #     "./data/MDQN_rgb/Breakout/{0,1,2}/{25750000,33000000,41750000,50000000}.tar"
    # )
    # cache_dset(dset)

    # load prev results if they exist
    res_path = root / f"{opt.fname}_{opt.model}.pt"
    results = torch.load(res_path) if res_path.is_file() and not opt.dev else []
    print("Reading and writing from: ", res_path.stem)

    # load checkpoint paths
    paths = sorted(root.glob("model_*"))
    if results:
        paths = [p for p in paths if int(p.stem.split("_")[-1]) > results[-1][0]]
    print(
        "Found {} models. Starting from: {}".format(
            len(paths), paths[0].stem.split("_")[-1]
        )
    )

    # load cached data
    trn_dset = from_cache("./data/linear_probe/trn_{00..01}")
    val_dset = from_cache("./data/linear_probe/val_00")

    for path in paths:
        step = int(path.stem.split("_")[-1])

        # model
        if opt.model == "SamplingWorldModel":
            enc = ImpalaEncoder(**cfg.encoder.args)
            wm = SamplingWorldModel(enc).from_checkpoint(path)
            z_dim = 256
        elif opt.model == "SamplingEncoder":
            wm = SamplingEncoder(**cfg.encoder.args).from_checkpoint(path)
            z_dim = 512
        else:
            raise ValueError("Model not understood.")
        wm = wm.to(device)

        # features
        trn_phi_dset = featurize_dset(trn_dset, wm, device)
        val_phi_dset = featurize_dset(val_dset, wm, device)

        # is this thing working?
        gc.collect()

        # linear probe
        lin = nn.Linear(z_dim, 1).to(device)
        trn_inp, trn_tgt = [t.to(device) for t in trn_phi_dset.tensors[:2]]
        val_inp, val_tgt = [t.to(device) for t in val_phi_dset.tensors[:2]]
        optim = torch.optim.LBFGS(
            lin.parameters(), history_size=10, max_iter=4, line_search_fn="strong_wolfe"
        )

        # optimize
        trn_losses, val_losses = [], []
        for _ in range(1000):

            def closure():
                optim.zero_grad()
                loss = nn.MSELoss()(lin(trn_inp), trn_tgt.unsqueeze(1))
                loss.backward()
                return loss

            trn_losses.append(optim.step(closure).detach().item())
            with torch.no_grad():
                val_losses.append(nn.MSELoss()(lin(val_inp), val_tgt.unsqueeze(1)))

        # log
        results.append(
            (
                step,
                torch.tensor(trn_losses).min().item(),
                torch.tensor(val_losses).min().item(),
            )
        )
        print("{:06d}. train: {:7.4f}  |  valid: {:7.4f}".format(*results[-1]))

    if not opt.dev:
        torch.save(results, res_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "root", type=str, help="path to the experiment containing models."
    )
    parser.add_argument("-x", "--dev", dest="dev", action="store_true", help="dev mode")
    parser.add_argument(
        "-f",
        "--fname",
        dest="fname",
        type=str,
        default="linear_probe",
        help="default name of the results file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        default="SamplingWorldModel",
        help="name of the feature extractor.",
    )

    # parser.add_argument("game", type=str, help="game name")
    # parser.add_argument(
    #     "-e", "--episodes", default=10, type=int, help="number of episodes"
    # )
    # parser.add_argument(
    #     "-v",
    #     "--variations",
    #     action="store_true",
    #     help="set mode and difficulty, interactively",
    # )
    # parser.add_argument(
    #     "-r", "--record", action="store_true", help="record png screens and sound",
    # )
    main(parser.parse_args())
