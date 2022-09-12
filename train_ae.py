"""Unsupervised perceptual compression on Atari, inspired by Stable Diffusion.
"""
from pathlib import Path

import torch
from liftoff import parse_opts
from torchvision.utils import save_image

from ul.models import ObservationRewardModel
from ul.autoencoder import AutoEncoderKL
from ul.data_loading import get_dset, get_loader


def sample_model(dsets, model, path, device, N=8):
    """Takes some datasets, samples N images and performs reconstruction and sampling
    from posterior.
    """
    if not isinstance(dsets, list):
        dsets = [dsets]
    rows = []
    for dset in dsets:
        try:
            samples = [dset[i][0] for i in torch.randperm(len(dset))[:N]]
        except TypeError:
            iter_dset = iter(dset)
            samples = [next(iter_dset) for _ in range(N)]
            samples = [x[0] for x in samples]

        x = torch.stack(samples)
        x = x.to(device)
        with torch.no_grad():
            x_, pzx, _ = model.forward(x)
            z = pzx.mode()
            xx = model.decoder(z)
            rows += [x, xx, x_]
    save_image(torch.cat(rows), path, nrow=N)


def run(opt):
    """Liftoff entry point."""
    torch.multiprocessing.set_sharing_strategy("file_system")

    opt.device = device = torch.device(opt.device)

    # get data
    trn_set = get_dset(opt.dset.name, split="trn", **opt.dset.args)
    val_set = get_dset(opt.dset.name, split="val", **opt.dset.args)

    # get model
    # model = AutoEncoderKL.from_opt(opt.model).to(device)
    model = ObservationRewardModel(opt).to(device)
    print(model)

    # config dataloader
    trn_ldr = get_loader(trn_set, **opt.loader.args)

    # train
    for _ in range(opt.epochs):
        for step, (x, _) in enumerate(trn_ldr):
            x = x.to(device)

            model.train(x)

            if step % 5000 == 0:
                samples_path = Path(opt.out_dir) / "samples"
                if not samples_path.is_dir():
                    samples_path.mkdir()

                sample_model(
                    [trn_set, val_set],
                    model,
                    f"{samples_path.as_posix()}/{model.step:08d}.png",
                    device,
                )

            if step % 50_000 == 0 and step != 0:
                torch.save(
                    {"model": model.state_dict()},
                    f"{opt.out_dir}/model_{model.step:08d}.pkl",
                )


def main():
    """Guard"""
    run(parse_opts())


if __name__ == "__main__":
    main()
