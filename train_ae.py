"""Unsupervised perceptual compression on Atari, inspired by Stable Diffusion.
"""
from pathlib import Path

import torch
from liftoff import parse_opts
from torchvision.utils import save_image

import ul.models as models
from ul.data_loading import get_seq_loader


def sample_model(ldr, model, path, device, N=8):
    """Takes a DataLoader, samples N images and performs reconstruction and sampling
    from posterior.
    """
    xs = [next(iter(ldr)) for _ in range(N)]
    x = torch.cat([x[0] for x in xs], dim=0).squeeze().to(device)
    with torch.no_grad():
        x_, pzx, _ = model.forward(x)
        z = pzx.mode()
        xx = model.decoder(z)

    img = torch.cat([x, xx, x_])
    for i in range(4):
        save_image(
            img[:, i, :, :].unsqueeze(1),
            path / f"{model.step:08d}_ch{i}.png",
            nrow=N,
        )


def run(opt):
    """Liftoff entry point."""
    torch.multiprocessing.set_sharing_strategy("file_system")

    opt.device = device = torch.device(opt.device)

    # get model
    model = getattr(models, opt.model.name).from_opt(opt.model)
    model.to(device)
    print(model)

    # config dataloader
    trn_ldr, _ = get_seq_loader(opt)
    val_ldr, _ = get_seq_loader(opt, split="val")

    # make output dir
    samples_path = Path(opt.out_dir) / "samples"
    if not samples_path.is_dir():
        samples_path.mkdir()

    # train
    for _ in range(opt.epochs):
        for step, (x, _, _, _) in enumerate(trn_ldr):

            if x.shape[0] != opt.loader.batch_size:
                continue

            x = x.squeeze().to(device)
            model.train(x)

            if step % 5000 == 0:
                sample_model(
                    val_ldr,
                    model,
                    samples_path,
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
