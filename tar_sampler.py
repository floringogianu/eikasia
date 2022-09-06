""" Sample ALE and dump in a tar.
"""
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rlog
import torch
import webdataset as wds
from liftoff import parse_opts

from src.agents.common import AtariNet, Policy, load_checkpoint, sample_episode
from src.envs.ale import ALEClassic, ALEModern


def tile_horizontaly(x):
    """Tile history in a single large image."""
    return torch.cat(torch.split(x.squeeze(), 1), dim=2).squeeze().numpy()


def compute_trajectory_stats(transitions):
    """Compute discounted and undiscounted, clipped and unclipped returns."""
    Gt, Gt_disc, Gt_clip, Gt_clip_disc = 0, 0, 0, 0
    Gts, Gts_disc, Gts_clip, Gts_clip_disc = [], [], [], []
    for i in reversed(range(len(transitions))):
        Gt = transitions[i][2] + Gt
        reward = transitions[i][2]
        Gt_disc = reward + 0.99 * Gt_disc
        Gt_clip = np.clip(reward, -1, 1) + Gt_clip
        Gt_clip_disc = np.clip(reward, -1, 1) + 0.99 * Gt_clip_disc
        Gts.insert(0, Gt)
        Gts_disc.insert(0, Gt_disc)
        Gts_clip.insert(0, Gt_clip)
        Gts_clip_disc.insert(0, Gt_clip_disc)
    return {
        "Gts": Gts,
        "Gts_disc": Gts_disc,
        "Gts_clip": Gts_clip,
        "Gts_clip_disc": Gts_clip_disc,
    }


def run(opt):
    """Entry path."""
    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True, relative_time=True)
    rlog.addMetrics(
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
        rlog.SumMetric("val_ep_cnt", metargs=["done"]),
        rlog.AvgMetric("val_avg_step", metargs=[1, "done"]),
        rlog.FPSMetric("val_fps", metargs=["val_frames"]),
    )

    device = torch.device("cpu")

    # configure env
    env = ALEModern(
        opt.game,
        torch.randint(100_000, (1,)).item(),
        sdl=False,
        device=device,
        clip_rewards_val=False,
    )

    # configure model
    model = AtariNet(env.action_space.n)
    model.eval()

    # sanity check
    rlog.info(opt)
    rlog.info("\n\n".join([str(o) for o in [env, model]]))

    # get the checkpoint paths
    opt.src_dir = f"{opt.src_dir}/{opt.game}/{opt.run_id}/"
    checkpoint_paths = list(sorted(Path(opt.src_dir).glob("model*")))
    checkpoint_paths = checkpoint_paths[::5]
    rlog.info(f"Found {len(checkpoint_paths)} models.")

    for _, ckpt_path in enumerate(checkpoint_paths):
        # load checkpoints
        ckpt = load_checkpoint(ckpt_path, device=device)
        ckpt_step = ckpt["step"]
        rlog.info(f"Loaded model @ step {ckpt_step}")

        # set model
        model.load_state_dict(ckpt["estimator_state"])
        model.to(device)
        policy = Policy(model=model, epsilon=opt.val_epsilon)

        # make paths
        dset_root = f"{opt.dset_root}/{opt.game}/{opt.run_id}/"
        os.makedirs(os.path.dirname(dset_root), exist_ok=True)

        # start sampling
        sample_cnt, ep_cnt = 0, 0

        # one archive per checkpoint
        sink = wds.TarWriter(f"{dset_root}/{ckpt_step:08d}.tar")
        summary_path = f"{dset_root}/summary.parquet"

        while sample_cnt <= opt.max_samples:

            # sample episode
            transitions = sample_episode(policy, env)
            extra = compute_trajectory_stats(transitions)
            state_keys = []

            for step, ((_, rgb), pi, reward, done) in enumerate(transitions):
                skey = "{}_s:{}_c:{:08d}_sid:{:06d}".format(
                    opt.game, opt.run_id, ckpt_step, sample_cnt
                )
                payload = {
                    "__key__": skey,
                    "state.png": rgb,
                    "ard.msg": {
                        "action": pi.action,
                        "reward": reward,
                        "done": done,
                    },
                    "extra.msg": {
                        "step": step,
                        "episode": ep_cnt,
                        **{k: float(v[step]) for k, v in extra.items()},
                    },
                }
                # store for header
                state_keys.append(skey)

                # write to disk
                sink.write(payload)
                sample_cnt += 1
            ep_cnt += 1

            # also save in columnar form
            # first augment the dict with some more info
            extra["state"] = state_keys
            extra["action"] = [t[1].action for t in transitions]
            extra["reward"] = [t[2] for t in transitions]
            extra["done"] = [t[3] for t in transitions]

            # make table and write
            table = pa.Table.from_pydict(extra)

            # for the first chunk of records
            if not Path(summary_path).exists():
                # create a parquet write object giving it an output file
                pqwriter = pq.ParquetWriter(summary_path, table.schema)
            pqwriter.write_table(table)

        # log
        rlog.traceAndLog(ckpt_step)

        # close open files
        sink.close()

    pqwriter.close()
    rlog.info("Aaand, done!")


def main():
    run(parse_opts())


if __name__ == "__main__":
    main()
