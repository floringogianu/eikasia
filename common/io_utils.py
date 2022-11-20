""" I/O utils.
"""
import os
import shutil
import socket
import subprocess
from argparse import Namespace
from datetime import datetime
from gzip import GzipFile
from pathlib import Path

import gpustat
import psutil
import rlog
import torch
import yaml
from cpuinfo import get_cpu_info
from termcolor import colored as clr


def config_to_string(
    cfg: Namespace, indent: int = 0, color: bool = True, verbose: bool = False
) -> str:
    """Creates a multi-line string with the contents of @cfg."""

    text = ""
    for key, value in cfg.__dict__.items():
        if key.startswith("__") and not verbose:
            # censor some fields
            pass
        else:
            ckey = clr(key, "yellow", attrs=["bold"]) if color else key
            text += " " * indent + ckey + ": "
            if isinstance(value, Namespace):
                text += "\n" + config_to_string(value, indent + 2, color=color)
            else:
                cvalue = clr(str(value), "white") if color else str(value)
                text += cvalue + "\n"
    return text


class YamlNamespace(Namespace):
    """PyLint will trigger `no-member` errors for Namespaces constructed
    from yaml files. I am using this inherited class to target an
    `ignored-class` rule in `.pylintrc`.
    """


def create_paths(args: Namespace) -> Namespace:
    """Creates directories for containing experiment results."""
    time_stamp = "{:%Y%b%d-%H%M%S}".format(datetime.now())
    if hasattr(args, "resume") and args.resume:
        # if resuming the experiment, out_dir should be the same
        args.out_dir = args.resume
    if not hasattr(args, "out_dir") or args.out_dir is None:
        # if there's no out_dir create it
        if not os.path.isdir("./results"):
            os.mkdir("./results")
        out_dir = f"./results/{time_stamp}_{args.experiment:s}"
        os.mkdir(out_dir)
        args.out_dir = out_dir
    elif not os.path.isdir(args.out_dir):
        # finally if out_dir is given in some way but it's not on the disk
        # just shout, crash and burn.
        raise Exception(f"Directory {args.out_dir} does not exist.")

    if not hasattr(args, "run_id"):
        args.run_id = 0

    return args


def dict_to_namespace(dct: dict) -> Namespace:
    """Deep (recursive) transform from Namespace to dict"""
    namespace = YamlNamespace()
    for key, value in dct.items():
        name = key.rstrip("_")
        if isinstance(value, dict) and not key.endswith("_"):
            setattr(namespace, name, dict_to_namespace(value))
        else:
            setattr(namespace, name, value)
    return namespace


def namespace_to_dict(namespace: Namespace) -> dict:
    """Deep (recursive) transform from Namespace to dict"""
    dct: dict = {}
    for key, value in namespace.__dict__.items():
        if isinstance(value, Namespace):
            dct[key] = namespace_to_dict(value)
        else:
            dct[key] = value
    return dct


def flatten_dict(dct: dict, prev_key: str = None) -> dict:
    """Recursive flattening a dict"""
    flat_dct: dict = {}
    for key, value in dct.items():
        new_key = f"{prev_key}.{key}" if prev_key is not None else key
        if isinstance(value, dict):
            flat_dct.update(flatten_dict(value, prev_key=new_key))
        else:
            flat_dct[new_key] = value
    return flat_dct


def recursive_update(d: dict, u: dict) -> dict:
    "Recursively update `d` with stuff in `u`."
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _expand_from_keys(keys: list, value: object) -> dict:
    """Expand [a, b c] to {a: {b: {c: value}}}"""
    dct = d = {}
    while keys:
        key = keys.pop(0)
        d[key] = {} if keys else value
        d = d[key]
    return dct


def expand_dict(flat_dict: dict) -> dict:
    """Expand {a: va, b.c: vbc, b.d: vbd} to {a: va, b: {c: vbc, d: vbd}}.

    Opposite of `flatten_dict`.

    If not clear from above we want:
    {'lr':              0.0011,
     'gamma':           0.95,
     'dnd.size':        2000,
     'dnd.lr':          0.77,
     'dnd.sched.end':   0.0,
     'dnd.sched.steps': 1000
    }
    to this:
    {'lr': 0.0011,
     'gamma': 0.95,
     'dnd': {'size': 2000,
             'lr': 0.77,
             'sched': {'end': 0.0,
                       'steps': 1000
    }}}
    """
    exp_dict = {}
    for key, value in flat_dict.items():
        if "." in key:
            keys = key.split(".")
            key_ = keys.pop(0)
            if key_ not in exp_dict:
                exp_dict[key_] = _expand_from_keys(keys, value)
            else:
                exp_dict[key_] = recursive_update(
                    exp_dict[key_], _expand_from_keys(keys, value)
                )
        else:
            exp_dict[key] = value
    return exp_dict


def get_git_info() -> str:
    """Return sha@branch.
    This can maybe be used when restarting experiments. We can trgger a
    warning if the current code-base does not match the one we are trying
    to resume from.
    """
    cmds = [
        ["git", "rev-parse", "--short", "HEAD"],  # short commit sha
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # branch name
    ]
    res = []
    try:
        for cmd in cmds:
            res.append(subprocess.check_output(cmd).strip().decode("utf-8"))
    except subprocess.CalledProcessError:
        rlog.info("No repo or no commits.")
        return "no-git"
    return "@".join(res)


def get_hardware():
    """Return some hardware info...

    pip install -U gpustat
    pip install -U py-cpuinfo

    import gpustat
    from cpuinfo import get_cpu_info
    """
    cpus = get_cpu_info()
    cuda_var = "CUDA_VISIBLE_DEVICES"
    try:
        gpus = {
            "used": os.environ[cuda_var] if cuda_var in os.environ else "all",
            "devices": {str(i): v.entry for i, v in enumerate(gpustat.new_query())},
        }
        gpus_ = {"used": gpus["used"], "devices": {}}
        for gpuid, gpu in gpus["devices"].items():
            gpus_["devices"][gpuid] = {key: gpu[key] for key in ["name", "uuid"]}
    except:
        gpus = {}
        gpus_ = {}
    vram = psutil.virtual_memory()

    # shorter version for printing
    cpus_ = {
        key: cpus[key]
        for key in [
            "brand_raw",
            "count",
            "hz_advertised_friendly",
            "hz_actual_friendly",
        ]
    }
    return {
        "full": {"cpu": cpus, "gpu": gpus, "vram": str(vram)},
        "short": {"cpu": cpus_, "gpu": gpus_, "vram": vram.total},
    }


def add_platform_info(cfg):
    hw = get_hardware()
    if isinstance(cfg, dict):
        cfg["__hardware"] = hw["full"]
        cfg["hardware"] = hw["short"]
        cfg["git"] = get_git_info()
        cfg["hostname"] = socket.gethostname()
        cfg["omp_num_threads"] = os.getenv("OMP_NUM_THREADS")
    elif isinstance(cfg, (Namespace, YamlNamespace)):
        cfg.__hardware = dict_to_namespace(hw["full"])
        cfg.hardware = dict_to_namespace(hw["short"])
        cfg.git = get_git_info()
        cfg.hostname = socket.gethostname()
        cfg.omp_num_threads = os.getenv("OMP_NUM_THREADS")
    return cfg


def read_config(cfg_path, info=True):
    """Read a config file and return a namespace."""
    with open(cfg_path) as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)
    if info:
        config_data = add_platform_info(config_data)
    return dict_to_namespace(config_data)


def sanitize_dict(d):
    d_ = {}
    for k, v in d.items():
        if isinstance(v, dict):
            d_[k] = sanitize_dict(v)
        # ugly...
        elif not isinstance(v, (bool, int, float, str, list, tuple, dict)):
            d_[k] = str(v)
        else:
            d_[k] = v
    return d_


def save_config(cfg):
    """Save namespace or dict to disk."""
    if isinstance(cfg, (Namespace, YamlNamespace)):
        cfg = namespace_to_dict(cfg)
    elif isinstance(cfg, dict):
        pass
    else:
        raise TypeError(f"Don't know what to do with cfg of type {type(cfg)}.")

    # who knows what I'm storing in there so...
    cfg = sanitize_dict(cfg)

    with open(Path(cfg["out_dir"]) / "post_cfg.yml", "w") as outfile:
        yaml.safe_dump(cfg, outfile, default_flow_style=False)


def checkpoint_agent(path, crt_step, **kwargs):
    save_every_replay = kwargs.get("save_every_replay", False)

    # save checkpoint
    to_save = {"step": crt_step}
    replay_path = None
    for k, v in kwargs.items():
        if k == "replay" and v is not None:
            replay_path = v.save(path, crt_step, save_all=save_every_replay)
        elif isinstance(v, (torch.nn.Module, torch.optim.Optimizer)):
            to_save[f"{k}_state"] = v.state_dict()
        elif isinstance(v, (Namespace, YamlNamespace)):
            to_save[k] = namespace_to_dict(v)
        else:
            to_save[k] = v

    if replay_path is not None:
        # save checkpoints only when saving the replay
        with open(f"{path}/checkpoint.gz", "wb") as f:
            with GzipFile(fileobj=f) as outfile:
                torch.save(to_save, outfile)

        # when saving every replay the replay file name is replay_xxx.gz
        if save_every_replay:
            shutil.copyfile(replay_path, Path(path) / "replay.gz")

        # sometimes saving the replay fails and we end up with a bad `replay.gz`.
        # therefore we make sure we have at least one good copy of the previous replay.
        shutil.copyfile(replay_path, Path(path) / "prev_replay.gz")
        # same for the checkpoint
        shutil.copyfile(Path(path) / "checkpoint.gz", Path(path) / "prev_checkpoint.gz")

    # save every model
    with open(f"{path}/model_{crt_step:08d}.gz", "wb") as f:
        with GzipFile(fileobj=f, mode="w") as outfile:
            torch.save(
                {k: v for k, v in to_save.items() if k in ["step", "estimator_state"]},
                outfile,
            )

    rlog.info(
        "Saved the agent's state"
        f"{'' if replay_path is not None else ' NOT'} including the replay."
    )


def load_checkpoint(path, verbose=False, device=None):
    if verbose:
        print(f"Resuming from {path}.")
    with open(path, "rb") as f:
        with GzipFile(fileobj=f) as inflated:
            return torch.load(inflated, map_location=device)
