from gzip import GzipFile

import torch
import torch.nn as nn
from torchvision import transforms as T
from ul.nets import RNEncoder as _RNEncoder
from ul.nets import WMEncoder as _WMEncoder


__all__ = ["AtariEncoder", "WMEncoder", "AchlioptasEncoder"]


class AtariEncoder(nn.Module):
    def __init__(self, path, inp_ch=4) -> None:
        super().__init__()
        self.inp_ch = inp_ch
        self.conv0 = nn.Conv2d(inp_ch, 32, kernel_size=8, stride=4)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.lin0 = nn.Linear(7 * 7 * 64, 512)
        expand = T.Lambda(lambda x: x.expand(-1, 4, -1, -1))
        self.expand = expand if inp_ch == 4 else nn.Identity()

        self._load_weights(path)

    def forward(self, x):
        x = self.expand(x)
        for conv in [self.conv0, self.conv1, self.conv2]:
            x = nn.functional.relu(conv(x), inplace=True)
        return x

    def _load_weights(self, path):
        with open(path, "rb") as file:
            with GzipFile(fileobj=file) as inflated:
                state = torch.load(inflated)["estimator_state"]
        convs = {k: v for k, v in state.items() if "_AtariNet__features" in k}
        convs = {
            f"conv{i//2}.{k.split('.')[-1]}": v
            for i, (k, v) in enumerate(convs.items())
        }
        if self.inp_ch < 4:
            print(f"Attempting surgery. Selecting last {self.inp_ch} input channels.")
            convs["conv0.weight"] = convs["conv0.weight"][:, -self.inp_ch :]
        lin0 = list(state.items())[-4:-2]
        state = {
            **convs,
            # "lin0.weight": lin0[0][1],
            # "lin0.bias": lin0[1][1],
        }

        print(f"Encoder layers: {', '.join(state.keys())}.")
        self.load_state_dict(state)


class OnlineMinMaxNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.min = 0
        self.max = 0

    def forward(self, x):
        self.min = min(x.min(), self.min)
        self.max = max(x.max(), self.max)
        return (x - self.min).div(self.max - self.min)


def _match_keys(keys, state):
    """This assumes the keys of the current model are shorter..."""
    state_ = {}
    for key in keys:
        for key_, p_ in state.items():
            if key in key_:
                state_[key] = p_
    return state_


class WMEncoder(nn.Module):
    def __init__(self, path, inp_ch=3, z_ch=8) -> None:
        super().__init__()
        self.encoder = _WMEncoder(inp_ch, z_ch=z_ch)
        self._load_weights(path)

    def forward(self, x):
        return self.encoder(x)[:, :4]

    def _load_weights(self, path):
        state = torch.load(path)["model"]
        state = _match_keys(list(self.state_dict().keys()), state)
        print(f"Encoder layers: {', '.join(state.keys())}.")
        self.load_state_dict(state)


class RNEncoder(nn.Module):
    def __init__(self, path, inp_ch=3, z_ch=8) -> None:
        super().__init__()
        self.encoder = _RNEncoder(inp_ch, z_ch=z_ch)
        self._load_weights(path)

    def forward(self, x):
        return self.encoder(x)[:, :4]

    def _load_weights(self, path):
        state = torch.load(path)["model"]
        state = _match_keys(list(self.state_dict().keys()), state)
        print(f"Encoder layers: {', '.join(state.keys())}.")
        self.load_state_dict(state)


# Untrained encoders


def achlioptas_init_(x):
    prob = torch.tensor([1 / 6, 4 / 6, 1 / 6])
    vals = torch.tensor([3]).sqrt() * torch.tensor([1.0, 0.0, -1.0])
    idxs = torch.multinomial(prob, x.numel(), replacement=True)
    with torch.no_grad():
        x.copy_(vals[idxs].view_as(x))
    return x


class AchlioptasEncoder(nn.Module):
    def __init__(self, inp_features=7056, out_features=24, out_ch=1) -> None:
        super().__init__()
        self.inp_features = inp_features
        self.out_features = out_features
        self.out_channels = out_ch
        w = achlioptas_init_(torch.zeros(out_ch, inp_features, out_features**2))
        self.W = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.einsum("bm,cmn -> bcn", x, self.W).unsqueeze(1)
        return x.view(
            x.shape[0], self.out_channels, self.out_features, self.out_features
        )

    def extra_repr(self) -> str:
        return "in_features={}, out_features={} x {}, channels={}".format(
            self.inp_features, self.out_features, self.out_features, self.out_channels
        )
