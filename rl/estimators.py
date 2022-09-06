""" Neural Network architecture for Atari games.
"""
from functools import partial
from itertools import chain
from math import sqrt

import torch
import torch.nn as nn
import rl.encoders as encoders


__all__ = ["AtariNet", "Encoder", "MLPQ", "MLPQ_C"]


def get_head(hidden_size, out_size, shared_bias=False):
    """Configures the default Atari output layers."""
    fc0 = nn.Linear(64 * 7 * 7, hidden_size)
    fc1 = (
        SharedBiasLinear(hidden_size, out_size)
        if shared_bias
        else nn.Linear(hidden_size, out_size)
    )
    return nn.Sequential(fc0, nn.ReLU(inplace=True), fc1)


def no_grad(module):
    """Callback for turning off the gradient of a module."""
    try:
        module.weight.requires_grad = False
    except AttributeError:
        pass


def variance_scaling_uniform_(tensor, scale=0.1, mode="fan_in"):
    r"""Variance Scaling, as in Keras.

    Uniform sampling from `[-a, a]` where:

        `a = sqrt(3 * scale / n)`

    and `n` is the number of neurons according to the `mode`.

    """
    # pylint: disable=protected-access,invalid-name
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    a = 3 * scale
    a /= fan_in if mode == "fan_in" else fan_out
    weights = nn.init._no_grad_uniform_(tensor, -a, a)
    # pylint: enable=protected-access,invalid-name
    return weights


class SharedBiasLinear(nn.Linear):
    """Applies a linear transformation to the incoming data: `y = xA^T + b`.
        As opposed to the default Linear layer it has a shared bias term.
        This is employed for example in Double-DQN.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    """

    def __init__(self, in_features, out_features):
        super(SharedBiasLinear, self).__init__(in_features, out_features, True)
        self.bias = nn.Parameter(torch.Tensor(1))

    def extra_repr(self):
        return "in_features={}, out_features={}, bias=shared".format(
            self.in_features, self.out_features
        )


def get_mlp(dims, activation_fn="ReLU"):
    """A generic MLP."""
    layers = []
    for i in range(len(dims) - 1):
        in_dim, out_dim = dims[i : i + 2]
        layers.append(nn.Linear(in_dim, out_dim, bias=True))
        if i != (len(dims) - 2):
            layers.append(getattr(nn, activation_fn)())
    return nn.Sequential(nn.Flatten(), *layers)


def _init(m, init_fn):
    if isinstance(m, (nn.Linear, CLinear, nn.Conv2d)):
        init_fn(m.weight)
        if hasattr(m, "bias"):
            m.bias.data.zero_()


INIT_FNS = {
    "xavier_uniform": partial(_init, init_fn=nn.init.xavier_uniform_),
    "variance_scaling_uniform": partial(
        _init, init_fn=partial(variance_scaling_uniform_, scale=1.0 / sqrt(3.0))
    ),
}


class Encoder(nn.Module):
    """A wrapper class for the various encoders we might use."""

    def __init__(self, name, encoder_kwargs, freeze=True) -> None:
        super(Encoder, self).__init__()

        self.encoder = getattr(encoders, name)(**encoder_kwargs)

        for param in self.encoder.parameters():
            param.requires_grad_(freeze)

    def forward(self, x):
        assert x.dtype == torch.uint8, "Expecting input of type uint8."
        assert x.ndim == 5, f"Expecting input of dimension B,T,C,H,W, not {x.shape}."

        x = x.view(x.shape[0], -1, *x.shape[-2:])  # collapse TxC
        x = x.float().div(255)
        return self.encoder(x)


# An MLP to be used in conjuntion with various encoders wrapped by the Encoder.


class MLPQ(nn.Module):
    """Estimator used for ATARI games. It uses a pretrained Encoder."""

    def __init__(  # pylint: disable=bad-continuation
        self,
        input_size,
        action_no,
        fc_layers=None,
        initializer="xavier_uniform",
        support=None,
        **kwargs,
    ):
        super(MLPQ, self).__init__()

        init_fns = list(INIT_FNS.keys())
        assert initializer in init_fns, f"Only implements {init_fns}."

        self.action_no = action_no

        # configure support if categorical
        self._support = None
        if support is not None:
            # handy to make it a Parameter so that model.to(device) works
            self._support = nn.Parameter(torch.linspace(*support), requires_grad=False)
            out_size = action_no * len(self._support)
        else:
            out_size = action_no

        self.head = get_mlp([input_size, *fc_layers, out_size])

        # reset the head
        self.head.apply(INIT_FNS[initializer])

    def forward(self, x, probs=False, log_probs=False):
        assert x.dtype == torch.float32, "Expecting input of type Float32."
        assert not (probs and log_probs), "Can't output both p(s, a) and log(p(s, a))"

        qs = self.head(x)

        # distributional RL
        # either return p(s,·), log(p(s,·)) or the distributional Q(s,·)
        if self._support is not None:
            logits = qs.view(qs.shape[0], self.action_no, len(self._support))
            if probs:
                return torch.softmax(logits, dim=2)
            if log_probs:
                return torch.log_softmax(logits, dim=2)
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.support.expand_as(qs_probs)).sum(2)
        # or just return Q(s,a)
        return qs

    @property
    def support(self):
        """Return the support of the Q-Value distribution."""
        return self._support


# Some experiments with aggregating channel-level information
# using linear layers.


class CLinear(nn.Module):
    def __init__(  # pylint: disable=bad-continuation
        self, inp_features, out_features, inp_channels=1
    ):
        super().__init__()
        self.inp_features = inp_features
        self.out_features = out_features
        self.inp_channels = inp_channels

        self.weight = nn.Parameter(
            torch.randn(inp_channels, inp_features, out_features)
        )

    def forward(self, x):
        return torch.einsum("btcm,cmn -> bcn", x, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, channels={}".format(
            self.inp_features, self.out_features, self.inp_channels
        )


class MLPQ_C(nn.Module):
    def __init__(  # pylint: disable=bad-continuation
        self,
        input_size,
        action_no,
        inp_ch=1,
        fc_layers=None,
        initializer="xavier_uniform",
        **kwargs,
    ):
        super().__init__()

        init_fns = list(INIT_FNS.keys())
        assert initializer in init_fns, f"Only implements {init_fns}."

        self.clin = CLinear(input_size, fc_layers[0], inp_ch)
        self.head = get_mlp([*fc_layers, action_no])

        # reset the head
        self.head.apply(INIT_FNS[initializer])
        self.clin.apply(INIT_FNS[initializer])

    def forward(self, x):
        assert x.dtype == torch.float32, "Expecting input of type Float32."
        assert x.ndim >= 4, f"Expecting input of dimension B,T,C,... not {x.shape}."

        x = x.flatten(3)  # B,T,C,...
        x = self.clin(x)
        x = x.max(1).values  # max pooling over the channels
        x = nn.functional.relu(x)
        return self.head(x)


# The classic Atari Net.


class AtariNet(nn.Module):
    """Estimator used for ATARI games."""

    def __init__(  # pylint: disable=bad-continuation
        self,
        action_no,
        input_ch=1,
        hist_len=4,
        hidden_size=256,
        shared_bias=False,
        initializer="xavier_uniform",
        support=None,
        spectral=None,
        **kwargs,
    ):
        super(AtariNet, self).__init__()

        assert initializer in (
            "xavier_uniform",
            "variance_scaling_uniform",
        ), "Only implements xavier_uniform and variance_scaling_uniform."

        self.__action_no = action_no
        self.__initializer = initializer
        self.__support = None
        self.spectral = spectral
        if support is not None:
            self.__support = nn.Parameter(
                torch.linspace(*support), requires_grad=False
            )  # handy to make it a Parameter so that model.to(device) works
            out_size = action_no * len(self.__support)
        else:
            out_size = action_no

        # get the feature extractor and fully connected layers
        self.__features = get_feature_extractor(hist_len * input_ch)
        self.__head = get_head(hidden_size, out_size, shared_bias)

        self.reset_parameters()

        # We allways compute spectral norm except when None or notrace
        if spectral is not None:
            self.__hooked_layers = hook_spectral_normalization(
                spectral,
                chain(self.__features.named_children(), self.__head.named_children()),
                **kwargs,
            )

    def forward(self, x, probs=False, log_probs=False):
        # assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div(255)
        assert not (probs and log_probs), "Can't output both p(s, a) and log(p(s, a))"

        x = self.__features(x)
        x = x.view(x.size(0), -1)
        qs = self.__head(x)

        # distributional RL
        # either return p(s,·), log(p(s,·)) or the distributional Q(s,·)
        if self.__support is not None:
            logits = qs.view(qs.shape[0], self.__action_no, len(self.support))
            if probs:
                return torch.softmax(logits, dim=2)
            if log_probs:
                return torch.log_softmax(logits, dim=2)
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.support.expand_as(qs_probs)).sum(2)
        # or just return Q(s,a)
        return qs

    @property
    def support(self):
        """Return the support of the Q-Value distribution."""
        return self.__support

    def get_spectral_norms(self):
        """Return the spectral norms of layers hooked on spectral norm."""
        return {
            str(idx): layer.weight_sigma.item() for idx, layer in self.__hooked_layers
        }

    def reset_parameters(self):
        """Weight init."""
        init_ = (
            nn.init.xavier_uniform_
            if self.__initializer == "xavier_uniform"
            else partial(variance_scaling_uniform_, scale=1.0 / sqrt(3.0))
        )

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                init_(module.weight)
                module.bias.data.zero_()

    @property
    def feature_extractor(self):
        """Return the feature extractor."""
        return self.__features

    @property
    def head(self):
        """Return the layers used as heads in Bootstrapped DQN."""
        return self.__head
