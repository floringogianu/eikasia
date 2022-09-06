from gzip import GzipFile
from pathlib import Path
from typing import NamedTuple

import rlog
import torch
import torch.nn as nn


class AtariNet(nn.Module):
    """Estimator used by DQN for ATARI games."""

    def __init__(self, action_no, distributional=False):
        super().__init__()

        self.action_no = out_size = action_no
        self.distributional = distributional

        # configure the support if distributional
        if distributional:
            support = torch.linspace(-10, 10, 51)
            self.__support = nn.Parameter(support, requires_grad=False)
            out_size = action_no * len(self.__support)

        # get the feature extractor and fully connected layers
        self.__features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.__head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_size),
        )

    def forward(self, x):
        """Define forward pass."""
        assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div(255)

        x = self.__features(x)
        qs = self.__head(x.view(x.size(0), -1))

        if self.distributional:
            logits = qs.view(qs.shape[0], self.action_no, len(self.__support))
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.__support.expand_as(qs_probs)).sum(2)
        return qs


class Episode:
    """An iterator accepting an environment and a policy, that returns
    experience tuples.
    """

    def __init__(self, env, policy, _state=None):
        self.env = env
        self.policy = policy
        self.__R = 0.0  # TODO: the return should also be passed with _state
        self.__step_cnt = -1
        if _state is None:
            self.__state, self.__done = self.env.reset(), False
        else:
            self.__state, self.__done = _state, False

    def __iter__(self):
        return self

    def __next__(self):
        if self.__done:
            raise StopIteration

        _pi = self.policy.act(self.__state)
        _state = self.__state.clone()
        self.__state, reward, self.__done, info = self.env.step(_pi.action)

        self.__R += reward
        self.__step_cnt += 1
        return (_state, _pi, reward, self.__state, self.__done, info)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print("Episode done")

    @property
    def Rt(self):
        """Return the expected return."""
        return self.__R

    @property
    def steps(self):
        """Return steps taken in the environment."""
        return self.__step_cnt


def load_checkpoint(fpath, device="cpu"):
    """Load a checkpoint."""
    fpath = Path(fpath)
    with fpath.open("rb") as file:
        with GzipFile(fileobj=file) as inflated:
            return torch.load(inflated, map_location=device)


class EpsilonGreedyOutput(NamedTuple):
    """The output of the epsilon greedy policy."""

    action: int
    q_value: float
    full: object


class Policy:
    """The policy."""

    def __init__(self, model, epsilon):
        self.model = model
        self.epsilon = epsilon

    def __call__(self, obs):
        return self.act(obs)

    def act(self, obs):
        """Receive an observation, take an action."""
        qvals = self.model(obs)
        if torch.rand((1,)).item() < self.epsilon:
            argmax_a = torch.randint(self.model.action_no, (1,))
            qsa = qvals.squeeze()[argmax_a.item()]
        else:
            qsa, argmax_a = qvals.max(1)
        return EpsilonGreedyOutput(
            action=argmax_a.item(), q_value=qsa.item(), full=qvals.flatten().numpy()
        )


def sample_episode(policy, env):
    """Sample one episode and return the full trajectory"""
    log = rlog.getRootLogger()
    episode, transitions = Episode(env, policy), []
    with torch.no_grad():
        for state, pi, reward, _, done, _ in episode:
            transitions.append([(state, env.get_rgb()), pi, reward, done])
            log.put(reward=reward, done=done, val_frames=1)
    return transitions
