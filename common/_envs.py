""" An object more closer to the Arcade Learning Environment that the one
provided by OpenAI Gym.
"""
from collections import deque

import gym
import numpy as np
import torch
from ale_py import ALEInterface, LoggerMode, roms
from gym.spaces import Box, Discrete
from rich.columns import Columns


try:
    import cv2
except ModuleNotFoundError as err:
    print(
        "OpenCV is required when using the ALE env wrapper. ",
        "Try `conda install -c conda-forge opencv`.",
    )

__all__ = ["MinAtar", "ALE"]


def _get_rom(game):
    try:
        rom = getattr(roms, game)
    except AttributeError:
        print(f"{len(roms.__all__)} available roms:", roms.__all__)
        raise
    return rom


class ALE:
    """ A wrapper over atari_py, the Arcade Learning Environment python
    bindings that follows the Dopamine protocol:
        - frame concatentation of `history_len=4`
        - maximum episode length of 108,000 frames
        - sticky action probability `sticky_action_p=0.25`
        - end game after only after all lives have been lost
        - clip rewards during training to (1, -1)
        - frame skipping of 4 frames
        - minimal action set

    All credits for this wrapper go to
    [@Kaixhin](https://github.com/Kaixhin/Rainbow/blob/master/env.py),
    except for the bugs, those go to me.

    Returns:
        env: An ALE object with settings simillar to Dopamine's environment.
    """

    # pylint: disable=too-many-arguments, bad-continuation
    def __init__(
        self,
        game,
        seed,
        device,
        clip_rewards_val=1,
        history_length=4,
        sticky_action_p=0.25,
        max_episode_length=108e3,
    ):
        # pylint: enable=bad-continuation
        self.game_name = game
        self.device = device
        self.sticky_action_p = sticky_action_p
        self.window = history_length
        self.clip_val = clip_rewards_val

        # configure ALE
        self.ale = ALEInterface()
        self.ale.setLoggerMode(LoggerMode.Error)
        self.ale.setInt("random_seed", seed)
        self.ale.setInt("max_num_frames_per_episode", int(max_episode_length))
        self.ale.setFloat("repeat_action_probability", self.sticky_action_p)
        self.ale.setInt("frame_skip", 1)  # we handle frame skipping in this wrapper
        self.ale.setBool("color_averaging", False)  # we use max pooling instead
        self.ale.loadROM(_get_rom(self.game_name))

        # buffer used for stacking frames
        self.state_buffer = deque([], maxlen=self.window)

        # configure action space
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.action_space = Discrete(len(self.actions))

    def _get_state(self):
        state = cv2.resize(
            self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_AREA,
        )
        return torch.tensor(state, dtype=torch.uint8, device=self.device)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(
                torch.zeros(84, 84, device=self.device, dtype=torch.uint8)
            )

    def reset(self):
        """ Reset the environment, return initial observation. """
        # reset internals
        self._reset_buffer()
        self.ale.reset_game()

        # process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0).unsqueeze(0).byte()

    def step(self, action):
        """ Advance the environment given the agent's action.

        Args:
            action (int): Agent's action.
        Returns:
            tuple: The environment's observation.
        """
        # repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device, dtype=torch.uint8)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)

        # clip the reward
        if self.clip_val:
            clipped_reward = max(min(reward, self.clip_val), -self.clip_val)
        else:
            clipped_reward = reward

        # return state, reward, done
        state = torch.stack(list(self.state_buffer), 0).unsqueeze(0).byte()
        return state, clipped_reward, done, {"true_reward": reward}

    def close(self):
        pass

    def __str__(self):
        """ User friendly representation of this class. """
        stochasticity = (
            f"{self.sticky_action_p:.2f}_sticky_action"
            if self.sticky_action_p
            else "deterministic"
        )
        return (
            "ALE(game={}, stochasticity={}, hist_len={}, repeat_act=4, clip_rewards={})"
        ).format(self.game_name, stochasticity, self.window, self.clip_val)