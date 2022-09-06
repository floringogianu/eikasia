""" Arcade Learning Environment wrappers for the classic and modern training protocols.

    All credits go to the original creator of this wrapper
    [@Kaixhin](https://github.com/Kaixhin/Rainbow/blob/master/env.py),
    except for the bugs, those go to me.
"""
from collections import deque

import numpy as np
import torch
from ale_py import ALEInterface, LoggerMode, roms
from gym.spaces import Box, Discrete

try:
    import cv2
except ModuleNotFoundError as err:
    print(
        "\nOpenCV is required when using the ALE env wrapper. ",
        "Try `conda install -c conda-forge opencv`.\n",
    )

__all__ = ["ALEModern", "ALEClassic"]


def _print_cols(arr, ncol=3):
    rows = [arr[offs : offs + ncol] for offs in range(0, len(arr), ncol)]
    for row in rows:
        tmplt = "{:<20}" * len(row)
        print(tmplt.format(*row))


def _get_rom(game):
    try:
        rom = getattr(roms, game)
    except AttributeError:
        print(f"{len(roms.__all__)} available roms:")
        _print_cols(roms.__all__)
        raise
    return rom


class ALE:
    # pylint: disable=too-many-arguments, bad-continuation
    def __init__(
        self,
        game,
        seed,
        device,
        clip_rewards_val=1,
        history_length=1,
        sticky_action_p=0.25,
        max_episode_length=108e3,
        sdl=False,
        mode=None,
        difficulty=None,
        minimal_action_set=True,
        obs_mode="RGB",
        obs_dims=(96, 96),
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
        if sdl:
            self.ale.setBool("sound", True)
            self.ale.setBool("display_screen", True)
        self.ale.loadROM(_get_rom(self.game_name))

        # set mode and difficulty
        self._set_mode(mode)
        self._set_difficulty(difficulty)

        # buffer used for stacking frames
        self.state_buffer = deque([], maxlen=self.window)
        # max pooling of last two rgb frames. Set in self.step
        self.rgb_observation = None
        # set the observation mode
        self.getScreen = (
            self.ale.getScreenGrayscale if obs_mode == "L" else self.ale.getScreenRGB
        )
        self.ch = 1 if obs_mode == "L" else 3
        self.obs_dims = obs_dims

        # configure action space
        actions = (
            self.ale.getMinimalActionSet()
            if minimal_action_set
            else self.ale.getLegalActionSet()
        )
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.action_space = Discrete(len(self.actions))

    def get_rgb(self):
        return self.rgb_observation

    def _get_state(self):
        state = cv2.resize(
            self.getScreen(),
            self.obs_dims,
            interpolation=cv2.INTER_AREA,
        )
        obs = torch.tensor(state, dtype=torch.uint8, device=self.device)

        if obs.ndim == 3:
            return obs.permute(2, 0, 1)
        return obs.unsqueeze(0)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(
                torch.zeros(
                    self.ch, *self.obs_dims, device=self.device, dtype=torch.uint8
                )
            )

    def reset(self):
        """Reset the environment, return initial observation."""
        # reset internals
        self._reset_buffer()
        self.ale.reset_game()

        # set the rgb initial observation
        self.rgb_observation = self.ale.getScreenRGB().copy()
        # process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0).unsqueeze(0).byte()

    def step(self, action):
        """Advance the environment given the agent's action.

        Args:
            action (int): Agent's action.
        Returns:
            tuple: The environment's observation.
        """
        # repeat action 4 times, max pool over last 2 frames.
        # we do this for both the full-sized RGB observation and the 84x84 BW
        frame_buffer = torch.zeros(
            2, self.ch, *self.obs_dims, device=self.device, dtype=torch.uint8
        )
        rgb_buffer = np.zeros((2, *self.ale.getScreenDims(), 3), dtype=np.uint8)

        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
                rgb_buffer[0] = self.ale.getScreenRGB()
            elif t == 3:
                frame_buffer[1] = self._get_state()
                rgb_buffer[1] = self.ale.getScreenRGB()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # rgb
        self.rgb_observation = rgb_buffer.max(0)

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

    def _set_mode(self, mode):
        if mode is not None:
            available_modes = self.ale.getAvailableModes()
            assert mode in available_modes, f"mode not in {available_modes}"
            self.ale.setMode(mode)

    def _set_difficulty(self, difficulty):
        if difficulty is not None:
            available_difficulties = self.ale.getAvailableDifficulties()
            assert (
                difficulty in available_difficulties
            ), f"difficulty not in {available_difficulties}"
            self.ale.setDifficulty(difficulty)

    def set_mode_interactive(self):
        # set modes and difficultes
        print("Available modes:        ", self.ale.getAvailableModes())
        print("Available difficulties: ", self.ale.getAvailableDifficulties())
        self._set_mode(int(input("Select mode: ")))
        self._set_difficulty(int(input("Select difficulty: ")))
        self.ale.reset_game()

    def __str__(self):
        """User friendly representation of this class."""
        stochasticity = (
            f"{self.sticky_action_p:.2f}_sticky_action"
            if self.sticky_action_p
            else "deterministic"
        )
        return (
            "ALE(game={}, stochasticity={}, hist_len={}, repeat_act=4, clip_rewards={})"
        ).format(self.game_name, stochasticity, self.window, self.clip_val)


class ALEModern(ALE):
    """A wrapper over atari_py, the Arcade Learning Environment python
    bindings that follows the Dopamine protocol, which in turn, follows (Machado, 2017):
        - frame concatentation of `history_len=4`
        - maximum episode length of 108,000 frames
        - sticky action probability `sticky_action_p=0.25`
        - end game after only after all lives have been lost
        - clip rewards during training to (1, -1)
        - frame skipping of 4 frames
        - minimal action set

    Returns:
        env: An ALE object with settings simillar to Dopamine's environment.
    """

    def __init__(self, game, seed, device, **kwargs):
        super().__init__(
            game,
            seed,
            device,
            history_length=4,
            obs_mode="L",
            obs_dims=(84, 84),
            **kwargs,
        )
