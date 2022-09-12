""" Entry point. """
from collections import deque
from pathlib import Path

import numpy as np
import rlog
import torch
import torch.optim as O
from liftoff import parse_opts

import common.io_utils as ioutil
from common.ale import ALE
from rl import estimators
from rl.agents import AGENTS
from rl.replay import ExperienceReplay
from rl.rl_routines import Episode, validate


class ShortMemory:
    def __init__(self, hist_len=1) -> None:
        self.mem = deque([], maxlen=hist_len)
        self.hist_len = hist_len
        self.obs_spec = None

    def push(self, x):
        if self.obs_spec is None:
            self.obs_spec = x.clone()
            self.reset()
        self.mem.append(x)

    def retrieve(self):
        return torch.stack(list(self.mem), 1)

    def push_retrieve(self, x):
        self.push(x)
        return self.retrieve()

    def reset(self):
        for _ in range(self.hist_len):
            self.mem.append(torch.zeros_like(self.obs_spec))


class CoolValAgent:
    def __init__(self, encoder, qval_fn, opt) -> None:
        self.encoder = encoder
        self.qval_fn = qval_fn
        self.policy_improvement = AGENTS[opt.agent.name]["policy_improvement"](
            qval_fn,
            opt.action_num,
            epsilon=opt.val_epsilon,
        )
        self.short_mem = ShortMemory(opt.agent.args["hist_len"])

    def act(self, obs):
        z = self.encoder(obs).detach()
        zz = self.short_mem.push_retrieve(z)
        pi = self.policy_improvement.act(zz)
        return pi

    def reset(self):
        self.short_mem.reset()


def _get_latent_dims(f, inp_dim):
    x = torch.ones(*inp_dim, device=list(f.parameters())[0].device, dtype=torch.uint8)
    z = f(x).detach()
    return z.shape[1:]


class CoolAgent:
    def __init__(
        self,
        encoder,
        qval_fn,
        replay,
        short_mem,
        policy_improvement,
        policy_evaluation,
        update_freq,
        target_update_freq,
    ) -> None:
        self.encoder = encoder
        self.qval_fn = qval_fn
        self.replay = replay
        self.policy_improvement = policy_improvement
        self.policy_evaluation = policy_evaluation
        self.short_mem = short_mem
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq

        # state
        self._transition = []
        self.total_steps = 0

    @staticmethod
    def _set_ckpt_path(opt):
        if "root" in opt.estimator.encoder.args:
            root = opt.estimator.encoder.args.pop("root")
            ckpt_idx = opt.estimator.encoder.args.pop("ckpt_idx")
            opt.estimator.encoder.args["path"] = f"{root}/model_{ckpt_idx}.pkl"
        return opt

    @classmethod
    def init_from_opts(cls, opt):

        opt = cls._set_ckpt_path(opt)
        encoder = estimators.Encoder(
            opt.estimator.encoder.name,
            opt.estimator.encoder.args,
            opt.estimator.encoder.freeze,
        )

        # infer the expected input size of the qvalue network
        inp_ch = 3 if opt.env.args["obs_mode"] == "RGB" else 1
        z_dims = _get_latent_dims(encoder, (1, 1, inp_ch, *opt.env.args["obs_dims"]))
        hist_len = opt.agent.args["hist_len"]
        print(z_dims, hist_len)
        z_size = hist_len * np.prod(z_dims)
        qval_fn = getattr(estimators, opt.estimator.qval_net.name)(
            z_size, opt.action_num, **opt.estimator.qval_net.args
        )

        encoder.to(opt.device)
        qval_fn.to(opt.device)

        # replay
        if isinstance(opt.agent.args["epsilon"], list):
            opt.replay["warmup_steps"] = opt.agent.args["epsilon"][-1]
            opt.replay["hist_len"] = hist_len

        return cls(
            encoder,
            qval_fn,
            ExperienceReplay(**opt.replay),
            ShortMemory(hist_len),
            AGENTS[opt.agent.name]["policy_improvement"](
                qval_fn, opt.action_num, **opt.agent.args
            ),
            AGENTS[opt.agent.name]["policy_evaluation"](
                qval_fn,
                getattr(O, opt.optim.name)(qval_fn.parameters(), **opt.optim.args),
                **opt.agent.args,
            ),
            opt.agent.args["update_freq"],
            opt.agent.args["target_update_freq"],
        )

    def act(self, obs):
        z = self.encoder(obs).detach()
        zz = self.short_mem.push_retrieve(z)
        pi = self.policy_improvement.act(zz)

        if len(self._transition) == 4:
            # self._transition contents: [prev_z, action, reward, done]
            self.replay.push([*self._transition[:-1], zz, self._transition[-1]])

        # reset self._transition
        del self._transition[:]
        self._transition = [zz, pi.action]

        return pi

    def learn(self, transition):
        """Here we orchestrate learning across the different components of the agent."""
        self.total_steps += 1
        self.policy_evaluation.estimator.train()
        self.policy_evaluation.target_estimator.train()

        _, _, reward, _, done, _ = transition
        # append to the transition tuple
        self._transition += [reward, done]

        if done:
            self.short_mem.reset()

        # learn if a minimum no of transitions have been pushed in Replay
        if self.replay.is_ready:
            if self.total_steps % self.update_freq == 0:
                # sample from replay and do a policy evaluation step
                batch = self.replay.sample()

                # compute the loss and optimize
                loss = self.policy_evaluation(batch)

                # stats
                rlog.put(
                    trn_loss=loss.loss.detach().mean().item(),
                    lrn_steps=batch[0].shape[0],
                )
                if hasattr(loss, "entropy"):
                    rlog.put(trn_entropy=loss.entropy.detach().item())

            if self.total_steps % self.target_update_freq == 0:
                self.policy_evaluation.update_target_estimator()

        # some more stats
        rlog.put(trn_reward=reward, trn_done=done, trn_steps=1)
        if self.total_steps % 50_000 == 0:
            msg = "[{0:6d}] R/ep={trn_R_ep:2.2f}, tps={trn_tps:2.2f}"
            rlog.info(msg.format(self.total_steps, **rlog.summarize()))


def train_one_epoch(
    env,
    agent,
    epoch_step_cnt,
    total_steps=0,
    last_state=None,
):
    """Policy iteration for a given number of steps."""

    while True:
        # do policy improvement steps for the length of an episode
        # if _state is not None then the environment resumes from where
        # this function returned.
        for transition in Episode(env, agent, _state=last_state):

            agent.learn(transition)
            total_steps += 1

            # exit if done
            if total_steps % epoch_step_cnt == 0:
                _, _, _, state_, _, _ = transition
                return total_steps, state_

        # Do not attempt to resume episode if it finished.
        last_state = None


def get_env(opt, clip_rewards_val=1):
    return ALE(
        opt.env.name,
        np.random.randint(1_000_000),
        opt.device,
        clip_rewards_val=clip_rewards_val,
        **opt.env.args,
    )


def run(opt):
    """Entry point of the program."""

    if __debug__:
        print("Code might have assertions. Use -O in liftoff.")

    ioutil.create_paths(opt)

    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True, relative_time=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_R_ep", metargs=["trn_reward", "trn_done"]),
        rlog.SumMetric("trn_ep_cnt", metargs=["trn_done"]),
        rlog.AvgMetric("trn_loss", metargs=["trn_loss", 1]),
        rlog.FPSMetric("trn_tps", metargs=["trn_steps"]),
        rlog.FPSMetric("lrn_tps", metargs=["lrn_steps"]),
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
        rlog.SumMetric("val_ep_cnt", metargs=["done"]),
        rlog.AvgMetric("val_avg_step", metargs=[1, "done"]),
        rlog.FPSMetric("val_fps", metargs=["val_frames"]),
    )
    if opt.agent.name == "M-DQN":
        rlog.addMetrics(rlog.AvgMetric("trn_entropy", metargs=["trn_entropy", 1]))

    # Initialize the objects we will use during training.
    trn_env = get_env(opt)
    val_env = get_env(opt, clip_rewards_val=False)
    opt.action_num = trn_env.action_space.n
    agent = CoolAgent.init_from_opts(opt)

    is_replay_on_disk = Path(opt.out_dir).joinpath("replay.gz").is_file()
    # if we loaded a checkpoint
    if is_replay_on_disk:
        raise NotImplementedError("It needed refactoring anyway.")
    else:
        steps = 0
        start_epoch = 1
        # add some hardware and git info, log and save
        opt = ioutil.add_platform_info(opt)

    rlog.info(
        "\n{}\n{}\n{}\n".format(
            ioutil.config_to_string(opt), agent.encoder, agent.qval_fn
        )
    )
    ioutil.save_config(opt, opt.out_dir)

    # Start training

    last_state = None  # used by train_one_epoch to know how to resume episode.
    for epoch in range(start_epoch, opt.epoch_cnt + 1):

        # train for 250,000 steps
        steps, last_state = train_one_epoch(
            trn_env,
            agent,
            opt.train_step_cnt,
            total_steps=steps,
            last_state=last_state,
        )
        rlog.traceAndLog(epoch * opt.train_step_cnt)

        # validate for 125,000 steps
        validate(
            CoolValAgent(agent.encoder, agent.qval_fn, opt),
            val_env,
            opt.valid_step_cnt,
            rlog.getRootLogger(),
        )
        rlog.traceAndLog(epoch * opt.train_step_cnt)

        # save the checkpoint
        # if opt.save:
        #     ioutil.checkpoint_agent(
        #         opt.out_dir,
        #         steps,
        #         estimator=policy_evaluation.estimator,
        #         target_estimator=policy_evaluation.target_estimator,
        #         optim=policy_evaluation.optimizer,
        #         cfg=opt,
        #         replay=(
        #             replay
        #             if (epoch % opt.replay_save_freq == 0 or epoch == opt.epoch_cnt)
        #             else None
        #         ),
        #         save_every_replay=(opt.replay_save_freq == 1),
        #     )

        # kill-switch
        if Path(opt.out_dir).joinpath(".SAVE_AND_STOP").is_file():
            Path(opt.out_dir).joinpath(".SAVE_AND_STOP").unlink()
            raise TimeoutError(f"Killed by kill-switch @ {epoch}/{opt.epoch_cnt}.")


def main():
    """Liftoff"""
    run(parse_opts())


if __name__ == "__main__":
    main()
