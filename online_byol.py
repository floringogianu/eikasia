""" Entry point. """
from pathlib import Path

import numpy as np
import rlog
import torch
import torch.nn as nn
import torch.optim as O
from liftoff import parse_opts

import common.io_utils as ioutil
from common.ale import ALE
from rl import estimators
from rl.agents import AGENTS
from rl.replay import ExperienceReplay
from rl.rl_routines import Episode, validate
from train_byol import ImpalaEncoder


class SamplingWorldModel(nn.Module):
    """Used to encode the states, step by step."""

    def __init__(self, encoder, N=512, M=256) -> None:
        super().__init__()
        self.encoder = encoder
        self.cls_loop = nn.GRU(N, M)
        self.M = M
        self.N = N
        self.h0 = None

    def forward(self, x):
        assert x.dtype == torch.uint8, "Observations have to be uint8."
        x = x.float().div(255.0)
        z = self.encoder(x.squeeze(0))
        z = z.view(1, 1, self.N)
        # unroll
        out, self.h0 = self.cls_loop(z, self.h0)
        return out  # time now becomes "history"

    def reset(self, device=torch.device("cuda")):
        self.h0 = torch.zeros(1, 1, self.M, device=device)

    def load_checkpoint_(self, path):
        keys = self.state_dict().keys()
        state = torch.load(path)["model"]
        state = {".".join(k.split(".")[1:]): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if k in keys}
        self.load_state_dict(state)
        return self


class CoolValAgent:
    def __init__(self, encoder, qval_fn, opt) -> None:
        self.encoder = encoder
        self.qval_fn = qval_fn
        self.policy_improvement = AGENTS[opt.agent.name]["policy_improvement"](
            qval_fn,
            opt.action_num,
            epsilon=opt.val_epsilon,
        )

    def act(self, obs):
        with torch.no_grad():
            z = self.encoder(obs).detach()
            return self.policy_improvement.act(z)

    def reset(self):
        self.encoder.reset()


class CoolAgent:
    def __init__(
        self,
        encoder,
        qval_fn,
        replay,
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
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq

        # state
        self._transition = []
        self.total_steps = 0

    @classmethod
    def init_from_opts(cls, opt):

        encoder = SamplingWorldModel(ImpalaEncoder(1, **opt.estimator.encoder.args))
        encoder = encoder.load_checkpoint_(
            "{}/model_{:08d}.pkl".format(
                opt.estimator.encoder.root, opt.estimator.encoder.cidx
            )
        )

        z_size = encoder.M
        # get the value function
        qval_fn = getattr(estimators, opt.estimator.qval_net.name)(
            opt.action_num, input_size=z_size, **opt.estimator.qval_net.args
        )

        encoder.to(opt.device)
        qval_fn.to(opt.device)

        # replay
        if isinstance(opt.agent.args["epsilon"], list):
            opt.replay["warmup_steps"] = opt.agent.args["epsilon"][-1]
        opt.replay["hist_len"] = opt.agent.args["hist_len"]

        return cls(
            encoder,
            qval_fn,
            ExperienceReplay(**opt.replay),
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
        with torch.no_grad():
            z = self.encoder(obs).detach()
            pi = self.policy_improvement.act(z)

        if len(self._transition) == 4:
            # self._transition contents: [prev_z, action, reward, done]
            self.replay.push([*self._transition[:-1], z.cpu(), self._transition[-1]])

        # reset self._transition
        del self._transition[:]
        # store z_t, a_t
        self._transition = [z.cpu(), pi.action]

        return pi

    def learn(self, transition):
        """Here we orchestrate learning across the different components of the agent."""
        self.total_steps += 1
        self.policy_evaluation.estimator.train()
        self.policy_evaluation.target_estimator.train()

        _, _, reward, _, done, _ = transition

        # append to the temporary transition tuple
        self._transition += [reward, done]

        # reset the hidden state when done
        if done:
            self.encoder.reset()

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
        torch.cuda.empty_cache()
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
    ioutil.save_config(opt)

    torch.cuda.empty_cache()
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
