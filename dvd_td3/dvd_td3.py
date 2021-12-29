import copy
import multiprocessing
import os
from typing import List, Optional, Dict

import numpy as np
import torch
from pbrl.algorithms.td3 import Policy, ReplayBuffer
from pbrl.algorithms.td3.net import DoubleQ
from pbrl.algorithms.trainer import Trainer
from pbrl.common import Logger, update_dict
from pbrl.common.pickle import CloudpickleWrapper

from dvd_td3.async_worker import worker
from dvd_td3.bandits import Bandits
from dvd_td3.loss import DvD


class DvDTD3(Trainer):
    def __init__(
            self,
            config_policy: dict,
            population_size: int,
            bandits: Optional[List[float]] = None,
            sample_obs: int = 20,
            buffer_size: int = 1000000,
            batch_size: int = 256,
            gamma: float = 0.99,
            noise_target: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            double_q: bool = True,
            tau: float = 0.005,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            reward_scaling: Optional[float] = None
    ):
        super(DvDTD3, self).__init__()
        self.population_size = population_size
        self.bandits = None
        self.div_weight = 0.0
        if bandits:
            self.bandits = Bandits(arms=bandits)
            self.div_weight = self.bandits.value
        self.sample_obs = sample_obs
        self.dvd = DvD()
        config_policy['rnn'] = None
        self.policies = [
            Policy(critic=False, **config_policy) for _ in range(self.population_size)
        ]
        self.policy = self.policies[0]
        self.critic = DoubleQ(
            obs_dim=self.policy.observation_space.shape,
            action_dim=self.policy.action_space.shape[0],
            hidden_sizes=self.policy.hidden_sizes,
            activation=self.policy.activation
        ).to(self.policy.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        self.buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=self.policy.observation_space,
            action_space=self.policy.action_space
        )

        self.batch_size = batch_size
        self.gamma = gamma
        self.noise_target = noise_target
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.double_q = double_q
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.optimizers_actor = [
            torch.optim.Adam(policy.actor.parameters(), self.lr_actor) for policy in self.policies
        ]
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr_critic
        )
        for policy in self.policies:
            # shared obs_filter, reward_filter and critic
            policy.rms_obs = self.policy.rms_obs
            policy.rms_reward = self.policy.rms_reward
            policy.critic = self.critic
            policy.critic_target = self.critic_target
        self.reward_scaling = reward_scaling

    def y(
            self,
            policy: Policy,
            observations_next: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor
    ):
        with torch.no_grad():
            actions_target, _ = policy.actor_target.forward(observations_next)
            noises_target = torch.clamp(
                torch.randn_like(actions_target) * self.noise_target,
                -self.noise_clip,
                self.noise_clip
            )
            actions_target = torch.clamp(actions_target + noises_target, -1.0, 1.0)
            q1_target, q2_target = self.critic_target.forward(observations_next, actions_target)
            q_target = torch.min(q1_target, q2_target)
            y = rewards + (1.0 - dones) * self.gamma * q_target
        return y

    def critic_loss(self):
        observations_all = []
        actions_all = []
        y_all = []
        for policy in self.policies:
            observations, actions, observations_next, rewards, dones = self.buffer.sample(self.batch_size)
            observations = self.policy.normalize_observations(observations)
            observations_next = self.policy.normalize_observations(observations_next)
            if self.reward_scaling:
                rewards = rewards / self.reward_scaling
            rewards = self.policy.normalize_rewards(rewards)
            observations, actions, observations_next, rewards, dones = map(
                policy.n2t,
                (observations, actions, observations_next, rewards, dones)
            )
            observations_all.append(observations)
            actions_all.append(actions)
            y_all.append(self.y(policy, observations_next, rewards, dones))

        q1, q2 = self.critic.forward(torch.cat(observations_all), torch.cat(actions_all))
        y = torch.cat(y_all)
        q1_loss = 0.5 * ((y - q1) ** 2).mean()
        q2_loss = 0.5 * ((y - q2) ** 2).mean()

        return q1_loss, q2_loss, observations_all

    def diversity_det(self, observations_all: List[torch.Tensor]) -> torch.Tensor:
        observations_all = torch.cat(observations_all)
        # sample observation
        observations = observations_all[torch.randint(observations_all.shape[0], size=(self.sample_obs,))]
        # embedding_dim = embed_size
        embeddings = torch.stack([policy.actor.forward(observations)[0].flatten() for policy in self.policies])
        det = self.dvd.forward(embeddings)
        return det

    def test_async(
            self,
            remotes: List[multiprocessing.connection.Connection],
            episode_test: int
    ):
        rewards_sum = 0.0
        test_info = dict()
        for remote in remotes:
            remote.send(('test', episode_test))
        for i in range(self.population_size):
            episode_rewards = remotes[i].recv()
            test_info['{}/reward'.format(i)] = episode_rewards
            for episode_reward in episode_rewards:
                rewards_sum += episode_reward
        return test_info, rewards_sum / self.population_size / episode_test

    def update_async(self, remotes: List[multiprocessing.connection.Connection]):
        loss_info = dict()
        self.critic.train()

        q1_loss, q2_loss, observations_all = self.critic_loss()
        critic_loss = q1_loss + q2_loss
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        if self.iteration % self.policy_freq == 0:
            for observations, remote in zip(observations_all, remotes):
                remote.send(('update', (observations, self.policy.rms_obs, (1.0 - self.div_weight))))
            det = self.diversity_det(observations_all)

            for optimizer_actor in self.optimizers_actor:
                optimizer_actor.zero_grad()

            if self.div_weight:
                (-self.div_weight * torch.log(det)).backward()

            for i in range(self.population_size):
                grad, policy_loss = remotes[i].recv()
                for k, v in self.policies[i].actor.named_parameters():
                    if v.grad is None:
                        v.grad = grad[k]
                    else:
                        v.grad += grad[k]
                self.optimizers_actor[i].step()

            Trainer.soft_update(self.critic, self.critic_target, self.tau)
            for policy in self.policies:
                Trainer.soft_update(policy.actor, policy.actor_target, self.tau)

            loss_info['div_weight'] = self.div_weight
            loss_info['det'] = det.item()

        loss_info['q1'] = q1_loss.item()
        loss_info['q2'] = q2_loss.item()
        return loss_info

    def learn(
            self,
            env_fn,
            timestep: int,
            seed: int,
            logger: Optional[Logger] = None,
            log_interval=0,
            timestep_update=1,
            bandits_interval=500,
            n_collect_steps=1000,
            episode_test=1,
            start_timestep=0,
            max_episode_steps=np.inf
    ):
        # dvd-td3 is running asynchronously
        # using update_async() instead of update()
        ctx = multiprocessing.get_context('spawn')
        remotes = []
        ps = []
        for i in range(self.population_size):
            remote, remote_worker = ctx.Pipe()
            p = ctx.Process(
                target=worker,
                args=(
                    remote_worker,
                    remote,
                    self.policies[i],
                    CloudpickleWrapper(env_fn),
                    seed + i
                ),
                daemon=False
            )
            p.start()
            ps.append(p)
            remote_worker.close()
            remotes.append(remote)

        timestep += self.timestep
        info = {
            'train/reward': [],
            'train/episode_step': []
        }
        episode_rewards = np.zeros(self.population_size)
        episode_steps = np.zeros(self.population_size, dtype=int)
        returns = np.zeros(self.population_size)

        for remote in remotes:
            remote.send(('reset', None))
        observations = np.asarray([remote.recv() for remote in remotes])
        if log_interval and self.timestep == 0:
            test_info, _ = self.test_async(remotes, episode_test)
            update_dict(info, test_info, 'test/')
            logger.log(self.timestep, info)
        while True:
            for remote in remotes:
                remote.send(('step', self.timestep < start_timestep))
            actions, observations_next, rewards, dones = map(
                np.asarray,
                zip(*[remote.recv() for remote in remotes])
            )

            self.timestep += self.population_size
            episode_rewards += rewards
            episode_steps += 1

            self.policy.normalize_observations(observations, True)
            self.policy.normalize_rewards(rewards, True, returns)
            self.buffer.append(
                observations,
                actions,
                observations_next,
                rewards,
                dones & (episode_steps < max_episode_steps)
            )
            observations = observations_next
            for i in range(self.population_size):
                if dones[i]:
                    info['train/reward'].append(episode_rewards[i])
                    info['train/episode_step'].append(episode_steps[i])
                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
                    returns[i] = 0.0

            if self.timestep > n_collect_steps:
                for i in range(self.population_size):
                    loss_info = self.update_async(remotes)
                    update_dict(info, loss_info, 'loss/')
                    self.iteration += 1
                    # update bandits after every iteration
                    if self.iteration % bandits_interval == 0:
                        test_info, reward_mean = self.test_async(remotes, episode_test)
                        update_dict(info, test_info, 'test/')
                        if self.bandits:
                            self.bandits.update(reward_mean)
                            self.div_weight = self.bandits.sample()

                done = self.timestep >= timestep
                if log_interval and (self.iteration % log_interval == 0 or done):
                    logger.log(self.timestep, info)
                if done:
                    break
        for remote in remotes:
            remote.send(('close', None))
        for p in ps:
            p.join()

    def load(self, filename: str):
        for i in range(self.population_size):
            filename_policy = filename.format(i)
            if os.path.exists(filename_policy):
                pkl = torch.load(filename_policy)
                print(pkl)

    def save(self, filename: str):
        for i in range(self.population_size):
            pkl = {
                'timestep': self.timestep,
                'iteration': self.iteration,
                'lr_actor': self.lr_actor,
                'lr_critic': self.lr_critic,
                'actor': {k: v.cpu() for k, v in self.policies[i].actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()},
                'rms_obs': self.policies[i].rms_obs,
                'rms_reward': self.policies[i].rms_reward,
                'optimizer_actor': self.optimizers_actor[i].state_dict(),
                'optimizer_critic': self.optimizer_critic.state_dict()
            }
            torch.save(pkl, filename.format(i))
