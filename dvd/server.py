import copy
from typing import Optional, Dict, List

import numpy as np
import torch

from dvd.bandit import TS
from dvd.loss import LogDet
from pbrl.algorithms.dqn.buffer import ReplayBuffer
from pbrl.algorithms.trainer import Trainer
from pbrl.common import update_dict
from pbrl.common.map import auto_map
from pbrl.common.rms import RunningMeanStd
from pbrl.pbt import PBT


class DvD(PBT):
    def __init__(
            self,
            arms,
            sample_size: int = 20,
            batch_size: int = 256,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            repeat: int = 1000,
            tau: float = 0.005,
            gamma: float = 0.99,
            noise_target: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            double_q: bool = True,
            buffer_size: int = 1000000,
            optimizer=torch.optim.Adam,
            **kwargs
    ):
        super(DvD, self).__init__(exploit=False, **kwargs)
        self.gamma = gamma
        self.noise_target = noise_target
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.double_q = double_q
        self.batch_size = batch_size
        self.repeat = repeat
        self.buffer = ReplayBuffer(buffer_size=buffer_size)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau
        self.logdet = LogDet()

        cmd = self.recv()
        assert cmd == 'init'
        self.policies = tuple(self.objs)
        for remote in self.remotes:
            remote.send(None)

        self.optimizer_actors = optimizer(
            tuple({'params': policy.actor.parameters()} for policy in self.policies),
            lr=self.lr_actor
        )
        self.optimizer_critic = optimizer(
            self.critic.parameters(), lr=self.lr_critic
        )
        self.timestep = 0
        self.iteration = 0
        self.best = 0
        for policy in self.policies:
            policy.actor.train()
            policy.actor_target.eval()
        self.sample_size = sample_size
        self.bandit = TS(arms=arms)
        self.div_coef = 0.0
        self.rms_obs: Optional[RunningMeanStd] = None
        self.rms_reward: Optional[RunningMeanStd] = None

    @property
    def critic(self):
        return self.policies[0].critic

    @property
    def critic_target(self):
        return self.policies[0].critic_target

    @property
    def policy(self):
        return self.policies[self.best]

    def normalize_observations(self, observations):
        if self.rms_obs is not None:
            observations = self.rms_obs.normalize(observations)
        return observations

    def normalize_rewards(self, rewards):
        if self.rms_reward is not None:
            rewards = self.rms_reward.normalize(rewards)
        return rewards

    def critic_loss(self):
        observations, actions, observations_next, rewards, dones = self.buffer.sample(self.batch_size)

        # use global RunningMeanStd when we share a central Q-function
        obs_global = self.normalize_observations(observations)
        obs_next_global = self.normalize_observations(observations_next)

        rewards = self.normalize_rewards(rewards)

        # use local RunningMeanStd
        obs_next_worker = self.policy.normalize_observations(observations_next)

        obs_global, obs_next_global, obs_next_worker, actions, rewards, dones = auto_map(
            self.policy.n2t,
            (obs_global, obs_next_global, obs_next_worker, actions, rewards, dones)
        )

        with torch.no_grad():
            actions_target, _ = self.policy.actor_target.forward(obs_next_worker)
            noises_target = torch.clamp(
                torch.randn_like(actions_target) * self.noise_target,
                -self.noise_clip,
                self.noise_clip
            )
            actions_target = torch.clamp(actions_target + noises_target, -1.0, 1.0)

            q1_target, q2_target = self.critic_target.forward(obs_next_global, actions_target)
            q_target = torch.min(q1_target, q2_target)
            y = rewards + ~dones * self.gamma * q_target

        q1, q2 = self.critic.forward(obs_global, actions)
        td1_loss = 0.5 * torch.square(y - q1).mean()
        td2_loss = 0.5 * torch.square(y - q2).mean()

        return td1_loss, td2_loss, observations

    def policy_loss(self, observations):
        indices = np.random.randint(self.batch_size, size=self.sample_size)
        sampled_observations = auto_map(lambda x: x[indices], observations)
        policy_losses = []
        embeddings = []
        glo_obs = auto_map(self.policy.n2t, self.normalize_observations(observations))
        for policy in self.policies:
            obs_worker = auto_map(policy.n2t, policy.normalize_observations(observations))

            actions, _ = policy.actor.forward(obs_worker)
            if self.double_q:
                q1, q2 = self.critic.forward(glo_obs, actions)
                policy_loss = torch.min(q1, q2)
            else:
                policy_loss = self.critic.Q1(glo_obs, actions)
            # normalize reward
            # policy_loss = (policy_loss / (torch.std(policy_loss.detach()) + 1e-8)).mean()
            policy_loss = policy_loss.mean()
            policy_losses.append(policy_loss)

            obs_dpp_worker = auto_map(policy.n2t, policy.normalize_observations(sampled_observations))
            embedding, _ = policy.actor.forward(obs_dpp_worker)
            embeddings.append(embedding.flatten())

        logdet = self.logdet.forward(embeddings)
        det = logdet.exp()
        return policy_losses, det

    def train_loop(self, loss_info_global: Dict, loss_info_workers: List[Dict]):
        self.critic.train()

        td1_loss, td2_loss, raw_observations = self.critic_loss()
        critic_loss = td1_loss + td2_loss
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        self.critic.eval()

        loss_info_global['td1'].append(td1_loss.item())
        loss_info_global['td2'].append(td2_loss.item())

        if self.iteration % self.policy_freq == 0:
            policy_losses, det = self.policy_loss(raw_observations)

            actor_loss = (self.div_coef - 1.0) * torch.stack(policy_losses).sum() - self.div_coef * det
            self.optimizer_actors.zero_grad()
            actor_loss.backward()
            self.optimizer_actors.step()

            Trainer.soft_update(self.critic, self.critic_target, self.tau)

            loss_info_global['det'].append(det.item())
            for worker_id in range(self.worker_num):
                loss_info_workers[worker_id]['policy'].append(policy_losses[worker_id].item())
                policy = self.policies[worker_id]
                Trainer.soft_update(policy.actor, policy.actor_target, self.tau)

    def run(self):
        last_max_return = None
        info = {}
        while True:
            cmd = self.recv()
            if cmd == 'update':
                for worker_id in range(self.worker_num):
                    timestep_local, buffer_slice, rms_obs, rms_reward = self.objs[worker_id]
                    self.timestep += timestep_local
                    ptr = self.buffer.ptr
                    buffer_size = self.buffer.buffer_size
                    assert timestep_local <= buffer_size
                    if ptr + timestep_local <= buffer_size:
                        self.buffer.data[ptr:ptr + timestep_local] = buffer_slice
                    else:
                        self.buffer.data[ptr:] = buffer_slice[:buffer_size - ptr]
                        self.buffer.data[:timestep_local + ptr - buffer_size] = buffer_slice[buffer_size - ptr:]
                    self.buffer.ptr = (ptr + timestep_local) % buffer_size
                    self.buffer.len = min(self.buffer.len + timestep_local, buffer_size)

                    if rms_obs is not None:
                        self.policies[worker_id].rms_obs = rms_obs
                        if worker_id == 0:
                            self.rms_obs = copy.deepcopy(rms_obs)
                        else:
                            self.rms_obs.extend(rms_obs)
                    if rms_reward is not None:
                        if worker_id == 0:
                            self.rms_reward = rms_reward
                        else:
                            self.rms_reward.extend(rms_reward)

                loss_info_global = dict(td1=[], td2=[], det=[])
                loss_info_global['lambda'] = self.div_coef
                loss_info_workers = [dict(policy=[]) for _ in range(self.worker_num)]

                for _ in range(self.repeat):
                    self.iteration += 1
                    self.train_loop(loss_info_global, loss_info_workers)

                for worker_id in range(self.worker_num):
                    self.remotes[worker_id].send((self.timestep, self.iteration, loss_info_workers[worker_id]))
                update_dict(info, loss_info_global, 'loss/')
            elif cmd == 'eval':
                returns = np.array(self.objs)
                self.best = np.argmax(returns)
                max_return = self.objs[self.best]
                current_solutions_info = dict(
                    max_fitness=max_return,
                    mean_fitness=np.mean(returns),
                    min_fitness=np.min(returns),
                    med_fitness=np.median(returns),
                )
                if self.bandit is not None:
                    if last_max_return is not None:
                        increment = max_return - last_max_return
                        self.bandit.update(increment > 0)
                    last_max_return = max_return
                    self.div_coef = self.bandit.sample()
                    for remote in self.remotes:
                        remote.send(None)
                update_dict(info, current_solutions_info, 'current/')
                self.logger.log(self.timestep, info)

            elif cmd == 'close':
                self.save()
                self.close()
                break

    def save(self):
        for worker_id in range(self.worker_num):
            pkl = {
                'timestep': self.timestep,
                'iteration': self.iteration,
                'actor': self.policies[worker_id].actor.state_dict(),
                'critic': self.critic.state_dict(),
                'rms_obs': self.policy.rms_obs,
                'rms_reward': self.policy.rms_reward,
                'optimizer_actor': self.optimizer_actors.state_dict(),  # todo:divide it into worker_num parts
                'optimizer_critic': self.optimizer_critic.state_dict()
            }
            torch.save(auto_map(map_cpu, pkl), '{}/{}.policy.pkl'.format(self.log_dir, worker_id))
