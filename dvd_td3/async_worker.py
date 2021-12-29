import multiprocessing
from typing import Optional

import numpy as np
import torch
from pbrl.algorithms.td3 import Policy
from pbrl.common.pickle import CloudpickleWrapper


def worker(
        remote: multiprocessing.connection.Connection,
        remote_parent: multiprocessing.connection.Connection,
        policy: Policy,
        env_fn: CloudpickleWrapper,
        seed: int
):
    remote_parent.close()
    policy.actor.eval()
    env = env_fn.x()
    env_test = env_fn.x()
    obs: Optional[np.ndarray] = None

    env.seed(seed)
    env_test.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    while True:
        cmd, data = remote.recv()
        if cmd == 'reset':
            obs = env.reset()
            remote.send(obs)

        elif cmd == 'step':
            random = data
            action = policy.step(np.expand_dims(obs, 0), None, random)[0].squeeze(0)
            wrapped_action = policy.wrap_actions(action)
            obs, reward, done, _ = env.step(wrapped_action)
            if done:
                obs = env.reset()
            remote.send((action, obs, reward, done))

        elif cmd == 'update':
            observations, rms_obs, policy_weight = data
            policy.rms_obs = rms_obs
            policy.actor.train()
            # observations have been normalized
            actions, _ = policy.actor.forward(observations)
            q1, q2 = policy.critic.forward(observations, actions)
            policy_loss = torch.min(q1, q2).mean()

            policy.actor.zero_grad()
            (-policy_weight * policy_loss).backward()

            grad = {k: v.grad.clone() for k, v in policy.actor.named_parameters()}
            remote.send((grad, policy_loss.item()))

            policy.actor.eval()

        elif cmd == 'test':
            episode_num = data
            episode_rewards = []
            for episode in range(episode_num):
                episode_reward = 0.0
                obs_ = env_test.reset()
                while True:
                    action_ = policy.act(np.expand_dims(obs_, 0), None)[0].squeeze(0)
                    obs_, reward_, done_, _ = env_test.step(
                        policy.wrap_actions(action_)
                    )
                    episode_reward += reward_
                    if done_:
                        episode_rewards.append(episode_reward)
                        break
            remote.send(episode_rewards)

        elif cmd == 'close':
            return
