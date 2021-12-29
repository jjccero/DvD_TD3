import argparse
import time

import gym
import numpy as np
import torch
from pbrl.common import Logger

from dvd_td3.dvd_td3 import DvDTD3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timestep', type=int, default=1000000)
    parser.add_argument('--bandits_interval', type=int, default=1)
    parser.add_argument('--obs_norm', action='store_true')
    parser.add_argument('--reward_norm', action='store_true')
    parser.add_argument('--reward_scaling', type=float, default=None)
    args = parser.parse_args()

    env_name = args.env
    seed = args.seed
    timestep = args.timestep

    population_size = 5
    start_timestep = 25000
    lr_actor = 3e-4
    lr_critic = 3e-4

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = '{}-{}-{}'.format(env_name, seed, int(time.time()))
    filename_log = 'result/{}'.format(log_dir)
    filename_policy = 'result/{}/policy-{}.pkl'.format(log_dir, '{}')

    env = gym.make(env_name)
    config_net = dict(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_sizes=[256, 256, 256],
        activation=torch.nn.ReLU,
        obs_norm=args.obs_norm,
        reward_norm=args.reward_norm,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    )
    trainer = DvDTD3(
        config_policy=config_net,
        population_size=population_size,
        bandits=[0.0, 0.5],
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        reward_scaling=args.reward_scaling
    )
    logger = Logger(filename_log)
    trainer.learn(
        env_fn=lambda: gym.make(env_name),
        timestep=timestep,
        seed=seed,
        logger=logger,
        log_interval=1000,
        timestep_update=5,
        bandits_interval=args.bandits_interval,
        start_timestep=start_timestep,
        max_episode_steps=env.spec.max_episode_steps
    )
    trainer.save(filename_policy)


if __name__ == '__main__':
    main()
