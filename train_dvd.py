import argparse
import time

import numpy as np
import torch

from dvd.server import DvD
from dvd.worker import worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_num', type=int, default=5)

    parser.add_argument('--env', type=str, default='Humanoid-v3')
    parser.add_argument('--env_num', type=int, default=10)
    parser.add_argument('--episode_test', type=int, default=1)
    parser.add_argument('--timestep', type=int, default=1000000)
    parser.add_argument('--test_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--start_timestep', type=int, default=25000)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--timestep_update', type=int, default=5000)
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument('--policy_freq', type=int, default=2)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--noise_explore', type=float, default=0.1)
    parser.add_argument('--noise_target', type=float, default=0.2)
    parser.add_argument('--double_q', action='store_true')  # whether min(Q1,Q2) when updating actor

    parser.add_argument('--lr_actor', type=float, default=3e-4)
    parser.add_argument('--lr_critic', type=float, default=3e-4)
    args = parser.parse_args()
    seed = args.seed

    policy_config = dict(
        hidden_sizes=[256, 256],
        activation=torch.nn.ReLU,
        gamma=0.99,
        device=torch.device('cuda:0')
    )

    trainer_config = dict(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        repeat=args.timestep_update,
        noise_target=args.noise_target,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        double_q=args.double_q,
        tau=args.tau,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        arms=[0.0, 0.5],
    )

    pbt = DvD(
        **trainer_config,
        worker_num=args.worker_num,
        worker_fn=worker,
        log_dir='dvd-td3/{}/{}-{}'.format(args.env, args.seed, int(time.time())),
        worker_params=dict(
            policy_config=policy_config,
            env=args.env,
            env_num=args.env_num,
            seed=args.seed,
            episode_test=args.episode_test,
            test_interval=args.test_interval,
            log_interval=args.log_interval,
            timestep=args.timestep,
            start_timestep=args.start_timestep,
            timestep_update=args.timestep_update,
            max_episode_steps=args.max_episode_steps
        )
    )
    pbt.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pbt.run()


if __name__ == '__main__':
    main()
