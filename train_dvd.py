import argparse
import time

import numpy as np
import torch

from dvd.server import DvD
from dvd.worker import worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--worker_num', type=int, default=5)
    parser.add_argument('--timestep', type=int, default=1000000)
    args = parser.parse_args()
    seed = args.seed

    policy_config = dict(
        hidden_sizes=[256, 256],
        activation=torch.nn.ReLU,
        gamma=0.99,
        obs_norm=True,
        device=torch.device('cuda:0')
    )
    pbt = DvD(
        arms=[0.0, 0.5],
        worker_num=args.worker_num,
        worker_fn=worker,
        policy_config=policy_config,
        env=args.env,
        env_num=10,
        seed=args.seed,
        timestep=args.timestep,
        start_timestep=5000,
        repeat=1000,
        timestep_update=1000,
        log_interval=5000,
        episode_num_test=1,
        log_dir='dvd-td3/{}/{}-{}'.format(args.env, args.seed, int(time.time()))
    )
    pbt.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pbt.run()


if __name__ == '__main__':
    main()
