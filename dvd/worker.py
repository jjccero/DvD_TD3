import multiprocessing

import gym
import numpy as np
import torch

from dvd.td3 import DvDTD3
from pbrl.algorithms.dqn import Runner
from pbrl.algorithms.td3 import Policy
from pbrl.algorithms.td3.net import DoubleQ
from pbrl.common import Logger
from pbrl.env import DummyVecEnv


def worker(
        worker_num: int, worker_id: int, log_dir: str,
        remote: multiprocessing.connection.Connection,
        policy_config: dict,
        env: str,
        env_num: int,
        seed: int,
        episode_test: int,
        test_interval: int,
        log_interval: int,
        timestep: int,
        start_timestep: int,
        timestep_update: int,
        max_episode_steps: int
):
    seed_worker = seed + worker_id
    torch.manual_seed(seed_worker)
    np.random.seed(seed_worker)
    env_train = DummyVecEnv([lambda: gym.make(env) for _ in range(env_num)])
    env_test = DummyVecEnv([lambda: gym.make(env) for _ in range(episode_test)])
    env_train.seed(seed_worker)
    env_test.seed(seed_worker)

    filename_log = '{}/{}'.format(log_dir, worker_id)

    policy = Policy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        critic_type=DoubleQ if worker_id == 0 else None,
        **policy_config
    )
    trainer = DvDTD3(
        policy=policy,
        worker_num=worker_num,
        buffer_size=start_timestep // worker_num + 1,
        remote=remote
    )
    runner_train = Runner(
        env=env_train,
        max_episode_steps=max_episode_steps,
        start_timestep=start_timestep // worker_num
    )
    runner_test = Runner(env=env_test)

    logger = Logger(filename_log)
    trainer.learn(
        timestep=timestep,
        runner_train=runner_train,
        timestep_update=timestep_update,
        logger=logger,
        log_interval=log_interval,
        runner_test=runner_test,
        test_interval=test_interval,
        episode_test=episode_test
    )

    remote.send(('close', None))
