import multiprocessing

import gym
import numpy as np
import torch
from pbrl.algorithms.dqn import Runner
from pbrl.algorithms.td3 import Policy
from pbrl.algorithms.td3.net import DoubleQ
from pbrl.common import Logger, update_dict
from pbrl.env import DummyVecEnv

from dvd.td3 import DvDTD3


def worker(
        worker_num: int, worker_id: int,
        remote: multiprocessing.connection.Connection,
        remote_parent: multiprocessing.connection.Connection,
        policy_config: dict,
        env: str,
        env_num: int,
        seed: int,
        episode_num_test: int,
        log_interval: int,
        log_dir: str,
        timestep: int,
        start_timestep: int,
        timestep_update: int
):
    remote_parent.close()

    seed_worker = seed + worker_id
    torch.manual_seed(seed_worker)
    np.random.seed(seed_worker)
    env_train = DummyVecEnv([lambda: gym.make(env) for _ in range(env_num)])
    env_test = DummyVecEnv([lambda: gym.make(env) for _ in range(1)])
    env_train.seed(seed_worker)
    env_test.seed(seed_worker)

    filename_log = '{}/{}'.format(log_dir, worker_id)
    logger = Logger(filename_log)
    policy = Policy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        critic_type=DoubleQ if worker_id == 0 else None,
        **policy_config
    )
    trainer = DvDTD3(
        policy=policy,
        worker_num=worker_num,
        buffer_size=10000,
        remote=remote
    )
    runner_train = Runner(
        env=env_train,
        max_episode_steps=1000,
        start_timestep=start_timestep // worker_num,
        fill=True
    )
    runner_test = Runner(env=env_test)

    info = dict()

    runner_test.reset()
    eval_info = runner_test.run(policy=policy, episode_num=episode_num_test)
    update_dict(info, eval_info, 'test/')
    logger.log(trainer.timestep, info)

    while trainer.timestep < timestep:
        score = np.mean(eval_info['reward'])
        remote.send(('eval', score))
        remote.recv()

        train_info = trainer.learn(
            timestep=timestep_update,
            runner_train=runner_train,
            timestep_update=timestep_update
        )

        # evaluate
        runner_test.reset()
        eval_info = runner_test.run(policy=policy, episode_num=episode_num_test)

        update_dict(info, train_info)
        update_dict(info, eval_info, 'test/')

        if trainer.iteration % log_interval == 0:
            logger.log(trainer.timestep, info)

    remote.send(('close', None))
