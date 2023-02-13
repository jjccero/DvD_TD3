import multiprocessing

import numpy as np

from pbrl.algorithms.dqn.buffer import ReplayBuffer
from pbrl.algorithms.td3.policy import Policy
from pbrl.algorithms.trainer import Trainer
from pbrl.common import update_dict


class DvDTD3(Trainer):
    def __init__(
            self,
            policy: Policy,
            worker_num: int,
            buffer_size: int,
            remote: multiprocessing.connection.Connection
    ):
        super(DvDTD3, self).__init__()
        self.policy = policy
        self.worker_num = worker_num
        self.buffer = ReplayBuffer(buffer_size)
        self.remote = remote
        self.remote.send(('init', self.policy))
        self.remote.recv()

    def update(self) -> dict:
        timestep_local = self.buffer.len
        assert timestep_local < self.buffer.buffer_size
        self.remote.send(
            (
                'update',
                (
                    timestep_local,
                    self.buffer.data[:timestep_local],
                    self.policy.rms_obs,
                    self.policy.rms_reward
                )
            )
        )
        self.timestep, self.iteration, loss_info = self.remote.recv()
        self.buffer.clear()
        return loss_info

    def learn(
            self,
            timestep: int,
            runner_train,
            timestep_update: int,
            logger=None,
            log_interval=0,
            runner_test=None,
            test_interval=0,
            episode_test=0
    ):
        assert timestep_update % self.worker_num == 0

        info = dict()

        runner_test.reset()
        test_info = runner_test.run(policy=self.policy, episode_num=episode_test)
        update_dict(info, test_info, 'test/')
        logger.log(self.timestep, info)

        while self.timestep < timestep:
            score = np.mean(test_info['reward'])
            self.remote.send(('eval', score))
            self.remote.recv()

            train_info = super(DvDTD3, self).learn(
                timestep=timestep_update // self.worker_num,
                runner_train=runner_train,
                timestep_update=timestep_update // self.worker_num
            )
            update_dict(info, train_info)
            if self.timestep % test_interval == 0:
                # evaluate
                runner_test.reset()
                test_info = runner_test.run(policy=self.policy, episode_num=episode_test)
                update_dict(info, test_info, 'test/')

            if self.timestep % log_interval == 0:
                logger.log(self.timestep, info)

        score = np.mean(test_info['reward'])
        self.remote.send(('eval', score))
        self.remote.recv()
        return info
