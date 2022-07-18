import multiprocessing

from pbrl.algorithms.dqn.buffer import ReplayBuffer
from pbrl.algorithms.td3.policy import Policy
from pbrl.algorithms.trainer import Trainer


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
            timestep_update: int,
            **kwargs
    ):
        assert timestep_update % self.worker_num == 0
        return super(DvDTD3, self).learn(timestep_update=timestep_update // self.worker_num, **kwargs)
