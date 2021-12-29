import multiprocessing
import time

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.w = torch.nn.Parameter(torch.ones(2))
        self.b = torch.nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return self.w * x + self.b


def worker(
        remote: multiprocessing.connection.Connection,
        model: Model
):
    x = torch.Tensor([1.0, -1.0]).cuda()
    while True:
        _ = remote.recv()
        print('w: {}, b: {}.'.format(model.w.data, model.b.data))
        # forward
        y = model(x)
        loss = y.sum()
        model.zero_grad()
        # backward
        loss.backward()
        # the gradients are shared tensors, therefore, remember to clone gradients
        # which may be modified when optimizer.zero_grad() by other process
        grads = {k: v.grad.clone() for k, v in model.named_parameters()}
        remote.send(grads)


if __name__ == '__main__':
    model = Model()
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # define a process context
    ctx = multiprocessing.get_context('spawn')
    remote, remote_worker = ctx.Pipe()
    p = ctx.Process(
        target=worker,
        args=(remote_worker, model)
    )
    p.start()
    # after starting the subprocess 'p', the remote_worker can be closed
    remote_worker.close()
    while True:
        time.sleep(1)
        remote.send(None)
        grad = remote.recv()
        optimizer.zero_grad()
        # add the gradients before each gradient step
        for k, v in model.named_parameters():
            # the gradient may be None at first
            if v.grad is None:
                v.grad = grad[k] + 0.0
            else:
                v.grad += grad[k]
        optimizer.step()
