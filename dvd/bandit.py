from typing import List

import numpy as np


class TS:
    def __init__(self, arms=List[float], random_choice=6):
        self.arms = arms
        self.arm_num = len(arms)
        self.alpha = np.ones(self.arm_num, dtype=int)
        self.beta = np.ones(self.arm_num, dtype=int)
        self.arm = 0
        self.choices = 0
        self.random_choice = random_choice

    @property
    def value(self):
        return self.arms[self.arm]

    def update(self, reward):
        self.choices += 1
        if reward:
            self.alpha[self.arm] += 1
        else:
            self.beta[self.arm] += 1

    def sample(self):
        if self.choices < self.random_choice:
            self.arm = np.random.choice(self.arm_num)
        else:
            self.arm = np.argmax(np.random.beta(self.alpha, self.beta))
        return self.value

    def clear(self):
        self.alpha[:] = 1
        self.beta[:] = 1
        self.choices = 0


class UCB:
    def __init__(self, arms=List[float]):
        self.arms = arms
        self.arm_num = len(arms)
        self.alpha = np.zeros(self.arm_num, dtype=int)
        self.t = np.zeros(self.arm_num, dtype=int)
        self.arm = 0
        self.choices = 0
        self.random_choices = np.repeat(np.arange(self.arm_num), 2)
        np.random.shuffle(self.random_choices)

    @property
    def value(self):
        return self.arms[self.arm]

    def update(self, reward):
        self.choices += 1
        self.t[self.arm] += 1
        if reward:
            self.alpha[self.arm] += 1

    def sample(self):
        if self.choices < self.random_choices.shape[0]:
            self.arm = self.random_choices[self.choices]
        else:
            est = self.alpha / self.t + np.sqrt(2 * np.log(self.choices) / self.t)
            self.arm = np.argmax(est)
        return self.value

    def clear(self):
        self.alpha[:] = 0
        self.t[:] = 0
        self.choices = 0
