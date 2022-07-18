# DvD-TD3: Diversity via Determinants for TD3 version

The implementation of paper
[Effective Diversity in Population Based Reinforcement Learning](https://arxiv.org/abs/2002.00632).

## Install

Install [pbrl](https://github.com/jjccero/pbrl) and clone this repo:

```
git clone https://github.com/jjccero/DvD_TD3
cd DvD_TD3
python train_dvd.py
```

## Notes

### Kernel Matrix

When DPP kernel matrix uses **dot product kernel** (or **cosine similarity**, see [loss.py](dvd/loss.py)) instead of
**RBF** as entry, we can take a linear mapping to make the value between 0 and 1. The **beta** makes the matrix
positive-definite.

### log det

I'm not sure whether to take the logarithms of determinant. The author believes that this does not matter. In addition,
I find that the numerical instability of **log det** may be the reason for the gradients explosion or disappearance of
Actors' network, so I use **det** instead of **log det** loss for optimization.

---
Thank **Jack Parker-Holder** (the author of the paper) for his help. And welcome to get in touch with me if you have any
questions about this implementation.

