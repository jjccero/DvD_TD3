# DvD-TD3: Diversity via Determinants for TD3 version

The implementation of paper
[Effective Diversity in Population Based Reinforcement Learning](https://arxiv.org/abs/2002.00632).

## Install

Install [pbrl](https://github.com/jjccero/pbrl) and clone this repo:

```
git clone https://github.com/jjccero/DvD_TD3
cd DvD_TD3
python train.py
```

## Notes

I train agents using multiprocessing, and [demo_grad.py](demo_grad.py) shows how gradients are transferred in different
processes.

When DPP kernel matrix uses **dot product kernel** (or **cosine similarity**, see [loss.py](dvd_td3/loss.py)) as entry, we can take a
linear mapping to make the value between 0 and 1.

Training may cost a lot because evaluation (bandits' update) after every iteration, so I reduced the frequency of
evaluation to 0.01.

---
Thank **Jack Parker-Holder** (the author of the paper) for his help.  
And welcome to get in touch with me if you have any questions about this implementation.

