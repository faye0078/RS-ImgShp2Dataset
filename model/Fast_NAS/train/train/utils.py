import copy
import json
import logging
import os
import time

import numpy as np
import torch

class TaskPerformer(object):
    def __init__(self, maxval, delta=0.3):
        """
        Args:
          maxval (float) : initial maximum value
          delta (float) : how large difference (in %) is allowable between curval and maxval

        """
        self.maxval = maxval
        self.delta = delta
        self.scheduler = {
            100: 0.9,  # after k steps, multiply by v
            200: 0.8,
            300: 0.7,
            400: 0.6,
        }
        self.n_steps = 0
        self.decay = 0.99

    def _update_delta(self):
        mult = self.scheduler.get(self.n_steps, 1.0)
        self.delta *= mult
        self.n_steps += 1

    def _update_maxval(self, newval):
        self.maxval = self.decay * self.maxval + (1.0 - self.decay) * newval

    def step(self, newval):
        self._update_delta()
        self._update_maxval(newval)
        if newval > self.maxval:
            self.n_steps += 1
            return True
        prct = 1.0 - np.random.uniform(0.0, high=self.delta)
        if newval > (self.maxval * prct):
            return True
        return False

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def try_except(func):
    """Try / except wrapper

    Args:
      func (lambda) : function to execute

    Returns fun output or 0 otherwise
    """

    def wrapper_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError:
            return 0

    return wrapper_func

def init_polyak(do_polyak, module):
    if not do_polyak:
        return None
    else:
        try:
            return copy.deepcopy(list(p.data for p in module.parameters()))
        except RuntimeError:
            return None

def apply_polyak(do_polyak, module, avg_param):
    if do_polyak:
        try:
            for p, avg_p in zip(module.parameters(), avg_param):
                p.data.copy_(avg_p)
        except RuntimeError:
            return None


