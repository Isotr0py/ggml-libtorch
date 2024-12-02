import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Set the seed of each random module.
    `torch.manual_seed` will set seed on all devices.

    Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
