# This file contains some helper functions

import torch
import torchvision
from matplotlib import pyplot as plt


def gaussian_noise_corruption(x, amount):
    """ corrupt the input 'x' by mixing it with gaussian noise according to the parameter 'amount'"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1-amount) + noise * amount

