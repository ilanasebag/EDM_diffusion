import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F

from utils import UNet 


def train(dataset, epochs=5, batch_size=128):

    train_dataloader = DataLoader(dataset, batch_size=batch_size)
