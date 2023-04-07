import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mean(tensor: torch.Tensor):
    return tensor.mean(0, keepdim=True)


def average_diff(tensor1: torch.Tensor, tensor2: torch.Tensor):
    if isinstance(tensor1, list):
        tensor = []
        for t1, t2 in zip(tensor1, tensor2):
            tensor.append(t1 - mean(t2))
    else:
        tensor = tensor1 - mean(tensor2)
    return tensor


class GANLoss(nn.Module):
    def __init__(self, metric: str, smoothing: float = 1.0):
        super().__init__()
        self.register_buffer("label_real", torch.tensor(1.0))
        self.register_buffer("label_fake", torch.tensor(0.0))
        self.metric = metric
        self.smoothing = smoothing

    def forward(self, pred_real, pred_fake, mode):
        if mode == "G":
            return self.loss_G(pred_real, pred_fake)
        elif mode == "D":
            return self.loss_D(pred_real, pred_fake)
        else:
            raise ValueError

    def loss_D(self, pred_real, pred_fake):
        loss = 0
        if self.metric == "nsgan":
            loss += F.softplus(-pred_real).mean()
            loss += F.softplus(pred_fake).mean()
        elif self.metric == "wgan":
            loss += -pred_real.mean()
            loss += pred_fake.mean()
        elif self.metric == "lsgan":
            target_real = self.label_real.expand_as(pred_real) * self.smoothing
            target_fake = self.label_fake.expand_as(pred_fake)
            loss += F.mse_loss(pred_real, target_real)
            loss += F.mse_loss(pred_fake, target_fake)
        elif self.metric == "hinge":
            loss += F.relu(1 - pred_real).mean()
            loss += F.relu(1 + pred_fake).mean()
        elif self.metric == "ragan":
            loss += F.softplus(-1 * average_diff(pred_real, pred_fake)).mean()
            loss += F.softplus(average_diff(pred_fake, pred_real)).mean()
        elif self.metric == "rahinge":
            loss += F.relu(1 - average_diff(pred_real, pred_fake)).mean()
            loss += F.relu(1 + average_diff(pred_fake, pred_real)).mean()
        elif self.metric == "ralsgan":
            loss += torch.mean((average_diff(pred_real, pred_fake) - 1.0) ** 2)
            loss += torch.mean((average_diff(pred_fake, pred_real) + 1.0) ** 2)
        else:
            raise NotImplementedError
        return loss

    def loss_G(self, pred_real, pred_fake):
        loss = 0
        if self.metric == "nsgan":
            loss += F.softplus(-pred_fake).mean()
        elif self.metric == "wgan":
            loss += -pred_fake.mean()
        elif self.metric == "lsgan":
            target_real = self.label_real.expand_as(pred_fake)
            loss += F.mse_loss(pred_fake, target_real)
        elif self.metric == "hinge":
            loss += -pred_fake.mean()
        elif self.metric == "ragan":
            loss += F.softplus(average_diff(pred_real, pred_fake)).mean()
            loss += F.softplus(-1 * average_diff(pred_fake, pred_real)).mean()
        elif self.metric == "rahinge":
            loss += F.relu(1 + average_diff(pred_real, pred_fake)).mean()
            loss += F.relu(1 - average_diff(pred_fake, pred_real)).mean()
        elif self.metric == "ralsgan":
            loss += torch.mean((average_diff(pred_real, pred_fake) + 1.0) ** 2)
            loss += torch.mean((average_diff(pred_fake, pred_real) - 1.0) ** 2)
        else:
            raise NotImplementedError
        return loss
