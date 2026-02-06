import torch
import torch.nn as nn


class BasicLosses(nn.Module):
    """L1, L2, RMSE with optional masking."""

    @staticmethod
    def mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return torch.abs(pred - target).mean()
        return (torch.abs(pred - target) * mask).mean()

    @staticmethod
    def mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return torch.square(pred - target).mean()
        return (torch.square(pred - target) * mask).mean()

    @staticmethod
    def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.square(pred - target).mean())
