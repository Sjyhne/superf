import torch
import torch.nn as nn


class BasicLosses(nn.Module):
    """Basic regression loss functions."""

    @staticmethod
    def mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Mean Absolute Error (L1) Loss."""
        if mask is None:
            return torch.abs(pred - target).mean()
        else:
            return (torch.abs(pred - target) * mask).mean()

    @staticmethod
    def mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Mean Squared Error (L2) Loss."""
        if mask is None:
            return torch.square(pred - target).mean()
        else:
            return (torch.square(pred - target) * mask).mean()

    @staticmethod
    def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Root Mean Squared Error Loss."""
        return torch.sqrt(torch.square(pred - target).mean())
