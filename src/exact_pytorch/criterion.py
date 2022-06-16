import torch
from .integral import PositiveNormalProb


class EXACTLoss(torch.nn.Module):
    """Implementation of the EXACT loss for accuracy optimization.

    See [EXACT: How to Train Your Accuracy (2022)](https://arxiv.org/pdf/2205.09615.pdf) for more details.
    """
    def __init__(self, sample_size=16):
        super().__init__()
        self._sample_size = sample_size

    def __call__(self, logits, labels):
        """Compute expected error rate.

        Args:
            logits: Logits tensor with shape (..., C).
            labels: Ground truth classes with shape (...).

        Returns:
            Expected error rate.
        """
        prefix = list(labels.shape)
        num_classes = logits.shape[-1]

        mean = logits.take_along_dim(labels.unsqueeze(-1), -1) - logits
        mask = torch.ones_like(mean, dtype=torch.bool)
        mask.scatter_(-1, labels.unsqueeze(-1), False)
        mean = mean[mask].reshape(*(prefix + [num_classes - 1]))
        probs = PositiveNormalProb.apply(mean, self._sample_size)
        return 1 - probs.mean()  # Error rate.
