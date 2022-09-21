import torch
from .integral import PositiveNormalProb


class EXACTLoss(torch.nn.Module):
    """Implementation of the EXACT loss for accuracy optimization.

    Returns expected error rate.

    See [EXACT: How to Train Your Accuracy (2022)](https://arxiv.org/pdf/2205.09615.pdf) for more details.

    Args:
        reduction: Output error reduction type (`mean` or `none`).
        sample_size: Sample size used for integration.
        disable_batch_norm: Don't use batch normalization if STD is provided.
        margin: Logits margin used to truncate error.
    """
    def __init__(self, reduction="mean", sample_size=16,
                 disable_batch_norm=False, margin=None):
        if reduction not in ["mean", "none"]:
            raise ValueError("Unknown aggregation: {}".format(reduction))
        super().__init__()
        self._reduction = reduction
        self._sample_size = sample_size
        self._disable_batch_norm = disable_batch_norm
        self._margin = margin

    def __call__(self, logits, labels, std=None):
        """Compute loss.

        Args:
            logits: Logits tensor with shape (..., C).
            labels: Labels tensor with shape (...).
            std: Logits std to perform scaling after margin.

        Returns:
            Loss value with the shape depending on the reduction method.
        """
        prefix = list(labels.shape)
        num_classes = logits.shape[-1]

        if (std is not None) and (not self._disable_batch_norm):
            logits = (logits - logits.mean()) / logits.std()  # Batch normalization.
        mean = logits.take_along_dim(labels.unsqueeze(-1), -1) - logits
        mask = torch.ones_like(mean, dtype=torch.bool)
        mask.scatter_(-1, labels.unsqueeze(-1), False)
        mean = mean[mask].reshape(*(prefix + [num_classes - 1]))  # (..., C - 1).
        if self._margin is not None:
            if std is None:
                raise RuntimeError("Margin is only applicable with explicit STD paramater.")
            mean = torch.clip(mean, max=self._margin)
        if std is not None:
            mean = mean / std
        probs = PositiveNormalProb.apply(mean, self._sample_size)  # (...).
        if self._reduction == "mean":
            return 1 - probs.mean()
        else:
            assert self._reduction == "none"  # Check in init.
            return 1 - probs
