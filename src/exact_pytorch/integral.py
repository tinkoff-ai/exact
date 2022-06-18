"""Genz integration algorithm for the covariance matrix with fixed diagonal and non-diagonal values."""

import math
import numpy as np
import torch


def non_diag(a):
    """Get non-diagonal elements of matrices.

    Args:
        a: Matrices tensor with shape (..., N, N).

    Returns:
        Non-diagonal elements with shape (..., N, N - 1).
    """
    n = a.shape[-1]
    prefix = list(a.shape)[:-2]
    return a.reshape(*(prefix + [n * n]))[..., :-1].reshape(*(prefix + [n - 1, n + 1]))[..., 1:].reshape(*(prefix + [n, n - 1]))


def get_cholesky(dim, diag=2, alt=1, device=None, dtype=None):
    """Get diagonal and non-diagonal elements of the Cholesky decomposition.

    Cholesky matrix has the form
    D_1  0  ...  0
    A_1 D_2 ...  0
    A_1 A_2 ...  0
    ...
    A_1 A_2 ... D_n
    """
    delta = diag - alt
    l_diag = np.empty(dim)
    l_alt = np.empty(dim)
    s = 0.0
    for i in range(dim):
        l_diag[i] = math.sqrt(diag - s)
        l_alt[i] = l_diag[i] - delta / l_diag[i]
        s += l_alt[i] ** 2
    l_diag = torch.tensor(l_diag, dtype=dtype, device=device)
    l_alt = torch.tensor(l_alt, dtype=dtype, device=device)
    return l_diag, l_alt


def cholesky_matrix(l_diag, l_alt):
    """Create Cholesky decomposition matrix from diagonal and non-diagonal
    elements of this matrix."""
    l = torch.diag_embed(l_diag)  # (D, D).
    for i in range(len(l_diag) - 1):
        l[i + 1:, i] = l_alt[i]
    return l


def genz_integral_impl(mean, cov_diag : int = 2, cov_alt : int = 1, n : int = 10):
    """Compute multivariate density in the orthant > 0 with covariance matrix equal to I + 1.

    Args:
        mean: Distribution mean with shape (B, D).

    Returns:
        Tuple of integral values with shape (B) and gradients with shape (B, D).
    """
    SQ2 = math.sqrt(2)
    NORMAL_NORM = math.sqrt(2 / math.pi)

    b, dim = mean.shape
    sample = torch.rand(dim, n, b, 1, dtype=mean.dtype, device=mean.device)  # (D, N, B, D).
    mean = -mean.T.reshape(dim, 1, b, 1)  # (D, N, B, D).

    l_diag, l_alt = get_cholesky(dim, diag=cov_diag, alt=cov_alt, dtype=mean.dtype, device=mean.device)
    l_diag_sq = l_diag * SQ2

    # Create buffers.
    ds_arg = torch.zeros(dim, n, b, dim + 1, dtype=mean.dtype, device=mean.device)  # (D, N, B, D + 1).
    ds = torch.empty_like(ds_arg)  # (D, N, B, D + 1).
    es = torch.ones(1, dtype=mean.dtype, device=mean.device)  # (1).
    y_sums = torch.zeros_like(ds[0])  # (N, B, D + 1).

    ds_arg[0] = mean[0] / l_diag_sq[0]
    ds[0] = torch.erf(ds_arg[0])

    for i in range(1, dim):
        interp = torch.lerp(ds[i - 1], es, sample[i - 1])
        interp[..., i - 1] = ds[i - 1, ..., i - 1]
        y = (torch.erfinv(interp) * SQ2).clip(-5, 5)  # (N, B, D + 1).
        y_sums += y * l_alt[i - 1]
        ds_arg[i] = (mean[i] - y_sums) / l_diag_sq[i]  # (...).
        ds[i] = torch.erf(ds_arg[i])  # (...).
    deltas = 1 - ds
    for i in range(dim):
        deltas[i, ..., i] = NORMAL_NORM / l_diag[i] * (-(ds_arg[i, ..., i]).square()).exp()
    deltas /= 2
    sums = deltas.prod(dim=0).mean(0)  # (B, D + 1).
    integrals = sums[:, -1]
    gradients = sums[:, :-1]
    return integrals, gradients


def integral(mean, n=16, reorder=True):
    """Compute multivariate density in the orthant > 0 with covariance matrix equal to I + 1.

    Args:
        mean: Mean tensors with shape (..., D).
        n: Sample size.
        reorder: Whether to use reordered integration or not.

    Returns:
        Integral values with shape (...) if mean_grad is False and gradient values with shape (..., D) otherwise.
    """
    mean = mean.detach()
    prefix = list(mean.shape[:-1])
    b = int(np.prod(prefix))
    dim = mean.shape[-1]

    mean = mean.reshape(b, dim)  # (B, D).

    if reorder:
        mean, order = mean.sort(dim=-1)  # (B, D), (B, D).
    else:
        order = torch.arange(dim, dtype=mean.dtype, device=mean.device).reshape([1, dim])  # (B, D).

    integrals, gradients = genz_integral_impl(mean, n=n)

    if reorder:
        # Sums: (B, D).
        iorder = order.argsort(dim=-1)  # (B, D).
        gradients = gradients.take_along_dim(iorder, -1)

    integrals = integrals.reshape(*(prefix or [[]]))
    gradients = gradients.reshape(*(prefix + [dim]))
    return integrals, gradients


class PositiveNormalProb(torch.autograd.Function):
    """Compute probability of all elements being more than zero.

    Orthant integration is preformed for the normal distribution with
    covariance matrix of the special form. All diagonal elements of
    the covariance matrix are equal to 2. Non-diagonal elements are
    equal to 1.

    Probability is differentiable w.r.t. the distribution mean.

    """
    @staticmethod
    def forward(self, mean, n=16):
        with torch.no_grad():
            integrals, gradients = integral(mean, n=n)
        self.save_for_backward(integrals, gradients)
        return integrals

    @staticmethod
    def backward(self, grad_output):
        integrals, gradients = self.saved_tensors
        return gradients * grad_output.unsqueeze(-1), None
