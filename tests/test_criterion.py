#!/usr/bin/env python3
import random
from contextlib import contextmanager
from unittest import TestCase, main

import numpy as np
import torch
from scipy import stats

from exact_pytorch import EXACTLoss


@contextmanager
def tmp_seed(seed):
    """Centext manager for temporary random seed (random and Numpy modules)."""
    state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield None
    finally:
        random.setstate(state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


class TestCriterion(TestCase):
    def test_exact(self):
        mean = torch.ones(2).double()
        concentrations = torch.ones(1).double()
        labels = torch.full([], 1).long()
        criterion = EXACTLoss(sample_size=1024)

        # 2 classes.
        targets = torch.tensor([
            [-1., 0.],
            [2., 0.]
        ]).double()
        bias = torch.tensor([0., 1.]).double()
        logits = torch.nn.functional.linear(mean, targets, bias)
        loss = criterion(logits * concentrations, labels)
        accuracy = (-loss).exp().item()
        accuracy_gt = stats.norm.cdf(4 / np.sqrt(2))
        self.assertAlmostEqual(accuracy, accuracy_gt, 5)

        self._test_gradients([logits, concentrations],
                             (lambda logits, concentrations:
                              criterion(logits * concentrations, labels)))

        # 3 classes.
        targets = torch.tensor([
            [-1., 0.],
            [2., 0.],
            [4., 1.]
        ]).double()
        bias = torch.tensor([0., 1., 0.]).double()
        self._test_gradients([logits, concentrations],
                             (lambda logits, concentrations:
                              criterion(logits * concentrations, labels)))

    def _test_gradients(self, parameters, loss_fn, eps=1e-3):
        placeholders = [torch.tensor(p.numpy(), requires_grad=True, dtype=torch.double) for p in parameters]
        with tmp_seed(0):
            loss_base = loss_fn(*placeholders)
        loss_base.backward()
        loss_base = loss_base.item()

        grad_norm = self._norm([p.grad for p in placeholders])
        updated_parameters = [p - p.grad * eps / grad_norm for p in placeholders]
        with tmp_seed(0):
            loss_update = loss_fn(*updated_parameters).item()
        self.assertTrue(loss_update < loss_base)

        with torch.no_grad():
            for i, p in enumerate(placeholders):
                shape = p.shape
                p_grad = p.grad.flatten()
                p = p.flatten()
                for j, v in enumerate(p):
                    delta_p = p.clone()
                    delta_p[j] += eps
                    if len(shape) > 1:
                        delta_p = delta_p.reshape(*shape)
                    delta_placeholders = list(placeholders)
                    delta_placeholders[i] = delta_p
                    with tmp_seed(0):
                        loss = loss_fn(*delta_placeholders).item()
                    grad = (loss - loss_base) / eps
                    grad_gt = p_grad[j].item()
                    self.assertAlmostEqual(grad, grad_gt, delta=0.05)

    def _norm(self, parameters):
        return np.sqrt(np.sum([p.square().sum().item() for p in parameters]))

    def _random_update(self, parameters, eps):
        update = [torch.randn_like(p) for p in parameters]
        norm = self._norm(update)
        new_parameters = [p + eps * p_update / norm for p, p_update in zip(parameters, update)]
        return new_parameters


if __name__ == "__main__":
    main()
