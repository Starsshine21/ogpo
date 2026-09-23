import math

import pytest
import torch

from ogpo.trainer import adaptive_success_bc_lambda


def _combined_fraction(a: float, b: float, cosine: float, coefficient: float) -> float:
    total = math.sqrt(
        a * a
        + coefficient * coefficient * b * b
        + 2.0 * coefficient * a * b * cosine
    )
    return coefficient * b / total


def test_orthogonal_gradient_cap_is_twenty_percent() -> None:
    coefficient, _ = adaptive_success_bc_lambda(
        1.0, 1.0, 0.0, lambda0=1.0, rho=0.2
    )
    assert coefficient == pytest.approx(0.2 / math.sqrt(1.0 - 0.2**2))
    assert _combined_fraction(1.0, 1.0, 0.0, coefficient) == pytest.approx(0.2)


def test_large_bc_gradient_reduces_nominal_lambda() -> None:
    coefficient, cap = adaptive_success_bc_lambda(
        1.0, 100.0, 0.0, lambda0=0.005, rho=0.2
    )
    assert coefficient == pytest.approx(cap)
    assert coefficient < 0.005
    assert _combined_fraction(1.0, 100.0, 0.0, coefficient) == pytest.approx(0.2)


def test_small_bc_gradient_keeps_nominal_lambda() -> None:
    coefficient, cap = adaptive_success_bc_lambda(
        1.0, 0.01, 0.0, lambda0=0.005, rho=0.2
    )
    assert cap > 0.005
    assert coefficient == pytest.approx(0.005)


def test_two_backward_components_produce_finite_gradient_and_one_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))

    class CountingSGD(torch.optim.SGD):
        def __init__(self, params):
            super().__init__(params, lr=0.1)
            self.update_count = 0

        def step(self, closure=None):
            self.update_count += 1
            return super().step(closure)

    optimizer = CountingSGD([parameter])
    optimizer.zero_grad(set_to_none=True)
    flash_loss = parameter[0]
    flash_loss.backward()
    flash_gradient = parameter.grad.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    success_bc_loss = 100.0 * parameter[1]
    success_bc_loss.backward()
    bc_gradient = parameter.grad.detach().clone()
    a = float(flash_gradient.norm().item())
    b = float(bc_gradient.norm().item())
    cosine = float(torch.dot(flash_gradient, bc_gradient).item()) / (a * b)
    coefficient, _ = adaptive_success_bc_lambda(
        a, b, cosine, lambda0=0.005, rho=0.2
    )

    parameter.grad = flash_gradient + coefficient * bc_gradient
    assert torch.isfinite(parameter.grad).all()
    assert coefficient * b / float(parameter.grad.norm().item()) == pytest.approx(0.2)
    optimizer.step()
    assert optimizer.update_count == 1
