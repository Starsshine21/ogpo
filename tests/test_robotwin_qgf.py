from types import SimpleNamespace
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from robotwin_qgf import unnormalize, decode_actions, guided_euler
from robotwin_qgf import sample_qgf


def test_euler_sign_and_zero_exact():
    x = torch.randn(1, 50, 32)
    v = torch.randn_like(x)
    dt = torch.tensor(-.1)
    assert torch.equal(guided_euler(x, v, dt, None, 0), x+dt*v)
    assert torch.allclose(guided_euler(x, v, dt, torch.ones_like(x), 2), x+dt*v+.2)


def test_quantile_degenerate_and_padding_grad():
    stats = SimpleNamespace(q01=np.array([1., 2.]), q99=np.array([3., 2.]))
    x = torch.zeros(1, 3, requires_grad=True)
    y = unnormalize(x, stats, True)
    torch.testing.assert_close(y, torch.tensor([[2.0000005, 2., 0.]]))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[1.0000005, 1., 1.]]))


def test_decode_matches_numpy_and_ignores_padding():
    from openpi.transforms import Unnormalize, AbsoluteActions, compose
    from openpi.policies.aloha_policy import AlohaOutputs
    from openpi.shared.normalize import NormStats
    stats = NormStats(mean=np.arange(14, dtype=np.float32), std=np.full(14, 2., dtype=np.float32))
    transform = compose([Unnormalize({'state': stats, 'actions': stats}),
                         AbsoluteActions([True]*6+[False]+[True]*6+[False]), AlohaOutputs(False)])
    x = torch.randn(1, 50, 32, requires_grad=True)
    state = torch.randn(1, 32, dtype=torch.float64)
    y = decode_actions(transform, state, x)
    assert y.dtype == torch.float32
    expected = transform({'state': state[0].numpy().copy(), 'actions': x[0].detach().numpy().copy()})['actions']
    np.testing.assert_allclose(y[0].detach().numpy(), expected, rtol=1e-6, atol=1e-6)
    y.sum().backward()
    assert torch.count_nonzero(x.grad[..., 14:]) == 0
    assert torch.count_nonzero(x.grad[..., :14]) == 700


def test_clean_action_gradient_without_velocity_jacobian():
    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
            language = SimpleNamespace(config=SimpleNamespace())
            self.paligemma_with_expert = SimpleNamespace(
                paligemma=SimpleNamespace(language_model=language), forward=lambda **kw: (None, None))
        def _preprocess_observation(self, obs, train=False):
            return None, None, None, None, obs.state
        def embed_prefix(self, *args):
            return torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool)
        def _prepare_attention_masks_4d(self, x):
            return x[:, None]
        def denoise_step(self, state, pad, cache, x, t):
            assert not torch.is_grad_enabled()
            return self.weight*x
    model = Model()
    obs = SimpleNamespace(state=torch.zeros(1, 32))
    x = torch.ones(1, 2, 32)
    audit = {}
    def q(a):
        return a.sum((1, 2))[None].expand(10, -1)
    result = sample_qgf(model, 'cpu', obs, x, q, lambda s, a: a[..., :14], 1., audit, num_steps=2)
    # Each half-step base Euler cancels x (v=2x); clean-gradient is +1,
    # not (1-2*t), and only on real action dimensions.
    torch.testing.assert_close(result[..., :14], torch.full((1, 2, 14), .5))
    assert torch.count_nonzero(result[..., 14:]) == 0
    assert model.weight.grad is None
    assert len(audit['steps']) == 2


def test_unknown_decode_transform_fails_closed():
    import pytest
    with pytest.raises(ValueError, match='unsupported QGF'):
        decode_actions(SimpleNamespace(transforms=[object()]), torch.zeros(1, 32), torch.zeros(1, 50, 32))


def test_deployment_quantiles_keep_actual_degenerate_scale():
    stats = SimpleNamespace(q01=np.array([2., 1.]), q99=np.array([2., 3.]))
    def deployment_method(x, stats):
        return np.concatenate(((x[:2]+1)/2*(stats.q99-stats.q01+1e-6)+stats.q01, x[2:]))
    x = torch.zeros(1, 3, requires_grad=True)
    y = unnormalize(x, stats, True, deployment_method)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[.0000005, 1.0000005, 1.]]))
