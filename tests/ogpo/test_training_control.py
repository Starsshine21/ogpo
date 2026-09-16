import torch
import torch.nn as nn

from ogpo.training_control import TrainableSnapshot, ValidationEarlyStopper


def test_validation_early_stopper_uses_patience_and_min_delta():
    stopper = ValidationEarlyStopper(mode="min", patience=2, min_delta=0.1)

    assert stopper.update(1.0)
    assert not stopper.update(0.95)
    assert not stopper.should_stop
    assert stopper.update(0.8)
    assert not stopper.update(0.75)
    assert not stopper.update(0.76)
    assert stopper.should_stop


def test_trainable_snapshot_restores_only_trainable_parameters():
    module = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    module[0].requires_grad_(False)
    original_frozen = module[0].weight.detach().clone()
    original_trainable = module[1].weight.detach().clone()
    snapshot = TrainableSnapshot.capture(module)

    with torch.no_grad():
        module[0].weight.add_(10.0)
        module[1].weight.add_(10.0)
    snapshot.restore(module)

    assert not torch.equal(module[0].weight, original_frozen)
    assert torch.equal(module[1].weight, original_trainable)
