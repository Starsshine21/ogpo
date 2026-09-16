"""Opt-in collectives for equal-size actor replicas with complete G groups."""
import math

import torch
import torch.distributed as dist


def validate_config(config):
    actor = config.get('actor', {})
    if not actor.get('role_data_parallel', False):
        return False
    if not dist.is_initialized() or dist.get_world_size() != 2:
        raise ValueError('role_data_parallel requires exactly two initialized replicas')
    if (actor.get('ogpo_variant') != 'ca_chi2'
            or actor.get('advantage_mode') != 'conservative'
            or actor.get('full_ratio_mode') != 'ais_joint'
            or int(actor.get('actor_epochs_per_rollout', 1)) != 1
            or actor.get('post_update_kl_action') != 'rollback_cpu'):
        raise ValueError('role DP supports one-epoch conservative full-chain CA+ChiPO with rollback only')
    if any(float(config.get('regularization', {}).get(k, 0)) != 0
           for k in ('lambda_fm', 'lambda_success')):
        raise ValueError('role DP currently requires BC/FM disabled')
    return True


def gather_equal(tensor, dim=0):
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, tensor.contiguous())
    return torch.cat(parts, dim=dim)


def global_chipo(ca, q, ratio, **kwargs):
    from .chi2_regularization import apply_chipo_to_ca_advantage
    count = ca.shape[0]
    all_ca, all_q, all_ratio = gather_equal(ca), gather_equal(q, dim=1), gather_equal(ratio)
    advantages, stats = apply_chipo_to_ca_advantage(all_ca, all_q, all_ratio, **kwargs)
    start = dist.get_rank() * count
    return advantages[start:start + count], stats


@torch.no_grad()
def average_gradients(parameters, bucket_bytes=16*1024*1024):
    parameters = [p for p in parameters if p.requires_grad]
    if not parameters:
        raise ValueError('No trainable actor parameters')
    present = torch.tensor([p.grad is not None for p in parameters],
                           device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    # Bound communication storage even for a parameter larger than a bucket.
    for p, has_grad in zip(parameters, present.tolist(), strict=True):
        if not has_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        flat = p.grad.view(-1)
        size = max(1, bucket_bytes // flat.element_size())
        for start in range(0, flat.numel(), size):
            chunk = flat[start:start+size]
            dist.all_reduce(chunk, op=dist.ReduceOp.SUM)
            chunk.div_(dist.get_world_size())


def global_validation(kl, invalid, device):
    finite = math.isfinite(kl)
    value = torch.tensor(kl if finite else 0., device=device, dtype=torch.float64)
    bad = torch.tensor(bool(invalid) or not finite, device=device, dtype=torch.int32)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    return float(value.item() / dist.get_world_size()), bool(bad.item())
