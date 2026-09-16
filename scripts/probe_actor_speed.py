"""Isolated actor throughput probe; never saves or replaces actor checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def diagnostic_kl_call(call, *, beta: float, enabled: bool):
    with torch.set_grad_enabled(torch.is_grad_enabled() and not (enabled and beta == 0.0)):
        return call()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--microbatch', type=int, choices=[1, 2, 4], required=True)
    parser.add_argument('--diagnostic-kl-no-grad', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    import train_full_ogpo as entry
    from ogpo import trainer
    from train_udivl_critic import load_config

    random.seed(20260916)
    np.random.seed(20260916)
    torch.manual_seed(20260916)
    torch.cuda.manual_seed_all(20260916)
    cfg = load_config(root / 'configs/ogpo/robotwin_mixed1000_catq9k_actor_2k.yaml')
    cfg['actor']['gradient_microbatch_size'] = args.microbatch
    # Keep post-update validation microbatch identical across variants.
    cfg['actor']['kl_eval_microbatch_size'] = 1
    cfg['training'].update({
        'actor_steps': args.steps, 'resume_checkpoint': None,
        'actor_start_step': 0, 'checkpoint_interval': 0,
        'save_final_checkpoint': False,
        'metrics_path': str(args.output / 'metrics.jsonl'),
        'tensorboard_dir': None,
        'config_snapshot_path': str(args.output / 'resolved_config.yaml'),
        'periodic_checkpoint_dir': str(args.output / 'unused_checkpoints'),
        'checkpoint_path': str(args.output / 'DO_NOT_SAVE.pt'),
    })
    config_path = args.output / 'probe_config.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    original_kl = trainer._policy_kl_device_aware
    trainer._policy_kl_device_aware = lambda *a, **kw: diagnostic_kl_call(
        lambda: original_kl(*a, **kw),
        beta=float(cfg.get('regularization', {}).get('beta_kl', 0.01)),
        enabled=args.diagnostic_kl_no_grad,
    )
    original_update = entry.full_actor_update
    original_advantages = trainer.conservative_advantages_for_candidates
    index = 0
    records = []
    def audited_advantages(*a, **kw):
        candidates = a[2].detach().float().cpu().numpy()
        np.save(args.output / f'candidates_{index}.npy', candidates)
        result = original_advantages(*a, **kw)
        np.save(args.output / f'advantages_{index}.npy', result[0].detach().float().cpu().numpy())
        return result
    trainer.conservative_advantages_for_candidates = audited_advantages

    def measured_update(state, batch, config, **kwargs):
        nonlocal index
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        original_step = state.actor_optimizer.step
        def measured_step(*a, **kw):
            # Capture small deterministic samples from every trainable tensor.
            # This is a parity diagnostic, not a full-gradient equivalence proof.
            samples = [p.grad.detach().flatten()[:8].float().cpu()
                       for p in state.policy.parameters() if p.grad is not None]
            if samples:
                np.save(args.output / f'gradient_sample_{index}.npy', torch.cat(samples).numpy())
            return original_step(*a, **kw)
        state.actor_optimizer.step = measured_step
        started = time.perf_counter()
        try:
            metrics = original_update(state, batch, config, **kwargs)
        finally:
            state.actor_optimizer.step = original_step
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        record = {
            'index': index, 'microbatch': args.microbatch,
            'diagnostic_kl_no_grad': args.diagnostic_kl_no_grad,
            'update_seconds': time.perf_counter() - started,
            'peak_allocated_gib': [torch.cuda.max_memory_allocated(i) / 2**30
                                   for i in range(torch.cuda.device_count())],
            'peak_reserved_gib': [torch.cuda.max_memory_reserved(i) / 2**30
                                  for i in range(torch.cuda.device_count())],
            'task': str(batch.task_ids[0]),
            'action_sha256': hashlib.sha256(batch.action_chunks.detach().cpu().numpy().tobytes()).hexdigest(),
        }
        records.append(record)
        with (args.output / 'timing.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print('SPEED_PROBE', json.dumps(record), flush=True)
        index += 1
        return metrics
    entry.full_actor_update = measured_update
    sys.argv = ['train_full_ogpo.py', '--config', str(config_path)]
    start = time.perf_counter()
    try:
        entry.main()
    except Exception as exc:
        (args.output / 'status.json').write_text(json.dumps({
            'status': 'failed', 'error': repr(exc), 'elapsed_seconds': time.perf_counter()-start,
        }, indent=2))
        raise
    (args.output / 'status.json').write_text(json.dumps({
        'status': 'complete', 'elapsed_seconds': time.perf_counter()-start,
        'steady_update_seconds_mean': float(np.mean([r['update_seconds'] for r in records[1:]])),
        'note': 'First update excluded; timings include identical diagnostic I/O in every variant.',
    }, indent=2))


if __name__ == '__main__':
    main()
