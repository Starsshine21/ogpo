"""Isolated one/two-replica full-chain actor smoke. No checkpoint writes."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world not in (1, 2):
        raise ValueError('Only one or two four-card replicas supported')
    torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 4
    for i in range(4):
        assert torch.ones(1, device=f'cuda:{i}').item() == 1
        free, total = torch.cuda.mem_get_info(i)
        print('GPU', rank, i, torch.cuda.get_device_name(i), free/2**30, total/2**30, flush=True)
        if free < total - 2*2**30:
            raise RuntimeError('Allocated GPU already occupied; refusing smoke')
    if world == 2:
        dist.init_process_group('nccl', timeout=timedelta(minutes=40))
    output = args.output / f'rank{rank}'
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    import train_full_ogpo as entry
    from train_udivl_critic import load_config
    random.seed(20260916); np.random.seed(20260916); torch.manual_seed(20260916)
    torch.cuda.manual_seed_all(20260916)
    cfg = load_config(root/'configs/ogpo/robotwin_mixed1000_catq9k_actor_2k.yaml')
    cfg['actor']['role_data_parallel'] = world == 2
    cfg['actor']['gradient_microbatch_size'] = 1
    cfg['training'].update(actor_steps=args.steps, resume_checkpoint=None, actor_start_step=0,
        checkpoint_interval=0, save_final_checkpoint=False,
        metrics_path=str(output/'metrics.jsonl'), tensorboard_dir=None,
        config_snapshot_path=str(output/'resolved_config.yaml'),
        checkpoint_path=str(output/'DO_NOT_SAVE.pt'), periodic_checkpoint_dir=str(output/'unused'))
    path = output/'config.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    original = entry.full_actor_update
    counter = 0

    def measured(state, batch, config, **kwargs):
        nonlocal counter
        assert batch.batch_size == 4
        if counter == 0:
            # Initial model seed is shared; rollout random streams are independent.
            torch.manual_seed(10100 + rank)
            torch.cuda.manual_seed_all(10100 + rank)
        digest = hashlib.sha256(batch.action_chunks.cpu().numpy().tobytes()).hexdigest()
        if world == 2:
            digests = [None, None]
            dist.all_gather_object(digests, digest)
            assert digests[0] == digests[1], 'Global sampled batch differs across ranks'
            indices = torch.arange(rank*2, (rank+1)*2)
            batch = batch.index_select(indices)
            kwargs = {k: (v.index_select(indices) if v is not None else None)
                      for k, v in kwargs.items()}
            dist.barrier()
        for i in range(4):
            torch.cuda.synchronize(i); torch.cuda.reset_peak_memory_stats(i)
        start = time.perf_counter()
        metrics = original(state, batch, config, **kwargs)
        for i in range(4):
            torch.cuda.synchronize(i)
        elapsed = time.perf_counter() - start
        seconds = torch.tensor(elapsed, device='cuda:0', dtype=torch.float64)
        if world == 2:
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        # Sample every trainable parameter to catch replica update divergence.
        sample = torch.cat([p.detach().flatten()[:8].float().to('cuda:0')
                            for p in state.policy.parameters() if p.requires_grad])
        replica_delta = 0.
        if world == 2:
            peer = [torch.empty_like(sample), torch.empty_like(sample)]
            dist.all_gather(peer, sample)
            replica_delta = float((peer[0]-peer[1]).abs().max())
            if replica_delta > 1e-6:
                raise RuntimeError(f'Replica parameter samples diverged: {replica_delta}')
        record = dict(iteration=counter, rank=rank, replicas=world,
            global_state_batch=4, local_state_batch=batch.batch_size, candidate_group=4,
            global_batch_actions_sha256=digest, slowest_update_seconds=float(seconds),
            peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(4)],
            peak_reserved_gib=[torch.cuda.max_memory_reserved(i)/2**30 for i in range(4)],
            replica_parameter_sample_max_abs_diff=replica_delta,
            actor_grad_norm=metrics['actor_grad_norm'],
            global_post_update_reference_kl=metrics['post_update_reference_kl'],
            accepted=metrics['actor_update_accepted'])
        with (output/'timing.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print('ROLE_DP_SMOKE', json.dumps(record), flush=True)
        counter += 1
        return metrics

    entry.full_actor_update = measured
    sys.argv = ['train_full_ogpo.py', '--config', str(path)]
    try:
        entry.main()
        (output/'COMPLETE.json').write_text(json.dumps({'steps':counter, 'replicas':world}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
