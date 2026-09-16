"""Regenerate candidates from verified reconstructed observations, common to all critics."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from evaluate_ogpo_critic_v2 import ROOT, select_candidate_cache
from evaluate_multitask_ogpo_critic_readiness import load_pi05_pytorch_flow_policy, TASKS, atomic_torch
from train_udivl_critic import load_config
from prepare_carl import sha256


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    source, output = args.source_root.resolve(), args.output_root.resolve()
    m = json.loads((source / 'candidate_manifest.json').read_text())
    if sha256(m['source_candidate_cache']) != m['candidate_cache_sha256'] or sha256(m['base_actor_checkpoint']) != m['base_actor_sha256']:
        raise ValueError('source actor/cache changed')
    batch, _, _ = select_candidate_cache(Path(m['source_candidate_cache']), m['states_per_task'], m['seed'])
    images, wrists, states = [], [], []
    for row in m['states']:
        refpath = source / 'reconstructed_reference' / f'state_{row["global_state_index"]:04d}.json'
        ref = json.loads(refpath.read_text())
        if not ref['repeat_verified'] or ref['state_id'] != row['stable_state_id']:
            raise ValueError('unverified reconstructed observation')
        if sha256(refpath.with_suffix('.npz')) != ref['images_sha256']:
            raise ValueError('reference observation hash mismatch')
        with np.load(refpath.with_suffix('.npz')) as arrays:
            images.append(arrays['image']); wrists.append(arrays['wrist_image']); states.append(arrays['state'])
    state_tensor = torch.from_numpy(np.stack(states)).to(batch.observations.dtype)
    batch = replace(batch, images={'image_base': torch.from_numpy(np.stack(images)),
                                  'image_wrist': torch.from_numpy(np.stack(wrists))},
                    observations=state_tensor, proprioceptions=state_tensor.clone(),
                    critic_features=None, next_critic_features=None)
    config = load_config(ROOT / 'configs/ogpo/robotwin_click_bell_C8k_mt_headonly_actor_bs4_lr2e6_4k.yaml')
    flow, actor = config['flow'], config['actor']
    policy = load_pi05_pytorch_flow_policy(
        checkpoint_dir=Path(flow['checkpoint_dir']), train_config_name=str(flow['train_config']),
        image_mapping=dict(flow.get('image_mapping_override', flow.get('image_mapping', {}))),
        image_container_key=flow.get('image_container_key'), transpose_images_to_chw=bool(flow.get('transpose_images_to_chw', False)),
        environment_action_dim=batch.action_dim, num_steps=int(flow.get('num_steps', 10)),
        stochastic_variance=float(flow.get('stochastic_variance', .04)), sde_mode=str(flow.get('sde_mode', 'gaussian_adapter')),
        constant_noise_std=float(flow.get('constant_noise_std', .005)), learn_sde_std=bool(flow.get('learn_sde_std', True)),
        randn_clip_value=float(flow.get('randn_clip_value', 3)), residual_hidden_dim=int(actor.get('hidden_dim', 128)),
        residual_enabled=bool(actor.get('residual_enabled', True)), backend_train_mode='none', device='cuda')
    policy.eval()
    generator = torch.Generator(device='cuda').manual_seed(m['seed'])
    candidates = []
    for i in range(batch.batch_size):
        condition = policy.condition_from_batch(batch.index_select(torch.tensor([i])))
        result = policy.rollout(condition, group_size=4, generator=generator)
        actions = policy.flat_actions_to_environment(result.endpoint, policy.repeat_condition(condition, 4))
        candidates.append(actions.reshape(1, 4, batch.generated_horizon, batch.action_dim).cpu())
        print(f'live_candidate_generation={i+1}/{batch.batch_size}', flush=True)
    manifest = [{'task': row['task'], 'episode_id': row['episode_id'], 'timestep': row['state_index'],
                 'source_transition_index': row['source_transition_index']} for row in m['states']]
    cache = source / 'reconstructed_candidate_cache.pt'
    if cache.exists():
        raise FileExistsError(cache)
    atomic_torch(cache, {'schema_version': 1, 'seed': m['seed'], 'group_size': 4,
                        'count_per_task': m['states_per_task'], 'task_order': list(TASKS),
                        'manifest': manifest, 'batch': asdict(batch), 'candidate_actions': torch.cat(candidates),
                        'base_actor_checkpoint': m['base_actor_checkpoint'],
                        'protocol': 'reconstructed observations, unchanged episode/timestep selection; fresh deterministic candidates'})
    subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_carl.py'), '--output-dir', str(output),
                    '--candidate-cache', str(cache), '--states-per-task', str(m['states_per_task']),
                    ], check=True)


if __name__ == '__main__':
    main()
