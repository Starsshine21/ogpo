#!/usr/bin/env python3
"""Freeze replacement episodes in the original split slots; build bounded replay."""
import argparse
import gc
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import torch
from build_robotwin_multitask_streaming_replay import build_group, atomic_json, metadata_group_summary
from ogpo.replay import prepare_replay_from_config, save_replay
from train_udivl_critic import load_config

OUT = ROOT / 'outputs/ogpo/replays/mixed1000_bootstrap_v1'
REPLACEMENTS = {'place_cans_plasticbox': 'adjust_bottle', 'place_shoe': 'place_object_stand',
                'move_playingcard_away': 'press_stapler'}


def freeze():
    old = sorted((ROOT / 'outputs/robotwin_rollouts/guess1000_nohanging_placeshoe_v1').glob('*/shard_*/raw_rollouts/episode_*'))
    assert len(old) == 1000
    newroot = ROOT / 'outputs/robotwin_rollouts/other10_pi05_base_dense100_v1'
    replacement = {}
    for task in REPLACEMENTS.values():
        audit = json.loads((newroot / task / 'strict100_summary.json').read_text())
        assert audit['audit_passed'] and audit['episodes'] == 100
        paths = sorted((newroot / task).glob('shard_*/raw_rollouts/episode_*'))
        assert len(paths) == 100, (task, len(paths))
        replacement[task] = iter(paths)
    perm = torch.randperm(1000, generator=torch.Generator().manual_seed(27)).tolist()
    roles = {i: ('train' if n < 800 else 'validation' if n < 900 else 'heldout') for n, i in enumerate(perm)}
    rows = []
    for i, path in enumerate(old):
        task = path.parts[-4]
        source = next(replacement[REPLACEMENTS[task]]) if task in REPLACEMENTS else path
        meta = json.loads((source / 'meta.json').read_text())
        assert (source / 'frames.npz').is_file()
        assert not (meta['task_name'] == 'click_bell' and 7900000 <= int(meta['seed']) <= 7900049)
        rows.append(dict(episode_id=i, task=meta['task_name'], source=str(source.resolve()),
                         old_source=str(path), split=roles[i], success=bool(meta['success']), seed=meta['seed']))
    assert len({(r['task'], r['seed']) for r in rows}) == 1000
    tasks = sorted({r['task'] for r in rows})
    rng = random.Random(20260915)
    masks = {'schema_version': 1, 'p': 0.9, 'seed': 20260915, 'members': {}, 'statistics': {}}
    for member in range(5):
        masks['members'][str(member)] = {}
        masks['statistics'][str(member)] = {}
        for task in tasks:
            train = [r for r in rows if r['split'] == 'train' and r['task'] == task]
            included = [r['episode_id'] for r in train if rng.random() < 0.9]
            masks['members'][str(member)][task] = included
            masks['statistics'][str(member)][task] = dict(included=len(included), total=len(train), fraction=len(included)/len(train))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in [('split_manifest.json', {'episodes': rows, 'tasks': tasks, 'replacement_mapping': REPLACEMENTS}),
                          ('bootstrap_masks.json', masks)]:
        path = OUT / name
        if path.exists():
            assert json.loads(path.read_text()) == payload, f'refusing changed {path}'
        else:
            atomic_json(path, payload)
    print(json.dumps(masks['statistics'], indent=2), flush=True)


def build():
    manifest = json.loads((OUT / 'split_manifest.json').read_text())
    rows = manifest['episodes']
    paths = [Path(r['source']) for r in rows]
    config = load_config(ROOT / 'configs/ogpo/robotwin_multitask10_divl_vmean_headonly_taskoutcomebalanced_20k.yaml')
    summary = {}
    # Keep original slot-based DDP partition; masks are global, never resampled per rank.
    train = [r['episode_id'] for r in rows if r['split'] == 'train']
    for rank in range(4):
        ids = train[rank::4]
        path = OUT / f'train_rank{rank:02d}.pt'
        if path.exists():
            raise FileExistsError(path)
        batch = build_group(paths, ids, gamma=0.999, behavior_policy='pi05_pytorch_clean50_base')
        batch = prepare_replay_from_config(batch, config['data'])
        from ogpo.episode_bootstrap import MemberEpisodeBootstrapReplay
        sampler = MemberEpisodeBootstrapReplay(batch, masks=json.loads((OUT / 'bootstrap_masks.json').read_text()),
                                               task_names=manifest['tasks'])
        summary[str(rank)] = {'coverage': sampler.coverage, 'transitions': batch.batch_size}
        save_replay(batch, path)
        if rank == 0:
            save_replay(batch.index_select(torch.arange(8)), OUT / 'model_init_8.pt')
        atomic_json(OUT / 'build_progress.json', summary)
        print(f'built rank={rank} transitions={batch.batch_size}', flush=True)
        del sampler, batch
        gc.collect()
    from ogpo.zarr_replay import concat_chunk_batches
    val_parts = []
    for task in manifest['tasks']:
        eid = next(r['episode_id'] for r in rows if r['task'] == task and r['split'] == 'validation')
        val = prepare_replay_from_config(build_group(paths, [eid], gamma=0.999,
                 behavior_policy='pi05_pytorch_clean50_base'), config['data'])
        ids = torch.linspace(0, val.batch_size - 1, 8).round().long().unique()
        val_parts.append(val.index_select(ids))
        del val
        gc.collect()
    save_replay(concat_chunk_batches(val_parts), OUT / 'validation_80.pt')
    atomic_json(OUT / 'BUILD_COMPLETE.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    build() if args.build else freeze()
