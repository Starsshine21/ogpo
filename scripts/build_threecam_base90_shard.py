#!/usr/bin/env python3
from __future__ import annotations

import argparse, gc, json, random, sys
from dataclasses import fields
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from build_robotwin_critic_replay import _episode_rows
from train_udivl_critic import load_config
from ogpo.replay import add_monte_carlo_returns, prepare_replay_from_config
from ogpo.zarr_replay import rows_to_chunk_batch

def main():
 p=argparse.ArgumentParser(); p.add_argument('--rank',type=int,required=True); p.add_argument('--world-size',type=int,default=4); p.add_argument('--split-manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--split',choices=['train','heldout'],default='train'); a=p.parse_args()
 manifest=json.loads(a.split_manifest.read_text()); records=manifest[a.split]
 if a.split=='train': records=records[a.rank::a.world_size]
 rows=[]
 for eid,r in enumerate(records):
  rows.extend(_episode_rows(Path(r['source']),episode_id=eid,gamma=.999,behavior_policy='pi05_pytorch_clean50_base_threecam'))
 batch=add_monte_carlo_returns(rows_to_chunk_batch(rows)); del rows; gc.collect()
 cfg=load_config(ROOT/'configs/ogpo/threecam_flash_actor.yaml')
 prepared=prepare_replay_from_config(batch,cfg['data'])
 if set(prepared.images or {}) != {'image_base','image_left_wrist','image_right_wrist'}: raise ValueError('not three-camera replay')
 payload={f.name:getattr(prepared,f.name) for f in fields(prepared)}
 a.output.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,a.output)
 a.output.with_suffix(a.output.suffix+'.json').write_text(json.dumps({'split':a.split,'rank':a.rank,'episodes':len(records),'transitions':prepared.batch_size,'camera_keys':sorted(prepared.images)},indent=2)+'\n')
 print(a.output,flush=True)
if __name__=='__main__': main()
