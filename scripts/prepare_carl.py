"""Create an offline CARL manifest from an existing fixed base-candidate cache."""
import argparse
import hashlib
from pathlib import Path
import torch
import evaluate_ogpo_critic_v2 as v2

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate-cache',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--states-per-task',type=int,default=10)
    p.add_argument('--terminal-replay',type=Path)
    p.add_argument('--logged-replay',type=Path)
    args=p.parse_args()
    cache=torch.load(args.candidate_cache,map_location='cpu',weights_only=False)
    v2.TASKS=tuple(cache['task_order'])
    batch,actions,records=v2.select_candidate_cache(args.candidate_cache,args.states_per_task,int(cache['seed']))
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    m={'protocol':'CARL v2 Sparse Reward Policy Alignment','tasks':list(v2.TASKS),
       'source_candidate_cache':str(args.candidate_cache.resolve()),'candidate_cache_sha256':sha256(args.candidate_cache),
       'seed':int(cache['seed']),'states_per_task':args.states_per_task,'records':records,
       'base_actor_checkpoint':cache.get('base_actor_checkpoint')}
    for k in ['terminal_replay','logged_replay']:
        if getattr(args,k):m[k]=str(getattr(args,k).resolve())
    v2.atomic_json(out/'candidate_manifest.json',m)
    print('OFFLINE_CANDIDATE_BANK_READY',out,flush=True)

if __name__=='__main__':main()
