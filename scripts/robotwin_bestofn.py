"""Online chunk-level best-of-N using the unchanged RoboTwin base sampler."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def candidate_noise(seed,chunk_id,candidate_id,horizon,model_dim):
    # Candidate zero is bit-for-bit the existing base evaluation noise protocol.
    entropy=[int(seed),int(chunk_id)]
    if candidate_id:entropy += [int(candidate_id)]
    return np.random.default_rng(np.random.SeedSequence(entropy)).standard_normal((horizon,model_dim),dtype=np.float32)

def choose_qmean(raw_q):
    q=np.asarray(raw_q)
    if q.ndim!=2 or q.shape[1]!=10 or not np.isfinite(q).all():
        raise ValueError('expected finite raw Q [candidates,10]')
    return int(q.mean(axis=1).argmax())

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

class BestOfN:
    def __init__(self,checkpoint,config,output,base_checkpoint,group_size=4,device='cuda',verify_base=False):
        from train_udivl_critic import load_config
        from ogpo.replay import load_replay
        from ogpo.trainer import build_train_state,load_critic_checkpoint
        self.checkpoint=Path(checkpoint).resolve();self.config_path=Path(config).resolve()
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.group_size=int(group_size);self.verify_base=verify_base;self.base_verified=False
        if self.group_size<1:raise ValueError('group_size must be positive')
        cfg=load_config(self.config_path)
        init=load_replay(ROOT/cfg['data']['distributed_model_init_path'])
        state=build_train_state(cfg,init,device=device)
        payload=load_critic_checkpoint(self.checkpoint,state,load_optimizer=False)
        checkpoint_step=int(payload['training_step'])
        del payload
        self.critic=state.critic.eval().requires_grad_(False)
        self.template=init.index_select(torch.tensor([0]));self.device=device
        del state
        spec={'protocol':'base-bestofn-raw10mean-v1','critic_checkpoint':str(self.checkpoint),
            'critic_training_step':checkpoint_step,
            'critic_sha256':file_hash(self.checkpoint),'critic_config':str(self.config_path),
            'config_sha256':file_hash(self.config_path),'base_checkpoint':str(Path(base_checkpoint).resolve()),
            'base_sha256':file_hash(base_checkpoint),'group_size':self.group_size,
            'candidate_0_noise':'PCG64 SeedSequence([policy_noise_seed,chunk_id]) float32 standard normal',
            'extra_noise':'PCG64 SeedSequence([policy_noise_seed,chunk_id,candidate_id]) float32 standard normal',
            'selection':'argmax mean of 10 raw Q heads; lowest candidate id breaks ties',
            'mask':'planned execution prefix min(chunk horizon, remaining environment step budget); no future success labels',
            'sampler':'unchanged model.policy.infer, serial candidates; no OGPO/SDE adapter',
            'action_horizon':int(self.template.action_chunks.shape[1]),'action_dim':int(self.template.action_chunks.shape[2])}
        f=self.output/'PROTOCOL.json'
        if f.exists() and json.loads(f.read_text())!=spec:raise ValueError('output protocol differs; use a fresh directory')
        f.write_text(json.dumps(spec,indent=2)+'\n')

    @torch.no_grad()
    def select(self,policy,policy_observation,observation,prompt,task,noise_seed,chunk_id,episode_id,horizon,remaining_steps):
        from ogpo.value_critic_protocol import StateFeatures
        if noise_seed is None:raise ValueError('Best-of-N requires a fixed episode noise seed')
        if horizon!=self.template.action_chunks.shape[1]:raise ValueError('execution horizon differs from critic replay')
        start=time.monotonic()
        model_dim=int(policy._model.config.action_dim)
        noises=[];actions=[]
        for k in range(self.group_size):
            noise=candidate_noise(noise_seed,chunk_id,k,horizon,model_dim)
            a=np.asarray(policy.infer(policy_observation,noise=noise)['actions'][:horizon],dtype=np.float32).copy()
            if k==0 and self.verify_base and not self.base_verified:
                direct=np.asarray(policy.infer(policy_observation,noise=candidate_noise(noise_seed,chunk_id,0,horizon,model_dim))['actions'][:horizon],dtype=np.float32)
                if not np.array_equal(a,direct):raise RuntimeError('candidate0 differs from direct deterministic base inference')
                self.base_verified=True
                print('BESTOF1_BASE_ACTION_EXACT_PASS',flush=True)
            noises.append(noise);actions.append(a)
        actions=np.stack(actions)
        if actions.shape!=(self.group_size,horizon,self.template.action_chunks.shape[-1]) or not np.isfinite(actions).all():
            raise ValueError(f'invalid candidate shape/values {actions.shape}')
        generation_seconds=time.monotonic()-start
        robot=torch.from_numpy(np.asarray(observation['joint_action']['vector'],dtype=np.float32).copy())[None]
        images={key:torch.from_numpy(np.asarray(observation['observation'][camera]['rgb'],dtype=np.uint8).copy())[None]
                for key,camera in [('image_base','head_camera'),('image_wrist','right_camera')]}
        valid=min(horizon,int(remaining_steps))
        if valid<=0:raise ValueError('no remaining environment steps')
        mask=torch.arange(horizon)[None]<valid
        batch=replace(self.template,observations=robot,proprioceptions=robot.clone(),images=images,
            action_chunks=torch.from_numpy(actions[:1]),execution_masks=mask,
            executed_lengths=torch.tensor([valid]),languages=[prompt],task_ids=[task],
            critic_features=None,next_critic_features=None).to(self.device)
        features=self.critic.encode_state(batch)
        grouped=StateFeatures(readout=features.readout.repeat_interleave(self.group_size,dim=0))
        raw=self.critic.raw_q_ensemble_from_features(grouped,torch.from_numpy(actions).to(self.device),mask.to(self.device).expand(self.group_size,-1)).T.float().cpu().numpy()
        selected=choose_qmean(raw)
        f=self.output/f'episode_{episode_id:06d}_chunk_{chunk_id:04d}.npz'
        if f.exists():raise FileExistsError(f'candidate audit already exists: {f}')
        np.savez_compressed(f,actions=actions,noises=np.stack(noises),raw_q=raw,qmean=raw.mean(-1),
            selected_index=selected,selected_actions=actions[selected],execution_mask=mask.numpy(),
            policy_noise_seed=int(noise_seed),chunk_id=int(chunk_id),episode_id=int(episode_id),
            task=task,prompt=prompt,base_equivalence_verified=self.base_verified,
            generation_seconds=generation_seconds,total_seconds=time.monotonic()-start)
        print(f'BESTOFN chunk={chunk_id} selected={selected} qmean={raw.mean(-1).tolist()} seconds={time.monotonic()-start:.2f}',flush=True)
        return {'actions':actions[selected]}
