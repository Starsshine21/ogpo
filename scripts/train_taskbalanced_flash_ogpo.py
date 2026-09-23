"""Reuse full-entry multi-shard task cycling with canonical Flash transactions."""
import json
import random
import time
import numpy as np
import torch
import train_full_ogpo as entry
from ogpo.trainer import flash_actor_update, finalize_actor_update_transaction


def flash_update(state, batch, config, **kwargs):
    assert config['actor']['flash_enabled']
    assert config['flow']['backend_train_mode']=='action_expert'
    started=time.perf_counter()
    metrics=flash_actor_update(state,batch,config,actor_step=state.actor_step,**kwargs)
    metrics.update(finalize_actor_update_transaction(state,
        accepted=bool(metrics.get('actor_update_accepted',False)),config=config))
    metrics['actor_update_mode']='flash_selected_transition'
    metrics['actor_update_seconds']=time.perf_counter()-started
    print('FLASH_TIMING '+json.dumps({k:metrics.get(k) for k in
        ('actor_step','actor_update_seconds','actor_update_accepted','actor_grad_norm',
         'post_update_reference_kl','selected_step_min','selected_step_max','chi2_enabled')}),flush=True)
    return metrics


if __name__=='__main__':
    random.seed(20260918);np.random.seed(20260918);torch.manual_seed(20260918)
    entry.full_actor_update=flash_update
    print('ACTOR_ROUTE Flash selected-transition; full sampling; task-balanced; raw10 CA+ChiPO',flush=True)
    entry.main()
