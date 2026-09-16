"""Single-sample, clean-action Q guidance. No candidate selection or CA."""
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch

from robotwin_bestofn import candidate_noise, file_hash

ROOT = Path(__file__).resolve().parents[1]


def unnormalize(x, stats, quantiles, quantile_method=None):
    def tensor(a):
        return torch.as_tensor(np.asarray(a).copy(), device=x.device, dtype=x.dtype)
    if quantiles:
        if quantile_method is not None:
            # The deployment and training OpenPI copies differ in handling
            # degenerate quantiles. Extract the *loaded* affine map from two
            # constant probes; never convert/detach the actual action tensor.
            zero = np.zeros(x.shape[-1], dtype=np.float64)
            offset = quantile_method(zero, stats)
            scale = quantile_method(zero+1, stats)-offset
            np.testing.assert_allclose(quantile_method(zero+2, stats), offset+2*scale)
            return x*tensor(scale)+tensor(offset)
        low, high = tensor(stats.q01), tensor(stats.q99)
        dim = low.shape[-1]
        main = x[..., :dim]
        restored = torch.where(high-low < 1e-4, main+(high+low)/2,
                               (main+1)/2*(high-low+1e-6)+low)
        return torch.cat((restored, x[..., dim:]), -1)
    mean, std = tensor(stats.mean), tensor(stats.std)
    padding = x.shape[-1]-mean.shape[-1]
    if padding < 0:
        raise ValueError('normalizer wider than model output')
    mean = torch.nn.functional.pad(mean, (0, padding), value=0)
    std = torch.nn.functional.pad(std, (0, padding), value=1)
    return x*(std+1e-6)+mean


def decode_actions(output_transform, normalized_state, normalized_actions):
    """Differentiable replica of this base's output transforms; fail closed."""
    state, actions = normalized_state, normalized_actions
    for transform in output_transform.transforms:
        name = type(transform).__name__
        if name == 'Unnormalize':
            for key, stats in transform.norm_stats.items():
                if key == 'state':
                    state = unnormalize(state, stats, transform.use_quantiles, transform._unnormalize_quantile)
                elif key == 'actions':
                    actions = unnormalize(actions, stats, transform.use_quantiles, transform._unnormalize_quantile)
                else:
                    raise ValueError(f'unsupported normalization key: {key}')
        elif name == 'AbsoluteActions':
            if transform.mask is not None:
                mask = torch.as_tensor(transform.mask, device=actions.device, dtype=torch.bool)
                offset = torch.where(mask, state[..., :len(mask)], 0).unsqueeze(-2)
                actions = torch.cat((actions[..., :len(mask)]+offset, actions[..., len(mask):]), -1)
        elif name == 'AlohaOutputs':
            if transform.adapt_to_pi:
                raise ValueError('QGF currently requires adapt_to_pi=False')
            actions = actions[..., :14]
        else:
            raise ValueError(f'unsupported QGF output transform: {name}')
    # Deployment state/normalization can be float64; the replay and action
    # projection use float32. Tensor.to preserves the action derivative.
    return actions.float()


def guided_euler(x, velocity, dt, gradient, strength):
    # Pi0 denoises t=1 -> 0: dt is negative; ascent must have positive sign.
    base = x + dt*velocity
    return base if strength == 0 else base + (-dt)*strength*gradient


@torch.no_grad()
def sample_qgf(model, device, observation, noise, q_function, decoder, strength,
               audit, verify=False, num_steps=10):
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    images, img_masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
    prefix, pad, att = model.embed_prefix(images, img_masks, tokens, token_masks)
    attention = model._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = 'eager'
    _, cache = model.paligemma_with_expert.forward(attention_mask=attention,
        position_ids=torch.cumsum(pad, dim=1)-1, past_key_values=None,
        inputs_embeds=[prefix, None], use_cache=True)
    dt = torch.tensor(-1.0/num_steps, dtype=torch.float32, device=device)
    t = torch.tensor(1.0, dtype=torch.float32, device=device)
    x = noise
    while t >= -dt/2:
        velocity = model.denoise_step(state, pad, cache, x, t.expand(x.shape[0]))
        if strength == 0:
            x = guided_euler(x, velocity, dt, None, 0)
        else:
            with torch.enable_grad():
                clean = (x-t*velocity).detach().requires_grad_(True)
                actions = decoder(observation.state, clean)
                raw = q_function(actions)
                if raw.shape != (10, x.shape[0]):
                    raise ValueError(f'expected 10 raw Q heads, got {raw.shape}')
                q = raw.mean()
                gradient, = torch.autograd.grad(q, clean)
            if not torch.isfinite(gradient).all() or not torch.isfinite(raw).all():
                raise RuntimeError('nonfinite QGF gradient/Q')
            if verify and not audit.get('gradient_check'):
                direction = gradient / gradient.norm().clamp_min(1e-12)
                eps = 0.01
                plus = q_function(decoder(observation.state, clean.detach()+eps*direction)).mean()
                minus = q_function(decoder(observation.state, clean.detach()-eps*direction)).mean()
                finite = float((plus-minus)/(2*eps))
                analytic = float((gradient*direction).sum())
                relative = abs(finite-analytic)/max(abs(finite), abs(analytic), 1e-7)
                if analytic <= 1e-8 or relative > 0.15:
                    raise RuntimeError(f'QGF finite difference failed: {finite=} {analytic=} {relative=}')
                if torch.count_nonzero(gradient[..., 14:]):
                    raise RuntimeError('padded model dimensions received Q guidance')
                audit['gradient_check'] = dict(finite_difference=finite, autograd=analytic, relative_error=relative)
                print('QGF_ACTION_GRADIENT_PASS', audit['gradient_check'], flush=True)
            audit.setdefault('steps', []).append(dict(t=float(t), qmean=float(q),
                gradient_norm=float(gradient.norm()), correction_norm=float((-dt*strength*gradient).norm())))
            x = guided_euler(x, velocity, dt, gradient, strength)
        t += dt
    if not torch.isfinite(x).all():
        raise RuntimeError('nonfinite guided actions')
    return x


class QGF:
    def __init__(self, checkpoint, config, output, base_checkpoint, strength=1.0, verify=False, device='cuda'):
        from train_udivl_critic import load_config
        from ogpo.replay import load_replay
        from ogpo.trainer import build_train_state, load_critic_checkpoint
        if not np.isfinite(strength) or strength < 0:
            raise ValueError('QGF strength must be finite and nonnegative')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        cfg = load_config(Path(config))
        init = load_replay(ROOT/cfg['data']['distributed_model_init_path'])
        state = build_train_state(cfg, init, device=device)
        payload = load_critic_checkpoint(checkpoint, state, load_optimizer=False)
        self.critic = state.critic.eval().requires_grad_(False)
        self.template = init.index_select(torch.tensor([0]))
        self.device, self.strength, self.verify = device, float(strength), verify
        self.verified = False
        protocol = dict(protocol='single-sample-clean-action-QGF-v1', group_size=1,
            critic_checkpoint=str(Path(checkpoint).resolve()), critic_sha256=file_hash(checkpoint),
            critic_training_step=int(payload['training_step']), config_sha256=file_hash(config),
            base_checkpoint=str(Path(base_checkpoint).resolve()), base_sha256=file_hash(base_checkpoint),
            strength=self.strength, q='raw10 mean', CA=False, bestofn=False,
            clean_estimate='x_t - t*v; stop velocity Jacobian',
            update='x_next=x_t+dt*v-dt*strength*grad_clean_Q; dt=-1/num_steps',
            action_decode='differentiable base output transforms; critic receives environment-space chunk',
            noise='PCG64 SeedSequence([policy_noise_seed,chunk_id]) float32',
            mask='planned prefix min(horizon,remaining environment steps)',
            frozen_base=True, frozen_critic=True)
        path = self.output/'PROTOCOL.json'
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError('QGF protocol mismatch; use a fresh output')
        path.write_text(json.dumps(protocol, indent=2)+'\n')
        del payload, state

    @torch.no_grad()
    def infer(self, policy, policy_observation, observation, prompt, task, noise_seed,
              chunk_id, episode_id, horizon, remaining_steps):
        if noise_seed is None or horizon != self.template.action_chunks.shape[1]:
            raise ValueError('QGF requires fixed noise and matching critic action horizon')
        start = time.monotonic()
        policy._model.eval().requires_grad_(False)
        noise = candidate_noise(noise_seed, chunk_id, 0, horizon, policy._model.config.action_dim)
        robot = torch.from_numpy(np.asarray(observation['joint_action']['vector'], dtype=np.float32).copy())[None]
        images = {key: torch.from_numpy(np.asarray(observation['observation'][camera]['rgb'], dtype=np.uint8).copy())[None]
                  for key, camera in [('image_base', 'head_camera'), ('image_wrist', 'right_camera')]}
        valid = min(horizon, int(remaining_steps))
        if valid <= 0:
            raise ValueError('no remaining environment steps')
        mask = (torch.arange(horizon)[None] < valid).to(self.device)
        batch = replace(self.template, observations=robot, proprioceptions=robot.clone(), images=images,
            execution_masks=mask.cpu(), executed_lengths=torch.tensor([valid]), languages=[prompt], task_ids=[task],
            critic_features=None, next_critic_features=None).to(self.device)
        features = self.critic.encode_state(batch)
        def q_function(actions):
            return self.critic.raw_q_ensemble_from_features(features, actions, mask).float()
        def decoder(state, actions):
            return decode_actions(policy._output_transform, state, actions)
        original = policy._sample_actions
        audit = dict(task=task, episode_id=episode_id, chunk_id=chunk_id, policy_noise_seed=int(noise_seed))
        check = self.verify and not self.verified
        def run(strength):
            def sample(device, obs, noise=None, num_steps=10):
                if check:
                    expected = policy._output_transform(dict(state=obs.state[0].cpu().numpy().copy(),
                                                            actions=noise[0].cpu().numpy().copy()))['actions']
                    decoded = decoder(obs.state, noise)[0].cpu().numpy()
                    np.testing.assert_allclose(decoded, expected, rtol=2e-5, atol=2e-5)
                    audit['decoder_max_error'] = float(np.max(np.abs(decoded-expected)))
                    audit['output_transforms'] = [type(t).__name__ for t in policy._output_transform.transforms]
                return sample_qgf(policy._model, device, obs, noise, q_function, decoder,
                                  strength, audit, verify=check, num_steps=num_steps)
            try:
                policy._sample_actions = sample
                return policy.infer(policy_observation, noise=noise)
            finally:
                policy._sample_actions = original
        if check:
            base = policy.infer(policy_observation, noise=noise)['actions'].copy()
            zero = run(0)['actions']
            np.testing.assert_array_equal(base, zero)
            audit['zero_guidance_exact'] = True
            print('QGF_ZERO_GUIDANCE_EXACT_PASS', flush=True)
        result = run(self.strength)
        actions = np.asarray(result['actions'], dtype=np.float32)
        if actions.shape != (horizon, 14) or not np.isfinite(actions).all():
            raise RuntimeError(f'invalid QGF action shape/values: {actions.shape}')
        if any(p.grad is not None or p.requires_grad for p in self.critic.parameters()):
            raise RuntimeError('critic was not frozen')
        if any(p.grad is not None or p.requires_grad for p in policy._model.parameters()):
            raise RuntimeError('base was not frozen')
        audit['frozen_parameters_pass'] = True
        audit['total_seconds'] = time.monotonic()-start
        if check:
            audit['guided_vs_base_l2'] = float(np.linalg.norm(actions-base))
        raw = q_function(torch.from_numpy(actions[None]).to(self.device)).cpu().numpy()
        stem = self.output/f'episode_{episode_id:06d}_chunk_{chunk_id:04d}'
        if stem.with_suffix('.npz').exists():
            raise FileExistsError(stem)
        np.savez_compressed(stem.with_suffix('.npz'), noise=noise, actions=actions, raw_q=raw,
                            execution_mask=mask.cpu().numpy())
        stem.with_suffix('.json').write_text(json.dumps(audit, indent=2)+'\n')
        self.verified = self.verified or check
        print(f'QGF chunk={chunk_id} qmean={raw.mean():.6g} seconds={audit["total_seconds"]:.2f}', flush=True)
        return result
