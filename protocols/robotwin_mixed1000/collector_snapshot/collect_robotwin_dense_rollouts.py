#!/usr/bin/env python3
"""Collect RoboTwin rollouts at every environment step.

PI0.5 is still queried once per open-loop action chunk.  Unlike the legacy
collector, this collector records every action actually executed inside that
chunk.  After an episode finishes, each environment-step observation is paired
with the next H actually executed actions and the observation reached after
those actions.  Consequently ``actions`` remains [T, H, D] while T is the
number of real environment steps rather than the number of policy calls.
"""

from __future__ import annotations

import argparse
import builtins
import importlib
import importlib.util
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", "external/RoboTwin"))


def _install_compact_logging() -> None:
    """Suppress RoboTwin's per-environment-step chatter in eval logs."""
    original_print = builtins.print

    def compact_print(*args, **kwargs):
        message = " ".join(str(arg) for arg in args).lstrip()
        if message.startswith("step:") or message.startswith("successfully "):
            return
        original_print(*args, **kwargs)

    builtins.print = compact_print


def _prepare_policy_eval_task_state(task_env, task_name: str) -> None:
    """Initialize task state normally created only by an expert demonstration."""
    if task_name != "open_laptop" or hasattr(task_env, "arm_tag"):
        return
    task_module = importlib.import_module(task_env.__class__.__module__)
    face_prod = task_module.get_face_prod(
        task_env.laptop.get_pose().q, [1, 0, 0], [1, 0, 0]
    )
    task_env.arm_tag = task_module.ArmTag("left" if face_prod > 0 else "right")


@dataclass
class EpisodeMeta:
    episode_index: int
    prompt: str
    success: bool
    num_frames: int
    num_environment_steps: int
    num_policy_chunks: int
    action_horizon: int
    action_dim: int
    transition_semantics: str
    task_name: str
    task_config: str
    train_config: str
    model_name: str
    seed: int
    policy_noise_seed: int | None = None
    ogpo_checkpoint: str | None = None


def _rgb(observation: dict, camera: str) -> np.ndarray:
    return np.asarray(
        observation["observation"][camera]["rgb"], dtype=np.uint8
    ).copy()


def _state(observation: dict) -> np.ndarray:
    return np.asarray(observation["joint_action"]["vector"], dtype=np.float32).copy()


def _build_dense_arrays(
    micro_steps: list[dict], *, action_horizon: int, gamma: float, success: bool
) -> dict[str, np.ndarray]:
    if not micro_steps:
        raise ValueError("episode contains no executed environment steps")
    executed_actions = np.stack([step["action"] for step in micro_steps]).astype(
        np.float32, copy=False
    )
    count, action_dim = executed_actions.shape
    actions = np.zeros((count, action_horizon, action_dim), dtype=np.float32)
    execution_mask = np.zeros((count, action_horizon), dtype=np.bool_)
    executed_lengths = np.empty((count,), dtype=np.int64)
    next_image = np.empty_like(np.stack([step["image"] for step in micro_steps]))
    next_wrist_image = np.empty_like(
        np.stack([step["wrist_image"] for step in micro_steps])
    )
    next_state = np.empty_like(np.stack([step["state"] for step in micro_steps]))
    done = np.zeros((count,), dtype=np.bool_)
    chunk_return = np.zeros((count,), dtype=np.float32)
    discount = np.empty((count,), dtype=np.float32)

    for start in range(count):
        length = min(action_horizon, count - start)
        stop = start + length
        actions[start, :length] = executed_actions[start:stop]
        execution_mask[start, :length] = True
        executed_lengths[start] = length
        endpoint = micro_steps[stop - 1]
        next_image[start] = endpoint["next_image"]
        next_wrist_image[start] = endpoint["next_wrist_image"]
        next_state[start] = endpoint["next_state"]
        terminal = stop == count
        done[start] = terminal
        discount[start] = float(gamma**length)
        if success and terminal:
            chunk_return[start] = float(gamma ** (length - 1))

    arrays = {
        "image": np.stack([step["image"] for step in micro_steps]),
        "wrist_image": np.stack([step["wrist_image"] for step in micro_steps]),
        "state": np.stack([step["state"] for step in micro_steps]),
        "actions": actions,
        "execution_mask": execution_mask,
        "executed_lengths": executed_lengths,
        "executed_action": executed_actions,
        "next_image": next_image,
        "next_wrist_image": next_wrist_image,
        "next_state": next_state,
        "timestamp": np.asarray([step["timestamp"] for step in micro_steps], dtype=np.float32),
        "policy_chunk_id": np.asarray(
            [step["policy_chunk_id"] for step in micro_steps], dtype=np.int64
        ),
        "policy_chunk_offset": np.asarray(
            [step["policy_chunk_offset"] for step in micro_steps], dtype=np.int64
        ),
        "done": done,
        "chunk_return": chunk_return,
        "discount": discount,
    }
    _audit_dense_arrays(arrays, gamma=gamma, success=success)
    return arrays


def _audit_dense_arrays(
    arrays: dict[str, np.ndarray], *, gamma: float, success: bool
) -> None:
    count, horizon, action_dim = arrays["actions"].shape
    if count < 2 or horizon <= 1 or action_dim <= 0:
        raise ValueError(f"invalid dense rollout shape: {arrays['actions'].shape}")
    if float(np.abs(arrays["executed_action"]).max()) == 0.0:
        raise ValueError("all executed actions are zero")
    if not np.allclose(arrays["actions"][:, 0], arrays["executed_action"]):
        raise ValueError("first action in a window is not the action executed at that state")
    lengths = arrays["executed_lengths"]
    if not np.array_equal(arrays["execution_mask"].sum(axis=1), lengths):
        raise ValueError("execution mask and executed lengths disagree")
    expected_lengths = np.minimum(horizon, count - np.arange(count))
    if not np.array_equal(lengths, expected_lengths):
        raise ValueError("dense sliding-window lengths are incorrect")
    if not np.allclose(arrays["discount"], np.power(gamma, lengths), atol=2e-7):
        raise ValueError("discounts do not match gamma ** executed_length")
    terminal = np.arange(count) + lengths == count
    if not np.array_equal(arrays["done"], terminal):
        raise ValueError("terminal flags do not match action-window endpoints")
    if success:
        expected_return = np.where(terminal, np.power(gamma, lengths - 1), 0.0)
    else:
        expected_return = np.zeros_like(arrays["chunk_return"])
    if not np.allclose(arrays["chunk_return"], expected_return, atol=2e-7):
        raise ValueError("sparse terminal returns are incorrect")
    if count > horizon:
        endpoint = horizon
        if not np.array_equal(arrays["next_image"][0], arrays["image"][endpoint]):
            raise ValueError("chunk next_image is not the observation at t + horizon")
        if not np.allclose(arrays["next_state"][0], arrays["state"][endpoint]):
            raise ValueError("chunk next_state is not the state at t + horizon")


def main() -> None:
    if os.environ.get("ROBOTWIN_COMPACT_LOG", "1") == "1":
        _install_compact_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="pick_dual_bottles")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--train-config-name", default="pi05_aloha_full_base")
    parser.add_argument("--model-name", default="model_jax_clean")
    parser.add_argument("--checkpoint-id", default="30000")
    parser.add_argument(
        "--ogpo-checkpoint",
        type=Path,
        help="Load this OGPO actor checkpoint on top of the configured PyTorch PI0.5 base.",
    )
    parser.add_argument(
        "--pi05-checkpoint-dir",
        type=Path,
        default=Path(os.environ.get("PI05_CHECKPOINT_DIR", "checkpoints/pi05/model_clean50")),
        help="Base checkpoint directory used by an OGPO full-finetune checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--pi0-step", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--episode-manifest",
        type=Path,
        help=(
            "JSON manifest containing records with shard, seed, prompt, and "
            "policy_noise_seed. When set, no seed or prompt is sampled."
        ),
    )
    parser.add_argument(
        "--trust-episode-manifest",
        action="store_true",
        help=(
            "Skip RoboTwin's stochastic expert replay for seeds already certified "
            "by an episode manifest. Requires --episode-manifest."
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Evaluate success without storing dense image/action replay arrays.",
    )
    parser.add_argument(
        "--skip-expert-assertions",
        action="store_true",
        help=(
            "Treat AssertionError or TypeError from RoboTwin expert scene "
            "validation as an invalid seed and continue searching. "
            "Manifest-certified seeds still fail closed."
        ),
    )
    args = parser.parse_args()
    if args.trust_episode_manifest and args.episode_manifest is None:
        parser.error("--trust-episode-manifest requires --episode-manifest")

    root = ROBOTWIN_ROOT.resolve()
    os.chdir(root)
    import sys

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "script"))
    sys.path.insert(0, str(root / "policy" / "pi05" / "src"))
    sys.path.insert(0, str(root / "policy" / "pi05" / "packages" / "openpi-client" / "src"))

    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
    from policy.pi05.deploy_policy import encode_obs
    from script.eval_policy import class_decorator, eval_function_decorator, get_embodiment_config

    gei_path = root / "description" / "utils" / "generate_episode_instructions.py"
    spec = importlib.util.spec_from_file_location(
        "robotwin_generate_episode_instructions", gei_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {gei_path}")
    gei = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gei)
    generate_episode_descriptions = gei.generate_episode_descriptions

    raw_dir = args.output_dir / "raw_rollouts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing_meta_paths = sorted(raw_dir.glob("episode_*/meta.json"))
    existing_metas = [json.loads(path.read_text()) for path in existing_meta_paths]
    manifest_entries = None
    if args.episode_manifest is not None:
        manifest_payload = json.loads(args.episode_manifest.read_text(encoding="utf-8"))
        records = (
            manifest_payload["episodes"]
            if isinstance(manifest_payload, dict)
            else manifest_payload
        )
        manifest_entries = [
            record for record in records if int(record["shard"]) == int(args.seed)
        ]
        manifest_entries.sort(key=lambda record: int(record["episode_index"]))
        if len(manifest_entries) < int(args.num_episodes):
            raise ValueError(
                f"manifest shard {args.seed} has {len(manifest_entries)} episodes, "
                f"fewer than the requested {args.num_episodes}"
            )
        manifest_entries = manifest_entries[: int(args.num_episodes)]
        for index, meta in enumerate(existing_metas):
            expected = manifest_entries[index]
            for key in ("seed", "prompt", "policy_noise_seed"):
                if meta.get(key) != expected[key]:
                    raise ValueError(
                        f"existing episode {index} has {key}={meta.get(key)!r}, "
                        f"manifest requires {expected[key]!r}"
                    )
    if len(existing_metas) >= args.num_episodes:
        print(f"{raw_dir} already has {len(existing_metas)} episodes; nothing to collect")
        return

    with (root / "task_config" / f"{args.task_config}.yml").open(
        "r", encoding="utf-8"
    ) as handle:
        task_args = yaml.load(handle.read(), Loader=yaml.FullLoader)
    task_args.update(
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "ckpt_setting": args.model_name,
            "policy_name": "pi05",
            "eval_mode": True,
            "render_freq": 0,
            "eval_video_save_dir": None,
        }
    )

    embodiment_types = yaml.load(
        (Path(CONFIGS_PATH) / "_embodiment_config.yml").read_text(),
        Loader=yaml.FullLoader,
    )

    def embodiment_file(embodiment: str) -> str:
        value = embodiment_types[embodiment]["file_path"]
        if value is None:
            raise RuntimeError(f"no embodiment file for {embodiment}")
        return value

    embodiment_type = task_args.get("embodiment")
    if len(embodiment_type) == 1:
        task_args["left_robot_file"] = embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = embodiment_file(embodiment_type[0])
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = embodiment_file(embodiment_type[1])
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment must contain one or three items")
    task_args["left_embodiment_config"] = get_embodiment_config(
        task_args["left_robot_file"]
    )
    task_args["right_embodiment_config"] = get_embodiment_config(
        task_args["right_robot_file"]
    )

    task_env = class_decorator(args.task_name)
    get_model = eval_function_decorator("pi05", "get_model")
    reset_model = eval_function_decorator("pi05", "reset_model")
    model = get_model(
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "ckpt_setting": args.model_name,
            "policy_name": "pi05",
            "instruction_type": "unseen",
            "train_config_name": args.train_config_name,
            "model_name": args.model_name,
            "checkpoint_id": args.checkpoint_id,
            "pi0_step": args.pi0_step,
            "seed": args.seed,
        }
    )
    if args.ogpo_checkpoint is not None:
        import gc
        import torch

        # get_model supplies RoboTwin's observation-window wrapper. Replace only
        # its OpenPI policy so task setup, camera encoding, and action execution
        # remain identical to the base evaluation path.
        del model.policy
        gc.collect()
        torch.cuda.empty_cache()
        ogpo_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(ogpo_root / "src"))
        from ogpo.inference_policy import create_pi05_ogpo_inference_policy

        model.policy = create_pi05_ogpo_inference_policy(
            pi05_checkpoint_dir=args.pi05_checkpoint_dir,
            train_config_name=args.train_config_name,
            ogpo_checkpoint=args.ogpo_checkpoint,
            device="cuda",
            include_reference_policy=False,
            include_critic=False,
            inference_num_steps=10,
            require_model_flow_dim=False,
        )
        print(f"loaded OGPO actor checkpoint: {args.ogpo_checkpoint}", flush=True)
    if int(model.pi0_step) != int(args.pi0_step):
        raise ValueError(
            f"model clamped pi0_step from {args.pi0_step} to {model.pi0_step}"
        )

    now_seed = (
        int(manifest_entries[len(existing_metas)]["seed"])
        if manifest_entries is not None and len(existing_metas) < len(manifest_entries)
        else (
            max(int(meta.get("seed", 100000 * (1 + args.seed))) for meta in existing_metas)
            + 1
            if existing_metas
            else 100000 * (1 + args.seed)
        )
    )
    collected = len(existing_metas)
    while collected < args.num_episodes:
        manifest_entry = (
            None if manifest_entries is None else manifest_entries[collected]
        )
        if manifest_entry is not None:
            now_seed = int(manifest_entry["seed"])
        episode_info = None
        if not (manifest_entry is not None and args.trust_episode_manifest):
            try:
                task_env.setup_demo(
                    now_ep_num=collected, seed=now_seed, is_test=True, **task_args
                )
                episode_info = task_env.play_once()
                setup_success = bool(task_env.plan_success and task_env.check_success())
                task_env.close_env()
            except UnStableError:
                task_env.close_env()
                if manifest_entry is not None:
                    raise RuntimeError(
                        f"manifest seed {now_seed} became unstable during expert validation"
                    )
                now_seed += 1
                continue
            except (AssertionError, TypeError) as error:
                task_env.close_env()
                if manifest_entry is not None or not args.skip_expert_assertions:
                    raise
                print(
                    f"skipping expert-invalid seed={now_seed} "
                    f"assertion={error}",
                    flush=True,
                )
                now_seed += 1
                continue
            except Exception:
                task_env.close_env()
                raise
            if not setup_success:
                if manifest_entry is not None:
                    raise RuntimeError(
                        f"manifest seed {now_seed} failed expert validation"
                    )
                now_seed += 1
                continue

        task_env.setup_demo(
            now_ep_num=collected, seed=now_seed, is_test=True, **task_args
        )
        _prepare_policy_eval_task_state(task_env, task_args["task_name"])
        if manifest_entry is not None:
            prompt = str(manifest_entry["prompt"])
        else:
            if episode_info is None:
                raise RuntimeError("expert validation did not produce episode metadata")
            descriptions = generate_episode_descriptions(
                task_args["task_name"], [episode_info["info"]], 1
            )
            prompt = str(np.random.choice(descriptions[0]["unseen"]))
        policy_noise_seed = (
            int(manifest_entry["policy_noise_seed"])
            if manifest_entry is not None
            else None
        )
        task_env.set_instruction(prompt)
        reset_model(model)

        micro_steps: list[dict] = []
        environment_steps = 0
        action_dim = 0
        policy_chunk_id = 0
        try:
            while task_env.take_action_cnt < task_env.step_lim and not task_env.eval_success:
                observation = task_env.get_obs()
                if model.observation_window is None:
                    model.set_language(task_env.get_instruction())
                input_rgb, input_state = encode_obs(observation)
                model.update_observation_window(input_rgb, input_state)
                if policy_noise_seed is None:
                    policy_output = model.policy.infer(model.observation_window)
                else:
                    flow_policy = getattr(model.policy, "flow_policy", None)
                    model_action_dim = getattr(flow_policy, "model_action_dim", None)
                    if model_action_dim is None:
                        model_action_dim = int(model.policy._model.config.action_dim)
                    noise_rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [policy_noise_seed, policy_chunk_id]
                        )
                    )
                    policy_noise = noise_rng.standard_normal(
                        (int(model.pi0_step), int(model_action_dim)),
                        dtype=np.float32,
                    )
                    policy_output = model.policy.infer(
                        model.observation_window,
                        noise=policy_noise,
                    )
                sampled_actions = np.asarray(
                    policy_output["actions"][: model.pi0_step], dtype=np.float32
                )
                state_dim = _state(observation).shape[0]
                if sampled_actions.shape != (int(args.pi0_step), state_dim):
                    raise ValueError(
                        f"expected action chunk {(args.pi0_step, state_dim)}, "
                        f"got {sampled_actions.shape}"
                    )

                for offset, action in enumerate(sampled_actions):
                    if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success:
                        break
                    before = observation
                    before_count = int(task_env.take_action_cnt)
                    task_env.take_action(action)
                    if int(task_env.take_action_cnt) != before_count + 1:
                        raise RuntimeError("RoboTwin did not execute exactly one environment step")
                    observation = task_env.get_obs()
                    next_rgb, next_state_for_model = encode_obs(observation)
                    model.update_observation_window(next_rgb, next_state_for_model)
                    environment_steps += 1
                    action_dim = int(np.asarray(action).size)
                    if not args.metadata_only:
                        micro_steps.append(
                            {
                                "image": _rgb(before, "head_camera"),
                                "wrist_image": _rgb(before, "right_camera"),
                                "state": _state(before),
                                "action": np.asarray(action, dtype=np.float32).copy(),
                                "next_image": _rgb(observation, "head_camera"),
                                "next_wrist_image": _rgb(observation, "right_camera"),
                                "next_state": _state(observation),
                                "timestamp": float(before_count / 10.0),
                                "policy_chunk_id": policy_chunk_id,
                                "policy_chunk_offset": offset,
                            }
                        )
                policy_chunk_id += 1

            success = bool(task_env.eval_success)
            arrays = None
            if not args.metadata_only:
                arrays = _build_dense_arrays(
                    micro_steps,
                    action_horizon=int(args.pi0_step),
                    gamma=float(args.gamma),
                    success=success,
                )
            episode_name = f"episode_{collected:06d}"
            partial_dir = raw_dir / f".{episode_name}.partial"
            if partial_dir.exists():
                raise FileExistsError(
                    f"stale partial directory must be inspected: {partial_dir}"
                )
            partial_dir.mkdir()
            if arrays is not None:
                np.savez_compressed(partial_dir / "frames.npz", **arrays)
            meta = EpisodeMeta(
                episode_index=collected,
                prompt=prompt,
                success=success,
                num_frames=0 if args.metadata_only else environment_steps,
                num_environment_steps=environment_steps,
                num_policy_chunks=policy_chunk_id,
                action_horizon=int(args.pi0_step),
                action_dim=action_dim,
                transition_semantics=(
                    "evaluation metadata only"
                    if args.metadata_only
                    else "dense overlapping actual-executed action windows"
                ),
                task_name=args.task_name,
                task_config=args.task_config,
                train_config=args.train_config_name,
                model_name=args.model_name,
                seed=now_seed,
                policy_noise_seed=policy_noise_seed,
                ogpo_checkpoint=(
                    str(args.ogpo_checkpoint.resolve())
                    if args.ogpo_checkpoint is not None
                    else None
                ),
            )
            (partial_dir / "meta.json").write_text(
                json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8"
            )
            if arrays is not None:
                cv2.imwrite(
                    str(partial_dir / "thumb.png"),
                    cv2.cvtColor(arrays["image"][0], cv2.COLOR_RGB2BGR),
                )
            partial_dir.rename(raw_dir / episode_name)
            print(
                f"episode={collected + 1} success={success} "
                f"seed={now_seed} policy_noise_seed={policy_noise_seed} "
                f"environment_steps={environment_steps} "
                f"policy_chunks={policy_chunk_id}",
                flush=True,
            )
        finally:
            task_env.close_env(clear_cache=((collected + 1) % 10 == 0))

        collected += 1
        now_seed += 1

    print(f"shard_complete episodes={collected}", flush=True)


if __name__ == "__main__":
    main()
