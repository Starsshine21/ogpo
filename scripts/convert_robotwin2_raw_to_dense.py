#!/usr/bin/env python3
"""Convert RoboTwin2.0 ALOHA-AgileX raw clean demos to OGPO dense episodes.

The output deliberately matches ``collect_robotwin_dense_rollouts.py``: every
micro-step owns an overlapping action window, and the bootstrap observation is
the observation after the whole executed prefix (not merely the next frame).
"""
import argparse, json, pickle, zipfile
from io import BytesIO
from pathlib import Path
import h5py, numpy as np
from PIL import Image

TASKS = ["adjust_bottle","click_alarmclock","click_bell","dump_bin_bigbin","handover_mic","open_laptop","pick_dual_bottles","place_a2b_right","place_object_stand","press_stapler"]

def decode(x):
    if isinstance(x, np.ndarray) and x.dtype.kind in "SU": x = x.tobytes()
    if isinstance(x, np.bytes_): x = bytes(x)
    return np.asarray(Image.open(BytesIO(x)).convert("RGB"), dtype=np.uint8)

def convert_episode(h5, pkl, instruction, out, task, episode_id, shard, gamma):
    with h5py.File(h5, "r") as f:
        joints = np.asarray(f["joint_action/vector"], dtype=np.float32)
        head = [decode(x) for x in f["observation/head_camera/rgb"]]
        left_wrist = [decode(x) for x in f["observation/left_camera/rgb"]]
        right_wrist = [decode(x) for x in f["observation/right_camera/rgb"]]
    T = len(joints)
    if T < 2 or joints.shape[1] != 14: raise ValueError(f"{h5}: joints {joints.shape}")
    with open(pkl, "rb") as f: traj = pickle.load(f)
    statuses = [x.get("status", "") for x in traj.get("left_joint_path", []) + traj.get("right_joint_path", [])]
    success = bool(statuses) and all(str(x).lower() == "success" for x in statuses)
    rows = T - 1
    horizon = 50
    actions = np.empty((rows, horizon, 14), np.float32)
    masks = np.zeros((rows, horizon), bool)
    for t in range(rows):
        n = min(horizon, rows - t)
        actions[t, :n] = joints[t+1:t+1+n]
        actions[t, n:] = joints[-1]
        masks[t, :n] = True
    images = np.stack(head[:-1])
    left_wrists = np.stack(left_wrist[:-1])
    right_wrists = np.stack(right_wrist[:-1])
    next_images = np.empty_like(images)
    next_left_wrists = np.empty_like(left_wrists)
    next_right_wrists = np.empty_like(right_wrists)
    next_states = np.empty_like(joints[:-1])
    lengths = masks.sum(1).astype(np.int64)
    for t, length in enumerate(lengths):
        endpoint = t + int(length)
        next_images[t] = head[endpoint]
        next_left_wrists[t] = left_wrist[endpoint]
        next_right_wrists[t] = right_wrist[endpoint]
        next_states[t] = joints[endpoint]
    task_out = out / task / f"shard_{shard:02d}" / "raw_rollouts" / f"episode_{episode_id:06d}"
    task_out.mkdir(parents=True, exist_ok=True)
    terminal = np.arange(rows) + lengths == rows
    returns = np.zeros(rows, np.float32)
    if success:
        returns[terminal] = np.power(float(gamma), lengths[terminal] - 1)
    np.savez_compressed(task_out / "frames.npz", image=images, left_wrist_image=left_wrists, wrist_image=right_wrists, state=joints[:-1], actions=actions, executed_action=joints[1:], next_image=next_images, next_left_wrist_image=next_left_wrists, next_wrist_image=next_right_wrists, next_state=next_states, timestamp=np.arange(rows, dtype=np.float32) / 10.0, policy_chunk_id=np.arange(rows, dtype=np.int64) // horizon, policy_chunk_offset=np.arange(rows, dtype=np.int64) % horizon, execution_mask=masks, executed_lengths=lengths, chunk_return=returns, discount=np.power(float(gamma), lengths), done=terminal.astype(np.float32))
    meta={"task_name":task,"task_config":"demo_clean","train_config":"pi05_robotwin2_clean50_full","model_name":"model_clean50","prompt":instruction,"success":success,"num_frames":rows,"num_environment_steps":rows,"num_policy_chunks":int(np.ceil(rows / horizon)),"action_horizon":horizon,"action_dim":14,"transition_semantics":"dense overlapping actual-executed action windows","seed":episode_id,"policy_noise_seed":episode_id,"source":"RoboTwin2.0_aloha-agilex_clean_50","action_semantics":"absolute_joint_14d_next_state_chunk"}
    (task_out / "meta.json").write_text(json.dumps(meta,indent=2)+"\n")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--task",required=True); ap.add_argument("--limit",type=int,default=50); ap.add_argument("--gamma",type=float,default=.995); args=ap.parse_args()
    if args.input.is_dir():
        base = args.input
    else:
        with zipfile.ZipFile(args.input) as z:
            root=next(n.split('/')[0] for n in z.namelist() if n.endswith('.pkl'))
            stage=args.output/"_raw"/args.task; stage.mkdir(parents=True,exist_ok=True); z.extractall(stage)
        base=stage/root
    inst_dir=base/"instructions"; data_dir=base/"data"; traj_dir=base/"_traj_data"
    for i in range(args.limit):
        ins=json.loads((inst_dir/f"episode{i}.json").read_text()); prompt=(ins.get("seen") or [args.task])[0]
        convert_episode(data_dir/f"episode{i}.hdf5",traj_dir/f"episode{i}.pkl",prompt,args.output,args.task,i,i%4,args.gamma)
    print(json.dumps({"task":args.task,"episodes":args.limit,"output":str(args.output)}))
if __name__ == "__main__": main()
