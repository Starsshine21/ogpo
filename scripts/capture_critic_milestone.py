"""CPU-only sidecar: preserve an exact latest.pt step without touching training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import torch


def signature(path):
    s = path.stat()
    return s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_step(path):
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    return int(payload['training_step'])


def capture_once(source, destination, target):
    if destination.exists():
        if checkpoint_step(destination) != target:
            raise RuntimeError(f'existing destination has wrong step: {destination}')
        return dict(state='saved', training_step=target, path=str(destination), sha256=sha256(destination))
    before = signature(source)
    step = checkpoint_step(source)
    if signature(source) != before:
        return dict(state='retry')
    if step < target:
        return dict(state='waiting', training_step=step)
    if step > target:
        raise RuntimeError(f'missed target={target}; source is already step={step}: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.capture_{target}_', suffix='.partial', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        if signature(source) != before:
            return dict(state='retry')
        if checkpoint_step(temporary) != target:
            return dict(state='retry')
        digest = sha256(temporary)
        source_digest = sha256(source)
        if signature(source) != before or digest != source_digest:
            return dict(state='retry')
        # Atomic publication with no overwrite, unlike os.replace.
        os.link(temporary, destination)
        return dict(state='saved', training_step=target, path=str(destination),
                    source=str(source), source_signature=before, sha256=digest,
                    source_sha256=source_digest, saved_at=time.time())
    finally:
        temporary.unlink(missing_ok=True)  # Only this invocation's private temporary copy.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, action='append', required=True)
    parser.add_argument('--target', type=int, default=16000)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--timeout-hours', type=float, default=12)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.timeout_hours <= 0:
        parser.error('poll/timeout must be positive')
    torch.set_num_threads(1)
    runs = [p.resolve() for p in args.run_dir]
    previous, checked, results = {}, {}, {}
    started = time.monotonic()
    args.status.parent.mkdir(parents=True, exist_ok=True)
    while True:
        for run in runs:
            key = str(run)
            if results.get(key, {}).get('state') in ('saved', 'error'):
                continue
            source = run/'latest.pt'
            dest = run/'milestones'/f'critic_step_{args.target:05d}.pt'
            try:
                sig = signature(source)
                if not dest.exists() and previous.get(key) != sig:
                    previous[key] = sig
                    results[key] = dict(state='waiting_for_stable_file')
                    continue
                if not dest.exists() and checked.get(key) == sig:
                    continue
                result = capture_once(source, dest, args.target)
                results[key] = result
                if result['state'] == 'waiting':
                    checked[key] = sig
                print(key, json.dumps(result), flush=True)
            except RuntimeError as exc:
                if 'missed target=' in str(exc) or 'existing destination' in str(exc):
                    results[key] = dict(state='error', error=str(exc))
                else:
                    results[key] = dict(state='retry', error=str(exc))
                print(key, json.dumps(results[key]), flush=True)
            except Exception as exc:
                results[key] = dict(state='retry', error=repr(exc))
                print(key, json.dumps(results[key]), flush=True)
        status = dict(target=args.target, updated_at=time.time(), runs=results)
        tmp = args.status.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, indent=2)+'\n')
        os.replace(tmp, args.status)
        if all(results.get(str(r), {}).get('state') in ('saved', 'error') for r in runs):
            if any(v['state'] == 'error' for v in results.values()):
                raise SystemExit('CAPTURE_FAILED: see status JSON')
            print('ALL_CHECKPOINTS_CAPTURED', flush=True)
            return
        if time.monotonic()-started > args.timeout_hours*3600:
            raise SystemExit('CAPTURE_TIMEOUT: see status JSON')
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
