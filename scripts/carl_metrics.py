"""Offline sparse-reward policy alignment; no candidate environment labels."""
import numpy as np


def logged_advantages(candidate_q, logged_q):
    candidate_q = np.asarray(candidate_q, dtype=float)
    logged_q = np.asarray(logged_q, dtype=float)
    if candidate_q.ndim != 3 or candidate_q.shape[1:] != (4, 10) or logged_q.shape != (len(candidate_q), 10):
        raise ValueError('expected candidate Q [N,4,10], same-state logged Q [N,10]')
    if not np.isfinite(candidate_q).all() or not np.isfinite(logged_q).all():
        raise ValueError('nonfinite Q predictions')
    return logged_q.mean(axis=1) - candidate_q.mean(axis=(1, 2))


def policy_alignment(candidate_q, logged_q, successes):
    advantages = logged_advantages(candidate_q, logged_q)
    labels = np.asarray(successes)
    if labels.shape != advantages.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError('expected one binary SOURCE EPISODE outcome per state')
    labels = labels.astype(bool)
    result = {}
    for name, mask in [('success', labels), ('failure', ~labels)]:
        a = advantages[mask]
        result[name + '_state_count'] = int(len(a))
        result[name + '_logged_advantage_mean'] = float(a.mean()) if len(a) else float('nan')
        result[name + '_logged_advantage_positive_fraction'] = float((a > 0).mean()) if len(a) else float('nan')
    result['success_failure_advantage_separation'] = result['success_logged_advantage_mean'] - result['failure_logged_advantage_mean']
    return result
