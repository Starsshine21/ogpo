"""CARL v2: fully offline Sparse Reward Policy Alignment. No environment execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from evaluate_ogpo_critic_v2 import ROOT, TASKS, atomic_json, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True, help='Fixed candidate bank; simulator artifacts are not read')
    p.add_argument('--report-dir', type=Path, required=True, help='Fresh output directory, preserves historical reports')
    p.add_argument('--critic-spec', action='append', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    started = time.time()
    source = args.root.resolve()
    original = json.loads((source/'candidate_manifest.json').read_text())
    cache = Path(original['source_candidate_cache'])
    h = hashlib.sha256()
    with cache.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest() != original['candidate_cache_sha256']:
        raise ValueError('candidate cache changed')
    root = args.report_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    # Export only current offline protocol fields, not obsolete simulator metadata.
    manifest = {k:original[k] for k in ['source_candidate_cache','candidate_cache_sha256','seed','states_per_task','records','terminal_replay','logged_replay','base_actor_checkpoint','base_actor_sha256'] if k in original}
    manifest.update(tasks=original.get('tasks',list(TASKS)), protocol='CARL v2 Sparse Reward Policy Alignment',
        source_bank=str(source), source_manifest_sha256=hashlib.sha256((source/'candidate_manifest.json').read_bytes()).hexdigest())
    if original.get('sampling_note'):manifest['sampling_note']=original['sampling_note']
    atomic_json(root/'candidate_manifest.json',manifest)
    output = root/'critic_values'
    cmd = [sys.executable,str(ROOT/'scripts/evaluate_ogpo_critic_v2.py'),'--output-dir',str(output),
        '--candidate-cache',str(cache),'--states-per-task',str(manifest['states_per_task']),
        '--seed',str(manifest['seed']),'--device',args.device,'--task-manifest',str(root/'candidate_manifest.json')]
    for key,flag in [('terminal_replay','--terminal-replay'),('logged_replay','--logged-replay')]:
        if key in manifest:cmd += [flag,manifest[key]]
    for spec in args.critic_spec:cmd += ['--critic-spec',spec]
    subprocess.run(cmd,check=True)
    evaluated = json.loads((output/'candidate_manifest.json').read_text())
    if evaluated['records'] != manifest['records']:
        raise ValueError('scored states/actions differ from fixed manifest')
    with (output/'per_task_metrics.csv').open() as f:source_rows=list(csv.DictReader(f))
    with (output/'checkpoint_summary.csv').open() as f:times={r['checkpoint']:float(r['evaluation_seconds']) for r in csv.DictReader(f)}
    rows=[]
    for r in source_rows:rows.append({k:(v if k in ['checkpoint','task'] else float(v)) for k,v in r.items()})
    summaries=[]
    for spec in args.critic_spec:
        label=spec.split('::')[0];subset=[r for r in rows if r['checkpoint']==label]
        assert len(subset)==len(manifest['tasks'])
        summary={'checkpoint':label}
        for key in subset[0]:
            if key in ['checkpoint','task']:continue
            vals=np.array([r[key] for r in subset],dtype=float)
            if key.endswith('_count'):
                summary[key]=int(vals.sum())
            else:
                summary[key]=float(vals.mean())
                if not np.isfinite(vals).all():summary[key+'_defined_tasks']=int(np.isfinite(vals).sum())
        summary['evaluation_seconds']=times[label];summaries.append(summary)
    write_csv(root/'CARL_results.csv',summaries)
    write_csv(root/'CARL_per_task.csv',rows)
    atomic_json(root/'CARL_results.json',{'protocol':manifest,'macro':summaries,'per_task':rows})
    groups={
        'Sparse Reward Policy Alignment':['success_state_count','failure_state_count','success_logged_advantage_mean','success_logged_advantage_positive_fraction','failure_logged_advantage_mean','failure_logged_advantage_positive_fraction','success_failure_advantage_separation'],
        'Value Fidelity':['mc_spearman','rmse','terminal_q_mean','terminal_error'],
        'Candidate Reliability':['top1_agreement','pairwise_ranking_consistency','normalized_ranking_margin','candidate_disagreement'],
        'Advantage Density / Ensemble Diagnostics':['ca_nonzero_fraction','median_abs_advantage','mean_abs_advantage','q_head_correlation','ensemble_std']}
    def table(data,fields):
        cols=['checkpoint']+fields
        def fmt(v):return v if isinstance(v,str) else ('N/A' if not np.isfinite(v) else f'{v:.6g}')
        return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(fmt(r[c]) for c in cols)+' |' for r in data])
    report='# CARL v2 — Sparse Reward Policy Alignment\n\n'
    if manifest.get('sampling_note'):report+='Sampling note: '+manifest['sampling_note']+'\n\n'
    report+='完全离线：不执行candidate到环境。A_logged = raw10_mean Q(s,a_logged) − mean over G=4 base candidates of raw10_mean Q(s,a_i)。成功/失败标签来自该state所属历史episode，不是单步动作标签。正值比例严格使用A>0。\n\n'
    report+='各task等权macro，样本数列为总数。某task缺少成功或失败样本时对应指标和严格macro为N/A；定义覆盖数见CSV。不以剩余task的平均值冒充完整macro。CA幅度/密度和head一致率不是越高越好。\n\n'
    report+='Alignment只比较同一state的logged与candidate动作。Value Fidelity可使用独立logged anchors；来源见manifest。该指标是基于轨迹结果分组的策略相对评分诊断，不证明候选动作因果优劣或actor提升。\n\n'
    for title,fields in groups.items():report+='## '+title+' — Macro\n\n'+table(summaries,fields)+'\n\n'
    for task in manifest['tasks']:
        report+='## '+task+'\n\n'
        for title,fields in groups.items():report+='### '+title+'\n\n'+table([r for r in rows if r['task']==task],fields)+'\n\n'
    report+=f'总耗时：{time.time()-started:.1f}s。每模型耗时：{json.dumps(times)}。Raw Q、logged Q、来源episode outcome与逐state advantage均在critic_values/<model>/保存。\n'
    (root/'CARL_REPORT.md').write_text(report)
    print('CARL_ALIGNMENT_COMPLETE',root,flush=True)


if __name__=='__main__':main()
