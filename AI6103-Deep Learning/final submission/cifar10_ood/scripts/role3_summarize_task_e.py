"""Validate and summarize the seed-0 Task E matrix without modifying source runs.

Example:
  python work/role3_summarize_task_e.py --runs-root result/tc2_downloads/runs_role3
Exit 0 means all ten settings are complete or documented as diverged; exit 2
means at least one is pending, incomplete or invalid. A diverged baseline also
returns 2 because comparisons cannot be made. No missing metric is imputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'hcm_mix-k2-a1-decomp_s0_role3_'
OOD = ['CIFAR100', 'TIN', 'MNIST', 'SVHN', 'DTD', 'PLACES365']
COUNTS = dict(test=10000, CIFAR100=10000, TIN=10000, MNIST=10000,
              SVHN=26032, DTD=1880, PLACES365=36500)
SCORES = ['paper_eq2', 'official_cifar', 'official_toy', 'official_moons', 'norm_dev']
METRICS = ['auroc', 'aupr_out', 'aupr_in', 'fpr95', 'fpr95_hcm_repo']
BASE = dict(method='hcm', hcm_loss='official', loss_power=None, lambda_norm=0.0,
            mix_target='decomp', mix_k=2, mix_alpha=1.0, mix_prob=1.0,
            mix_per_sample=False, epochs=100, batch=64, lr=0.1, momentum=0.9,
            wd=1e-4, milestones=[30, 60, 90], gamma=0.1,
            warmup_mode='loss_scale', warmup_epochs=2, warmup_factor=0.1,
            seed=0, amp=False, deterministic=False, fake_data=False, max_steps=0)
VARIANTS = {
    'baseline': {},
    'lr_0p05': {'lr': 0.05},
    'lr_0p2': {'lr': 0.2},
    'warmup_5': {'warmup_epochs': 5},
    'warmup_none': {'warmup_mode': 'none'},
    'lambda_0p1': {'lambda_norm': 0.1},
    'lambda_1': {'lambda_norm': 1.0},
    'hcm_loss_paper': {'hcm_loss': 'paper'},
    'loss_power_1': {'loss_power': 1.0},
    'mix_target_soft': {'mix_target': 'soft'},
}


def read_json(path):
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must contain a JSON object')
    return value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def tail_failure(path):
    if not path.is_file():
        return []
    text = path.read_text(encoding='utf-8', errors='replace')[-12000:]
    markers = ('[diverged]', 'Traceback', 'Error', 'Exception', 'Killed',
               'OUT_OF_MEMORY', 'TIME LIMIT', 'No space left', 'CANCELLED')
    return [line[:600] for line in text.splitlines() if any(m in line for m in markers)][-10:]


def inspect_npz(path, split):
    with np.load(path, allow_pickle=False) as z:
        missing = set(('d', 'R', 'labels', 'logits', 'feat')) - set(z.files)
        if missing:
            raise ValueError(f'{split}: missing arrays {sorted(missing)}')
        d, r, labels = z['d'], z['R'], z['labels']
        n = COUNTS[split]
        if d.shape != (n, 10) or r.shape != (n,) or labels.shape != (n,):
            raise ValueError(f'{split}: expected {n} complete samples; shapes d={d.shape}, R={r.shape}, labels={labels.shape}')
        if not np.isfinite(d).all() or not np.isfinite(r).all():
            raise ValueError(f'{split}: non-finite d/R values')
        dnorm = np.linalg.norm(d.astype(np.float64), axis=1)
        info = dict(samples=n, bytes=path.stat().st_size,
                    mean_R=float(np.mean(r, dtype=np.float64)),
                    std_R=float(np.std(r, dtype=np.float64)),
                    min_R=float(np.min(r)), max_R=float(np.max(r)),
                    negative_R_fraction=float(np.mean(r < 0)),
                    dnorm_above_one_fraction=float(np.mean(dnorm > 1)),
                    mean_dnorm=float(np.mean(dnorm)))
        if split == 'test':
            logits = z['logits']
            if logits.shape != (n, 10) or not np.isfinite(logits).all():
                raise ValueError('test: invalid logits')
            if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, np.arange(10)).all():
                raise ValueError('test: invalid CIFAR-10 labels')
            info['test_acc'] = float(100 * np.mean(logits.argmax(axis=1) == labels))
    return info


def inspect_run(run, label, recompute=None):
    record = dict(label=label, run_name=run.name, path=str(run.resolve()), seed=0,
                  changes=VARIANTS[label], status='pending', issues=[], warnings=[],
                  exports={}, averages={}, per_ood={}, failure_evidence=[], training_complete=False)
    if not run.is_dir():
        record['issues'].append('Run directory does not exist')
        return record
    record['status'] = 'incomplete'
    record['failure_evidence'] = tail_failure(run.parent / '_logs' / f'{label}.log')
    try:
        cfg = read_json(run / 'config.json')
        record['config'] = cfg
        expected = {**BASE, **VARIANTS[label]}
        for key, val in expected.items():
            if key not in cfg or cfg[key] != val:
                record['issues'].append(f'config {key}: expected {val!r}, found {cfg.get(key)!r}')
        if cfg.get('device') != 'cuda':
            record['issues'].append(f"Expected GPU training, found device={cfg.get('device')!r}")
        metrics = read_json(run / 'metrics.json')
        record['training_metrics'] = clean(metrics)
        if metrics.get('diverged') is True:
            record['status'] = 'invalid' if record['issues'] else 'diverged'
            record['divergence_reason'] = 'train.py recorded a non-finite training loss'
            if not record['failure_evidence']:
                record['warnings'].append('Divergence flag exists; no matching error line was found in the copied training log')
            return record
        if metrics.get('diverged') is not False:
            record['issues'].append('metrics.json must explicitly record diverged=false')
        if metrics.get('epochs_completed') != 100:
            record['issues'].append(f"Expected 100 completed epochs, found {metrics.get('epochs_completed')!r}")
        final = run / 'final.pt'
        if not final.is_file() or final.stat().st_size == 0:
            record['issues'].append('Missing or empty final.pt')
        with (run / 'log.csv').open(newline='', encoding='utf-8-sig') as handle:
            logs = list(csv.DictReader(handle))
        epochs = [int(row['epoch']) for row in logs]
        if set(epochs) != set(range(1, 101)) or not epochs or epochs[-1] != 100:
            record['issues'].append('log.csv does not cover all epochs 1-100 with final epoch 100')
        if len(epochs) != len(set(epochs)):
            record['warnings'].append('log.csv has duplicate epoch rows, possibly from an interrupted checkpoint write')
        if any(not math.isfinite(float(row['train_loss'])) for row in logs):
            record['issues'].append('log.csv contains a non-finite training loss')
        for key in ('test_acc', 'test_acc_dir', 'mean_R', 'mean_dnorm', 'mean_u'):
            if not finite(metrics.get(key)):
                record['issues'].append(f'Missing/non-finite final {key}')
            elif logs and not math.isclose(float(logs[-1][key]), metrics[key], abs_tol=1e-8):
                record['issues'].append(f'Final log and metrics.json disagree on {key}')
        record['training_complete'] = not record['issues']
        for split in ['test'] + OOD:
            try:
                record['exports'][split] = inspect_npz(run / 'outputs' / f'{split}.npz', split)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as exc:
                record['issues'].append(str(exc))
        if 'test' in record['exports'] and finite(metrics.get('test_acc')):
            if not math.isclose(record['exports']['test']['test_acc'], metrics['test_acc'], abs_tol=0.011):
                record['issues'].append('Exported test accuracy differs from final training accuracy by more than 0.011 percentage points')
        scores = read_json(run / 'ood_hcm.json')
        for score in SCORES:
            per = scores.get(score, {})
            valid = True
            for dataset in OOD:
                values = per.get(dataset, {})
                for key in METRICS:
                    val = values.get(key)
                    if not finite(val) or not 0 <= val <= 1:
                        record['issues'].append(f'{score}/{dataset}: missing/invalid {key}')
                        valid = False
            if valid:
                avg = {key: float(np.mean([per[d][key] for d in OOD])) for key in METRICS}
                reported = per.get('_avg', {})
                if reported.get('n_sets') != 6:
                    record['issues'].append(f'{score}: _avg.n_sets is not 6')
                for key in METRICS:
                    if not finite(reported.get(key)) or not math.isclose(reported[key], avg[key], abs_tol=1e-9):
                        record['issues'].append(f'{score}: stored six-set average mismatch for {key}')
                record['averages'][score] = avg
                record['per_ood'][score] = {dataset: {key: per[dataset][key] for key in METRICS} for dataset in OOD}
        if recompute and len(record['exports']) == 7 and len(record['per_ood']) == 5:
            eval_run = recompute
            fresh = eval_run(str(run), SCORES)
            for score in SCORES:
                for dataset in OOD:
                    for key in METRICS:
                        if not math.isclose(fresh[score][dataset][key], scores[score][dataset][key], abs_tol=1e-9):
                            record['issues'].append(f'Recomputed metric mismatch: {score}/{dataset}/{key}')
            record['metrics_recomputed'] = True
        record['status'] = 'complete' if not record['issues'] else 'invalid'
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        record['issues'].append(f'{type(exc).__name__}: {exc}')
    return record


def display(value, places=2):
    return f'{value:.{places}f}' if finite(value) else '—'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'role1_hcm_cifar10' / 'runs_role3')
    parser.add_argument('--out', type=Path, default=ROOT / 'result' / 'task_e')
    parser.add_argument('--verify-metrics', action='store_true', help='Recompute all OOD metrics using the project implementation')
    parser.add_argument('--project', type=Path, default=ROOT / 'role1_hcm_cifar10')
    args = parser.parse_args(argv)
    recompute = None
    if args.verify_metrics:
        sys.path.insert(0, str(args.project.resolve()))
        from eval_hcm_ood import eval_run
        recompute = eval_run
    records = [inspect_run(args.runs_root / (PREFIX + label), label, recompute) for label in VARIANTS]
    baseline = records[0]
    if baseline['status'] == 'complete':
        for record in records[1:]:
            if record['status'] == 'complete':
                for key in ('torch', 'amp', 'deterministic'):
                    if record['config'].get(key) != baseline['config'].get(key):
                        record['issues'].append(f'Baseline environment differs: {key}')
                        record['status'] = 'invalid'
    averages, details, settings = [], [], []
    for record in records:
        tm = record.get('training_metrics', {})
        settings.append(dict(label=record['label'], run_name=record['run_name'], status=record['status'],
                             seed=0, changes=json.dumps(record['changes']),
                             epochs_completed=tm.get('epochs_completed'), test_acc_pct=tm.get('test_acc'),
                             train_time_min=tm.get('train_time_min'), mean_R=tm.get('mean_R'),
                             mean_dnorm=tm.get('mean_dnorm'), issues='; '.join(record['issues']),
                             failure_evidence='; '.join(record['failure_evidence'])))
        if record['status'] != 'complete':
            continue
        for score in SCORES:
            for dataset in ['_avg'] + OOD:
                values = record['averages'][score] if dataset == '_avg' else record['per_ood'][score][dataset]
                row = dict(label=record['label'], run_name=record['run_name'], seed=0, n_seeds=1,
                           score=score, dataset=dataset, n_sets=6 if dataset == '_avg' else 1,
                           test_acc_pct=tm['test_acc'], test_mean_R=record['exports']['test']['mean_R'],
                           test_mean_dnorm=record['exports']['test']['mean_dnorm'],
                           train_time_min=tm.get('train_time_min'))
                bvalues = None
                if baseline['status'] == 'complete':
                    bvalues = baseline['averages'][score] if dataset == '_avg' else baseline['per_ood'][score][dataset]
                    row['delta_test_acc_pp'] = tm['test_acc'] - baseline['training_metrics']['test_acc']
                for key in METRICS:
                    row[f'{key}_pct'] = 100 * values[key]
                    row[f'delta_{key}_pp'] = 100 * (values[key] - bvalues[key]) if bvalues else None
                (averages if dataset == '_avg' else details).append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    settled = baseline['status'] == 'complete' and all(r['status'] in ('complete', 'diverged') for r in records)
    manifest = dict(generated_utc=datetime.now(timezone.utc).isoformat(), runs_root=str(args.runs_root.resolve()),
                    complete=settled, seed=0, n_seeds=1, expected_settings=10, expected_ood_sets=OOD,
                    score_formulas=SCORES, records=records,
                    metric_convention='Higher uncertainty means OOD; fpr95 uses ID-positive OpenOOD convention; six-set unweighted arithmetic means; deltas are percentage points',
                    limitations='Single-seed sensitivity; no seed uncertainty estimates or significance claims; final epoch 100, no test-based checkpoint selection')
    (args.out / 'task_e_integrity.json').write_text(json.dumps(clean(manifest), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    setting_fields = ['label', 'run_name', 'status', 'seed', 'changes', 'epochs_completed', 'test_acc_pct', 'train_time_min', 'mean_R', 'mean_dnorm', 'issues', 'failure_evidence']
    metric_fields = ['label', 'run_name', 'seed', 'n_seeds', 'score', 'dataset', 'n_sets', 'test_acc_pct', 'delta_test_acc_pp', 'test_mean_R', 'test_mean_dnorm', 'train_time_min']
    metric_fields += [field for key in METRICS for field in (f'{key}_pct', f'delta_{key}_pp')]
    write_csv(args.out / 'task_e_settings.csv', settings, setting_fields)
    write_csv(args.out / 'task_e_summary.csv', averages, metric_fields)
    write_csv(args.out / 'task_e_by_ood.csv', details, metric_fields)
    lines = ['# 任务 E：HCM 敏感性实验', '',
             '状态：' + ('实验矩阵已完成（发散设置如实保留）。' if settled else '**尚未完成；下表仅呈现已通过完整性检查的结果。**'), '',
             '固定 seed=0，每组 100 epochs。属于单种子敏感性分析，不报告种子标准差或统计显著性。',
             '基准：decomp、k=2、alpha=1、lr=0.1、前 2 轮 loss×0.1、lambda_norm=0、official loss（默认 power=0.75）。',
             '`hcm_loss_paper` 同时改用论文损失族及其默认 power=2，因此是损失定义对比。`loss_power_1` 只改变 official 损失指数。',
             '所有指标使用最终第 100 轮模型。六 OOD 集合等权平均，百分比差值单位为百分点；AUROC/AUPR 越高越好，FPR@95 越低越好。',
             '主表固定 official_cifar；其余公式完整保留在附表，不按测试集效果选择检查点或公式。', '',
             '## 完整性与实验设置', '',
             '| 设置 | 相对基准的参数变化 | 状态 | 已完成 epoch | test accuracy (%) |',
             '|---|---|---|---:|---:|']
    for record in records:
        tm = record.get('training_metrics', {})
        change = ', '.join(f'{k}={v}' for k, v in record['changes'].items()) or '无（同环境重新训练）'
        lines.append(f"| `{record['label']}` | {change} | {record['status']} | {tm.get('epochs_completed', '—')} | {display(tm.get('test_acc'))} |")
    lines += ['', '## 主表：official_cifar（预先固定）', '',
              '| 设置 | 状态 | Accuracy (%) | Δ Accuracy (pp) | AUROC (%) | Δ AUROC (pp) | FPR@95 (%) | Δ FPR@95 (pp) |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    primary = {row['label']: row for row in averages if row['score'] == 'official_cifar'}
    for record in records:
        row = primary.get(record['label'], {})
        vals = [display(row.get(k)) for k in ('test_acc_pct', 'delta_test_acc_pp', 'auroc_pct', 'delta_auroc_pp', 'fpr95_pct', 'delta_fpr95_pp')]
        lines.append('| ' + ' | '.join([record['label'], record['status']] + vals) + ' |')
    lines += ['', '## 六个 OOD 集合的平均指标', '',
              '| 设置 | 公式 | AUROC (%) | Δ AUROC (pp) | AUPR-out (%) | AUPR-in (%) | FPR@95 (%) | Δ FPR@95 (pp) |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in averages:
        lines.append(f"| `{row['label']}` | `{row['score']}` | {display(row['auroc_pct'])} | {display(row['delta_auroc_pp'])} | {display(row['aupr_out_pct'])} | {display(row['aupr_in_pct'])} | {display(row['fpr95_pct'])} | {display(row['delta_fpr95_pp'])} |")
    if not averages:
        lines.append('| 尚无通过完整性检查的结果 | — | — | — | — | — | — | — |')
    lines += ['', '## 问题和发散记录', '']
    for record in records:
        for issue in record['issues'] + record['warnings']:
            lines.append(f"- `{record['label']}`：{issue}")
        if record['status'] == 'diverged':
            lines.append(f"- `{record['label']}`：{record['divergence_reason']}；已完成 {record['training_metrics'].get('epochs_completed')} epochs。")
        for evidence in record['failure_evidence']:
            lines.append(f"- `{record['label']}` 日志证据：`{evidence.replace('`', chr(39))}`")
    lines += ['', '## 交给 Role 6 的文件', '',
              '- `task_e_settings.csv`：十组参数、完成状态、准确率、R 与训练耗时。',
              '- `task_e_summary.csv`：五公式的六集合平均指标及相对基准变化。',
              '- `task_e_by_ood.csv`：五公式 × 六 OOD 的逐集合指标及变化。',
              '- `task_e_integrity.json`：完整性、样本数、原始配置、错误与发散审计。',
              '- 请一并保留各运行的 config.json、metrics.json、log.csv、ood_hcm.json 和训练日志。', '',
              'R 的解释需结合训练目标与本次结果：soft 的半径目标为 1，decomp 的半径目标为混合标签范数；不能仅因某一种公式表现好就认为 R 总是有效或总是无效。',
              '仅有 ID 测试集平均 R≈1 不足以证明各得分排序相同；还需检查逐样本 R、OOD 的 R 分布及方向范数。完整性 JSON 保留各 split 的 R 标准差、范围、负值比例及方向范数大于 1 的比例。',
              'AUPR 使用 average precision（AP）；out 以 OOD 为阳性，in 以 ID 为阳性。FPR 主表使用 `fpr95`（ID 阳性，首个达到至少 95% ID 召回的 ROC 点）；CSV 另保留反方向的 `fpr95_hcm_repo`，不可混用。', '']
    (args.out / 'task_e_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'out': str(args.out.resolve()), 'settled': settled,
                      'statuses': {r['label']: r['status'] for r in records}}, ensure_ascii=False, indent=2))
    return 0 if settled else 2


if __name__ == '__main__':
    raise SystemExit(main())
