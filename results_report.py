"""Read-only metric analysis; exports CSV tables, PNG/PDF figures and an HTML index."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml


def divide(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)


def statistics(metric, classes):
    cm = np.asarray(metric['confusion'], dtype=float)
    if cm.shape != (len(classes), len(classes)) or (cm < 0).any() or not np.isfinite(cm).all():
        raise ValueError('Invalid confusion matrix')
    tp, support, predicted = cm.diagonal(), cm.sum(1), cm.sum(0)
    total = cm.sum()
    iou = divide(tp, support + predicted - tp)
    dice = divide(2 * tp, support + predicted)
    precision, recall = divide(tp, predicted), divide(tp, support)
    accuracy = float(divide(tp.sum(), total))
    expected = float(divide((support * predicted).sum(), total ** 2))
    mean = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
    if metric.get('miou') is not None and not np.isclose(mean(iou), metric['miou']):
        raise ValueError('Stored mIoU disagrees with confusion matrix')
    rows = []
    for c, name in enumerate(classes):
        boundary = metric.get('boundary_iou', [None] * len(classes))[c]
        rows.append(dict(class_name=name, support_pixels=int(support[c]), predicted_pixels=int(predicted[c]),
                         precision_pct=100 * precision[c], recall_pct=100 * recall[c],
                         dice_f1_pct=100 * dice[c], iou_pct=100 * iou[c],
                         boundary_iou_pct=100 * boundary if boundary is not None else np.nan))
    summary = dict(pixel_accuracy_pct=100 * accuracy, miou_pct=100 * mean(iou),
                   foreground_miou_pct=100 * mean(iou[1:]), dice_pct=100 * mean(dice),
                   mean_class_recall_pct=100 * mean(recall), macro_precision_pct=100 * mean(precision),
                   frequency_weighted_iou_pct=100 * float(np.nansum(divide(support, total) * iou)) if total else np.nan,
                   cohen_kappa=float(divide(accuracy - expected, 1 - expected)),
                   boundary_iou_pct=100 * metric['mean_boundary_iou'] if metric.get('mean_boundary_iou') is not None else np.nan,
                   loss=metric.get('loss'), valid_pixels=int(total))
    return cm, rows, summary


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(run, evaluation, output):
    run, evaluation, output = Path(run), Path(evaluation), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new report output directory to avoid mixing runs')
    specs = yaml.safe_load((run / 'resolved.yaml').read_text(encoding='utf-8'))
    classes = specs['tasks'][0]['dataset']['classes']
    task_names = [t['dataset']['name'] for t in specs['tasks']]
    metrics = json.loads(evaluation.read_text(encoding='utf-8'))
    if 'matrix' in metrics or not set(metrics).issubset(task_names):
        raise ValueError('--evaluation must be the dataset-keyed JSON produced by evaluate.py')
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'{name}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
        figures.append(name + '.png')
    summaries = []
    for name, metric in metrics.items():
        index = task_names.index(name)
        prefix = f'task_{index:02d}'
        cm, rows, summary = statistics(metric, classes)
        summaries.append(dict(dataset=name, **summary))
        write_csv(output / f'{prefix}_per_class.csv', rows)
        write_csv(output / f'{prefix}_confusion_counts.csv',
                  [dict(truth=c, **{f'pred_{classes[j]}': int(cm[i,j]) for j in range(len(classes))}) for i,c in enumerate(classes)])
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(classes))
        for offset, key, label in [(-.25, 'iou_pct', 'IoU'), (0, 'dice_f1_pct', 'Dice/F1'), (.25, 'boundary_iou_pct', 'Boundary IoU')]:
            bars = ax.bar(x + offset, [r[key] for r in rows], .25, label=label)
            ax.bar_label(bars, fmt='%.1f', fontsize=8)
        ax.set(xticks=x, xticklabels=classes, ylim=(0, 105), ylabel='Percent', title=f'{name}: test scores')
        ax.tick_params(axis='x', rotation=20); ax.legend()
        save(fig, prefix + '_class_scores')
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, matrix, title in [(axes[0], divide(cm, cm.sum(1, keepdims=True)) * 100, 'Row-normalized: recall view'),
                                   (axes[1], divide(cm, cm.sum(0, keepdims=True)) * 100, 'Column-normalized: precision view')]:
            im = ax.imshow(matrix, vmin=0, vmax=100, cmap='Blues')
            ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
                   yticklabels=classes, xlabel='Predicted', ylabel='Ground truth', title=title)
            ax.tick_params(axis='x', rotation=35)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    v = matrix[i,j]
                    ax.text(j, i, f'{v:.1f}' if np.isfinite(v) else 'N/A', ha='center', va='center', color='white' if v > 50 else 'black')
            fig.colorbar(im, ax=ax, label='%')
        save(fig, prefix + '_confusion')
        for pidx, panel in enumerate(sorted((evaluation.parent / f'{prefix}_predictions').glob('sample_*.png'))[:10]):
            with Image.open(panel) as image:
                image = image.copy()
            w, h = image.size
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            for j, title in enumerate(['Input', 'Ground truth', 'Prediction']):
                axes[j].imshow(image.crop((j*w//3, 0, (j+1)*w//3, h)))
                axes[j].set_title(title); axes[j].axis('off')
            fig.suptitle(f'{name} / {panel.stem} — white mask areas are ignored')
            save(fig, f'{prefix}_prediction_{pidx:02d}')
    write_csv(output / 'test_summary.csv', summaries)
    history_file = run / 'history.json'
    history = json.loads(history_file.read_text()) if history_file.exists() else []
    flat = []
    for e in history:
        for split in ('training', 'validation'):
            m = e.get(split, {})
            flat.append(dict(task=e['task'], epoch=e['epoch'], split=split, learning_rate=e.get('learning_rate'),
                             **{k:m.get(k) for k in ['loss','total_loss','ewc_penalty','pixel_accuracy','dice','miou','mean_boundary_iou']}))
    write_csv(output / 'epoch_history.csv', flat)
    for i, name in enumerate(task_names):
        entries = [e for e in history if e['task'] == i]
        if not entries: continue
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, key in zip(axes.flat, ['loss','pixel_accuracy','dice','miou','mean_boundary_iou','ewc_penalty']):
            for split in ('training','validation'):
                y = [e.get(split, {}).get(key) for e in entries]
                if any(v is not None for v in y):
                    ax.plot([e['epoch'] for e in entries], [np.nan if v is None else v for v in y], label=split)
            ax.set(title=key, xlabel='Epoch'); ax.grid(alpha=.2)
            if ax.lines: ax.legend()
        fig.suptitle(name + ': learning curves (scores are fractions)')
        save(fig, f'task_{i:02d}_learning')
    continual = run / 'metrics.json'
    note = 'Continual-learning matrix unavailable: training may not have finished both tasks.'
    if continual.exists():
        result = json.loads(continual.read_text())
        matrix = result.get('matrix', [])
        if matrix:
            a = np.full((len(matrix), len(task_names)), np.nan)
            for i,row in enumerate(matrix):
                for j,m in enumerate(row): a[i,j] = m['miou'] if m['miou'] is not None else np.nan
            write_csv(output / 'continual_miou.csv', [dict(after_task=task_names[i], **{n:a[i,j] for j,n in enumerate(task_names)}) for i in range(len(a))])
            forgetting = []
            for j in range(len(matrix)-1):
                past = a[j:-1,j]
                best = np.nanmax(past) if np.isfinite(past).any() else np.nan
                forgetting.append(dict(dataset=task_names[j], forgetting_percentage_points=100*(best-a[-1,j]),
                                       backward_transfer_percentage_points=100*(a[-1,j]-a[j,j])))
            write_csv(output / 'forgetting.csv', forgetting)
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(a*100, vmin=0, vmax=100, cmap='viridis')
            ax.set(xticks=range(len(task_names)), xticklabels=task_names, yticks=range(len(a)),
                   yticklabels=[f'After {task_names[i]}' for i in range(len(a))], title='Task-boundary test mIoU (%)')
            for i in range(len(a)):
                for j in range(len(task_names)):
                    ax.text(j,i,f'{100*a[i,j]:.2f}' if np.isfinite(a[i,j]) else 'N/A',ha='center',va='center',color='white')
            fig.colorbar(im, ax=ax); save(fig, 'continual_miou')
            note = ('Continual figures use run/metrics.json task-boundary checkpoints. '
                    'Standalone evaluation may use a different checkpoint; these sources are not merged.')
    manifest = dict(run=str(run.resolve()), evaluation=str(evaluation.resolve()), classes=classes,
                    evaluated_datasets=list(metrics), note=note,
                    limitations=['No confidence intervals or multi-seed claims from a single run.',
                                 'No automatic claim of equivalence to HQF-Net splits/classes.',
                                 'Metric NaN/blank means undefined; Dice equals classwise F1.',
                                 'No training, replay, or model modification performed.'])
    (output / 'sources.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    links = ''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in sorted(output.glob('*.csv')))
    pictures = ''.join(f'<h2>{html.escape(n)}</h2><img src="{n}" style="max-width:100%">' for n in figures)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Segmentation results</title>'
        '<body style="font-family:Arial;max-width:1300px;margin:30px auto"><h1>Segmentation results</h1>'
        + '<p>' + html.escape(note) + '</p><p>Palette: other gray, building red, woodland green, water blue, road yellow.</p>'
        + '<ul>' + links + '</ul>' + pictures + '</body>', encoding='utf-8')
    print(json.dumps(summaries, indent=2))
    print('Report:', output / 'index.html')
    return summaries


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    generate(args.run, args.evaluation, args.output)
