import json
import pytest
import yaml

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
from results_report import statistics, generate


def test_statistics_and_report_exports(tmp_path):
    metric = {'confusion': [[8, 2], [1, 9]], 'miou': (8/11+9/12)/2,
              'mean_boundary_iou': .5, 'boundary_iou': [.4, .6], 'loss': .4}
    _, rows, summary = statistics(metric, ['a', 'b'])
    assert summary['pixel_accuracy_pct'] == pytest.approx(85)
    assert rows[0]['recall_pct'] == pytest.approx(80)
    assert rows[0]['precision_pct'] == pytest.approx(800/9)
    (tmp_path/'resolved.yaml').write_text(yaml.safe_dump({'tasks': [
        {'dataset': {'name': name, 'classes': ['a', 'b']}} for name in ['one', 'two']]}))
    (tmp_path/'eval.json').write_text(json.dumps({'one': metric, 'two': metric}))
    (tmp_path/'metrics.json').write_text(json.dumps({'matrix': [[metric], [metric, metric]]}))
    (tmp_path/'history.json').write_text(json.dumps([{'task': 0, 'epoch': 1,
        'training': {'loss': .8, 'miou': .4}, 'validation': {'loss': .6, 'miou': .5}}]))
    out = tmp_path/'out'
    generate(tmp_path, tmp_path/'eval.json', out)
    for f in ['test_summary.csv', 'task_00_per_class.csv', 'task_01_confusion.png',
              'continual_miou.csv', 'forgetting.csv', 'index.html', 'task_00_learning.pdf']:
        assert (out/f).is_file()
    with pytest.raises(FileExistsError):
        generate(tmp_path, tmp_path/'eval.json', out)


def test_undefined_scores_are_not_perfect():
    _, rows, summary = statistics({'confusion': [[0, 0], [0, 0]]}, ['a', 'b'])
    import math
    assert math.isnan(summary['miou_pct'])
    assert math.isnan(rows[0]['precision_pct'])


def test_inconsistent_metrics_fail():
    with pytest.raises(ValueError, match='disagrees'):
        statistics({'confusion': [[1, 0], [0, 1]], 'miou': .4}, ['a', 'b'])
