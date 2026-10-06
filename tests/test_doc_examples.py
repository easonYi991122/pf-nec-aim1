"""Execute the actual documented Python blocks with synthetic inputs."""
import json
from pathlib import Path
import re
from pf_nec import cli, config
from explore.aim1b import stage12, stage34
from test_evaluator import target_rows, prediction_table


ROOT = Path(__file__).resolve().parents[1]


def blocks(path):
    return re.findall(r'```python\n(.*?)\n```', (ROOT / path).read_text(), re.S)


def test_documented_paired_reference_example(monkeypatch):
    rows = target_rows(n_stays=20)
    def load(model, repeats):
        assert repeats == [1, 2, 3, 4, 5]
        probability = .1 + .7 * rows.y if model == 'GSAFE-LGB' else .5
        return rows.copy(), prediction_table(rows, probability)
    monkeypatch.setattr(cli, 'load_evaluation_inputs', load)
    example = blocks('docs/EVALUATION.md')[0]
    assert example == blocks('docs/en/EVALUATION.md')[0]
    exec(compile(example, 'docs/EVALUATION.md', 'exec'), {})
    result = json.loads((config.RUN_ROOT / 'reference_pair.json').read_text())
    assert result['attempted_draws'] == 2000
    assert result['comparisons'][0]['delta'] == .5
    assert result['family_complete'] is False


def test_documented_aim1b_card_and_raw_adapters(monkeypatch):
    examples = blocks('explore/README.md')
    assert examples == blocks('explore/README.en.md')
    exec(compile(examples[0], 'explore/README.md', 'exec'), {})
    tables = stage34.synthetic_tables(120)
    monkeypatch.setattr(stage12, 'read_table', lambda name: (tables[name], {}))
    monkeypatch.setattr(stage34, 'read_plain_table', lambda name: (tables[name], {}))
    monkeypatch.setattr(config, 'require_data', lambda: config.DATA_ROOT)
    exec(compile(examples[1], 'explore/README.md', 'exec'), {})
    result = json.loads((config.RUN_ROOT / 'aim1b_diagnostics.json').read_text())
    assert result['index_stays'] == 120
    assert len(result['diagnostics']) > 0
