"""Check reader docs and ensure a missing glossary entry cannot pass silently."""
from pathlib import Path
import pytest
from doc_checks import check_docs

ROOT = Path(__file__).resolve().parents[1]


def documentation():
    return {p.relative_to(ROOT).as_posix(): p.read_bytes()
            for p in ROOT.rglob('*') if p.is_file() and
            (p.suffix == '.md' or p.name == 'terminology.json')}


def test_reader_documentation_contract():
    result = check_docs(documentation())
    assert result['status'] == 'PASS'


@pytest.mark.parametrize('glossary', ['docs/GLOSSARY.md', 'docs/en/GLOSSARY.md'])
def test_missing_used_glossary_code_fails(glossary):
    files = documentation()
    lines = files[glossary].decode().splitlines()
    assert any(line.startswith('|`AUROC`|') for line in lines)
    files[glossary] = ('\n'.join(line for line in lines if not line.startswith('|`AUROC`|')) + '\n').encode()
    with pytest.raises(ValueError, match='Glossary coverage missing'):
        check_docs(files)


def test_hierarchical_wording_regression_fails():
    files = documentation()
    # Construct the legacy wording without introducing it into shipped prose.
    files['docs/TEAM_TASKS.md'] += ('\n' + 'control' + 'ler assigns tasks.\n').encode()
    with pytest.raises(ValueError, match='Hierarchy wording'):
        check_docs(files)


def test_unmatched_numeric_revision_fails():
    files = documentation()
    files['docs/RESULTS.md'] += b'\n0.123456789\n'
    with pytest.raises(ValueError, match='Bilingual numeric mismatch'):
        check_docs(files)


def test_new_unlisted_prose_code_fails():
    files = documentation()
    files['docs/TEAM_TASKS.md'] += b'\nMYSTERY42\n'
    with pytest.raises(ValueError, match='Unlisted prose code'):
        check_docs(files)


def test_unexplained_first_use_fails():
    files = documentation()
    files['README.en.md'] = files['README.en.md'].replace(
        b'necrotizing enterocolitis (NEC)', b'NEC', 1)
    with pytest.raises(ValueError, match='Explain abbreviation on first use'):
        check_docs(files)


@pytest.mark.parametrize('code', ['POD02', 'T3', 'logistic.C', 'PreopRiskFactor_<code>',
                                 'funddiagnosis_<code>', 'lcos*', '*_history', 'SpO2'])
def test_review_glossary_entries_are_required(code):
    files = documentation()
    for glossary in ('docs/GLOSSARY.md', 'docs/en/GLOSSARY.md'):
        rows = files[glossary].decode().splitlines()
        files[glossary] = ('\n'.join(r for r in rows if not r.startswith('|`' + code + '`|')) + '\n').encode()
    with pytest.raises(ValueError, match='Glossary coverage missing'):
        check_docs(files)


@pytest.mark.parametrize('wording', ['平' + '级', 'p' + 'eers', 'as ' + 'equals',
                                   '更早' + '开始', 'started ' + 'earlier',
                                   '项目' + '负责人', 'project ' + 'lead'])
def test_relationship_declarations_and_wrong_role_fail(wording):
    files = documentation()
    files['README.md'] += ('\n' + wording + '\n').encode()
    with pytest.raises(ValueError, match='Hierarchy wording'):
        check_docs(files)


def test_pod02_single_day_definition_fails():
    import json
    files = documentation()
    registry = json.loads(files['docs/terminology.json'])
    registry['terms']['POD02']['en'] = 'postoperative day 02'
    files['docs/terminology.json'] = json.dumps(registry).encode()
    with pytest.raises(ValueError, match='POD02 must describe'):
        check_docs(files)


def test_signal_platform_cannot_be_expanded_as_internal_module():
    files = documentation()
    files['docs/en/AIM2_LINKAGE_NOTE.md'] = files['docs/en/AIM2_LINKAGE_NOTE.md'].replace(
        b'external physiological-signal platform', b'historical attribution module')
    with pytest.raises(ValueError, match='T3 signal-platform context'):
        check_docs(files)


def test_new_raw_field_without_glossary_pattern_fails():
    files = documentation()
    files['docs/T5_FIELDS.md'] = files['docs/T5_FIELDS.md'].replace(b'```text\n', b'```text\nUnexplainedClinicalField\n')
    with pytest.raises(ValueError, match='T5 field glossary pattern missing'):
        check_docs(files)


def test_missing_task_contact_fails():
    files = documentation()
    files['docs/TEAM_TASKS.md'] = files['docs/TEAM_TASKS.md'].replace(b'issue', b'channel pending', 1)
    with pytest.raises(ValueError, match='Task card contact channel missing'):
        check_docs(files)
