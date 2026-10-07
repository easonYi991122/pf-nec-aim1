"""Offline checks for bilingual reader docs, shared terminology and peer wording."""
from collections import Counter
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re

NUMBER = re.compile(r'[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?')
PROPER_NAMES = {'Cardio-Thoracic', 'DeWitt', 'II', 'III', 'JavaScript', 'Lopez-Paz',
                'NVIDIA', 'Near-Infrared', 'SHapley', 'Torres-Viera'}
COMMON = {'NEC', 'PC4', 'PI', 'DUA', 'IRB', 'POD', 'AUROC', 'AUC', 'AP', 'OOF',
          'PPV', 'DCA', 'RSS', 'CPU', 'GPU', 'MRN', 'MDD', 'TCN', 'GRU', 'GRU-D',
          'MLP', 'STAT', 'VIS', 'ICU', 'CSV', 'JSON', 'SHAP', 'SMD'}
HIERARCHY = r'平级|peer|as equals|更早开始|started earlier|项目负责人|project lead|主控|controller|实名指派|teammates execute'
REVIEW_TERMS = {'POD02', 'T3', 'formal', 'source_path', 'abc', 'logistic.C',
                'S.parquet', 'F', 'D_I_N', 'severity_proxy_S', 'SpO2',
                'PreopRiskFactor_<code>', 'funddiagnosis_<code>', 'lcos*',
                '*_history', 'z_days_since_*', '*_unknown', 'dxg_fund_*',
                'chromsyndspecyn', 'extracardspecyn'}


def numbers(text):
    return Counter(NUMBER.findall(text.replace('−', '-').replace('–', '-')))


def glossary_rows(text):
    return re.findall(r'^\|`([^`]+)`\|([^\n]+)\|$', text, re.M)


def check_docs(files):
    """Check complete payloads; known-code coverage is not inferred from glossary."""
    docs = {name: data.decode() for name, data in files.items() if name.endswith('.md')}
    registry = json.loads(files['docs/terminology.json'])
    terms = registry['terms']
    if REVIEW_TERMS - set(terms):
        raise ValueError('Review terminology missing: ' + ', '.join(sorted(REVIEW_TERMS - set(terms))))
    if ('POD0–2' not in terms['POD02']['zh'] or
            'POD0–2' not in terms['POD02']['en'] or
            'postoperative day 02' in terms['POD02']['en']):
        raise ValueError('POD02 must describe the inclusive range')
    code_pattern = re.compile(r'(?<![A-Za-z0-9_])(?:' + '|'.join(
        re.escape(code) for code in sorted(terms, key=lambda x: (-len(x), x)))
        + r')(?![A-Za-z0-9_])')
    required = set(terms)
    rows_by_language = {}
    for name in ('docs/GLOSSARY.md', 'docs/en/GLOSSARY.md'):
        rows = glossary_rows(docs[name])
        codes = [code for code, meaning in rows]
        missing = required - set(codes)
        if missing:
            raise ValueError('Glossary coverage missing in ' + name + ': ' + ', '.join(sorted(missing)))
        if len(codes) != len(set(codes)) or codes != sorted(codes, key=lambda x: (x.casefold(), x)):
            raise ValueError('Glossary must have unique alphabetized one-line entries: ' + name)
        if any(not meaning.strip() for code, meaning in rows):
            raise ValueError('Empty glossary definition: ' + name)
        rows_by_language[name] = len(rows)
    uses = {}
    first_uses = {}
    for name, text in docs.items():
        if re.search(HIERARCHY, text, re.I):
            raise ValueError('Hierarchy wording in ' + name)
        if name in ('README.md', 'README.en.md'):
            if len(text.splitlines()) > 70 or re.search(r'〔[^〕]+〕|\[(?:fact|inference|suggestion|Approved decision)\]', text):
                raise ValueError('README reader-first length or tag rule: ' + name)
        used = set(code_pattern.findall(text))
        glossary = 'docs/en/GLOSSARY.md' if '/en/' in name or '.en.' in name else 'docs/GLOSSARY.md'
        available = {code for code, meaning in glossary_rows(docs[glossary])}
        if used - available:
            raise ValueError('Used code missing from glossary in ' + name)
        if 'GLOSSARY' not in name:
            uses[name] = sorted(used)
            prose = re.sub(r'```.*?```', '', text, flags=re.S)
            prose = re.sub(r'\]\([^)]*\)', ']', prose)
            # Exact identifiers and paths remain literal. Their interfaces are
            # documented separately, while abbreviations in running prose have
            # full meanings before first use.
            literal_free = re.sub(r'`[^`]+`', '', prose)
            remainder = code_pattern.sub('', literal_free)
            unknown = {token for token in re.findall(
                r'(?<![A-Za-z0-9_])[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*(?![A-Za-z0-9_])', remainder)
                if len(token) > 1 and (any(c.isdigit() for c in token) or sum(c.isupper() for c in token) > 1)}
            # English plural spellings share the singular glossary entry.
            unknown -= set(registry['plural_aliases']) | PROPER_NAMES
            if unknown:
                raise ValueError('Unlisted prose code in ' + name + ': ' + ', '.join(sorted(unknown)))
            seen = set()
            for match in code_pattern.finditer(prose):
                code = match.group()
                if code not in COMMON or code in seen:
                    continue
                seen.add(code)
                before = prose[max(0, match.start()-250):match.start()]
                if not before.rstrip().rstrip('`').endswith(('(', '（')):
                    raise ValueError('Explain abbreviation on first use in ' + name + ': ' + code)
            first_uses[name] = sorted(seen)
    patterns = [key.replace('<code>', '*') for key in terms if '*' in key or '<code>' in key]
    field_coverage = {}
    for name in ('docs/T5_FIELDS.md', 'docs/en/T5_FIELDS.md'):
        fields = re.search(r'```text\n(.*?)\n```', docs[name], re.S).group(1).splitlines()
        missing = [field for field in fields if field not in terms and
                   not any(fnmatchcase(field, pattern) for pattern in patterns)]
        if missing:
            raise ValueError('T5 field glossary pattern missing: ' + ', '.join(missing))
        field_coverage[name] = len(fields)
    for name, external in [('docs/AIM2_LINKAGE_NOTE.md', '外部生理信号平台'),
                           ('docs/en/AIM2_LINKAGE_NOTE.md', 'external physiological-signal platform')]:
        if external not in docs[name] or re.search(r'(历史归因模块|historical attribution module)\s*[（(]T3', docs[name]):
            raise ValueError('T3 signal-platform context is incorrect: ' + name)
    if 'outcome-conditioned development cohort (formal) effect analysis' in docs['explore/README.en.md']:
        raise ValueError('Ordinary formal adjective was expanded as a cohort')
    for name, label in [('docs/TEAM_TASKS.md', '状态：'), ('docs/en/TEAM_TASKS.md', 'Status:')]:
        cards = re.findall(r'^## T-[2-6][^\n]*\n\s*([^\n]+)', docs[name], re.M)
        if len(cards) != 5 or any(not row.startswith(label) for row in cards):
            raise ValueError('Task cards require first-line status, next step and contact: ' + name)
        if any('issue' not in row for row in cards):
            raise ValueError('Task card contact channel missing: ' + name)
    for name in ('docs/REFERENCES.md', 'docs/en/REFERENCES.md'):
        if '|Schramm - Neonate NEC.pdf|' not in docs[name]:
            raise ValueError('Reference filename must remain literal: ' + name)
    pairs = [('README.md', 'README.en.md'), ('AGENTS.md', 'AGENTS.en.md'),
             ('explore/README.md', 'explore/README.en.md')]
    pairs += [(name, 'docs/en/' + Path(name).name) for name in docs
              if name.startswith('docs/') and not name.startswith('docs/en/')]
    for zh, en in pairs:
        a, b = numbers(docs[zh]), numbers(docs[en])
        if a != b:
            raise ValueError('Bilingual numeric mismatch: ' + zh + ': ' + str((a - b, b - a)))
    return {'status': 'PASS', 'glossary_entries': rows_by_language,
            'human_documents': len(docs), 'bilingual_pairs': len(pairs),
            't5_fields_covered_by_terms_or_patterns': field_coverage,
            'known_codes_used': len(set().union(*(set(v) for v in uses.values()))),
            'uses': uses, 'first_use_checks': first_uses,
            'readme_lines': {p: len(docs[p].splitlines()) for p in ('README.md', 'README.en.md')}}
