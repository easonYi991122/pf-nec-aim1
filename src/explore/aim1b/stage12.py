"""Non-promoted Aim 1b measurement scaffolding; no causal-effect fit."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
from pf_nec import config


DATE_FIELDS = {
    'IndexSurgHosp': ('cardsurgdtSHIFT', 'icupacuadmitdttmSHIFT',
                      'hospadmitdtSHIFT', 'hospdischdtSHIFT'),
    'MechVent': ('icupacuadmitdttmSHIFT', 'ventstartSHIFT', 'ventendSHIFT'),
    'Sternum': ('cardsurgdtSHIFT', 'SternumDtSHIFT', 'SternumClosedDtSHIFT'),
    'ArterialLine': ('cardsurgdtSHIFT', 'artlinestartdtSHIFT', 'artlineenddtSHIFT'),
    'RiskSurgVIS': ('icupacuadmitdttmSHIFT', 'RSVISdttmSHIFT'),
    'NEC': ('cardsurgdtSHIFT', 'necbelldtSHIFT'),
    'Therapies': ('cardsurgdtSHIFT', 'icupacuadmitdttmSHIFT', 'CRRTarfDtTmSHIFT',
                  'CRRTPDstartDtSHIFT', 'CRRTPDendDtSHIFT',
                  'CRRTHemodialStartDtSHIFT', 'CRRTHemodialEndDtSHIFT',
                  'CRRTCVVHstartDtSHIFT', 'CRRTCVVHendDtSHIFT'),
}


INTERVAL_FIELDS = {
    'MechVent': ('ventstartSHIFT', 'ventendSHIFT'),
    'Sternum': ('SternumDtSHIFT', 'SternumClosedDtSHIFT'),
    'ArterialLine': ('artlinestartdtSHIFT', 'artlineenddtSHIFT'),
}


MISSING = {'', '.', 'NA', 'N/A'}


MONTHS = {m: i + 1 for i, m in enumerate('jan feb mar apr may jun jul aug sep oct nov dec'.split())}


DRAFT = '草案，待 PI/临床确认'


DRUGS = {'RSVISmilrin': '米力农', 'RSVISdopa': '多巴胺', 'RSVISepi': '肾上腺素',
         'RSVISnorepi': '去甲肾上腺素', 'RSVISdobut': '多巴酚丁胺', 'RSVISvasopress': '血管加压素'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def parse_date(raw):
    """Shifted naive dates; SAS two-digit year pivot 68, no timezone conversion."""
    v = str(raw).strip()
    if v in MISSING:
        return None
    m = re.fullmatch(r'(\d{1,2})-([A-Za-z]{3})-(\d{2})', v)
    clock = (0, 0, 0)
    if m is None:
        m = re.fullmatch(r'(\d{2})([A-Za-z]{3})(\d{2}):(\d{2}):(\d{2}):(\d{2})', v)
        if m:
            clock = tuple(map(int, m.groups()[3:]))
    if m is None:
        return None
    d, mon, yr = m.groups()[:3]
    year = int(yr) + (2000 if int(yr) <= 68 else 1900)
    try:
        return datetime(year, MONTHS[mon.lower()], int(d), *clock)
    except (ValueError, KeyError):
        return None


def number(raw):
    try:
        value = float(raw)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def summary(values, unit='POD（日历日）'):
    x = sorted(values)
    def q(p):
        a = (len(x) - 1) * p
        i = int(a)
        return float(x[i] + (x[min(i + 1, len(x) - 1)] - x[i]) * (a - i))
    return {'n': len(x), 'unit': unit, 'q1': q(.25) if x else None,
            'median': q(.5) if x else None, 'q3': q(.75) if x else None}


def count_metric(count, denominator, unit='住院'):
    assert 0 <= count <= denominator
    return {'count': count, 'denominator': denominator, 'unit': unit}


def frequency(values, unit):
    c = Counter(values)
    return {'denominator': len(values), 'unit': unit,
            'counts': {str(k): c[k] for k in sorted(c)}}


def prepare_rows(table, raw_rows):
    rows = []
    for raw in raw_rows:
        r = {k: str(v).strip() for k, v in raw.items()}
        for c in DATE_FIELDS[table]:
            r['_' + c] = parse_date(r.get(c, ''))
        rows.append(r)
    return rows


def read_table(table):
    p = config.DATA_ROOT / 'Raw CSV Files' / (table + '.csv')
    b = p.read_bytes()
    try:
        text, encoding = b.decode('utf-8-sig'), 'utf-8-sig'
    except UnicodeDecodeError:
        text, encoding = b.decode('cp1252'), 'cp1252'
    reader = csv.DictReader(io.StringIO(text, newline=''), strict=True)
    required = {'hospitalizationidNEW', *DATE_FIELDS[table]}
    if not reader.fieldnames or not required <= set(reader.fieldnames):
        raise ValueError('Missing required fields: ' + table)
    raw = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in raw):
        raise ValueError('Structural CSV failure: ' + table)
    rows = prepare_rows(table, raw)
    inv = {'rows': len(rows), 'encoding': encoding,
           'hospitalizations': len({r['hospitalizationidNEW'] for r in rows} - {''}),
           'blank_hospitalization_rows': sum(not r['hospitalizationidNEW'] for r in rows),
           'columns': reader.fieldnames, 'dates': {}}
    for c in DATE_FIELDS[table]:
        counts = Counter('missing' if r[c] in MISSING else
                         'parsed' if r['_' + c] is not None else 'parse_failure' for r in rows)
        inv['dates'][c] = {k: counts[k] for k in ('missing', 'parsed', 'parse_failure')}
        inv['dates'][c]['denominator_rows'] = len(rows)
    ids = [c for c in reader.fieldnames if c.endswith('idNEW') and c != 'hospitalizationidNEW']
    inv['record_key_excess_rows'] = {
        c: sum(n - 1 for key, n in Counter(r[c] for r in rows).items() if key and n > 1) for c in ids}
    inv['exact_duplicate_excess_rows'] = sum(n - 1 for n in Counter(
        tuple(r[c] for c in reader.fieldnames) for r in rows).values() if n > 1)
    return rows, inv


def get_day(row, field):
    value = row.get('_' + field)
    return value.date() if value is not None else None


def build_context(tables):
    keys = Counter(r['hospitalizationidNEW'] for r in tables['IndexSurgHosp'])
    index = {r['hospitalizationidNEW']: r for r in tables['IndexSurgHosp']
             if r['hospitalizationidNEW'] and keys[r['hospitalizationidNEW']] == 1}
    first = {}
    missing_nec = set()
    for r in tables['NEC']:
        k, d = r['hospitalizationidNEW'], get_day(r, 'necbelldtSHIFT')
        if d is None:
            missing_nec.add(k)
        elif k not in first or d < first[k]:
            first[k] = d
    groups = {}
    for t, rows in tables.items():
        groups[t] = defaultdict(list)
        for r in rows:
            if r['hospitalizationidNEW'] in index:
                groups[t][r['hospitalizationidNEW']].append(r)
    return index, first, missing_nec, groups


def relations(rows, start_field, end_field):
    groups = defaultdict(list)
    invalid = Counter()
    for r in rows:
        s, e = r.get('_' + start_field), r.get('_' + end_field)
        if s is None:
            invalid['missing_or_invalid_start'] += 1
        elif e is None:
            invalid['missing_or_invalid_end'] += 1
        elif e < s:
            invalid['reverse_interval'] += 1
        else:
            groups[r['hospitalizationidNEW']].append((s, e))
    counts = Counter()
    hosp = defaultdict(set)
    for k, rr in groups.items():
        for (sa, ea), (sb, eb) in itertools.combinations(rr, 2):
            kind = ('identical_interval' if (sa, ea) == (sb, eb) else
                    'strict_overlap' if max(sa, sb) < min(ea, eb) else
                    'equal_boundary' if ea == sb or eb == sa else
                    'zero_length_inside' if max(sa, sb) == min(ea, eb) else 'disjoint')
            counts[kind] += 1
            hosp[kind].add(k)
    pairs = sum(len(rr) * (len(rr) - 1) // 2 for rr in groups.values())
    assert sum(counts.values()) == pairs
    return {'row_denominator': len(rows), 'excluded': dict(invalid),
            'pair_denominator': pairs, 'hospitalizations_with_pairs': sum(len(v) > 1 for v in groups.values()),
            'counts': {c: {'pairs': counts[c], 'hospitalizations': len(hosp[c])}
                       for c in ('identical_interval', 'strict_overlap', 'equal_boundary',
                                 'zero_length_inside', 'disjoint')},
            'definition': '同住院所有无序有效区间对；保留重复，分区互斥；不证明器械更换或重启。'}


def nec_order(rows, field, index, first, missing_nec):
    counts = Counter()
    stays = defaultdict(set)
    for r in rows:
        k = r['hospitalizationidNEW']
        d = get_day(r, field)
        if k not in index:
            kind = 'unlinked'
        elif d is None:
            kind = 'event_date_missing_or_invalid'
        elif k in missing_nec:
            kind = 'NEC_date_incomplete'
        elif k not in first:
            kind = 'no_recorded_NEC'
        elif d < first[k]:
            kind = 'before_first_NEC'
        elif d == first[k]:
            kind = 'same_day_ambiguous'
        else:
            kind = 'after_first_NEC'
        counts[kind] += 1
        stays[kind].add(k)
    kinds = ('unlinked', 'event_date_missing_or_invalid', 'NEC_date_incomplete',
             'no_recorded_NEC', 'before_first_NEC', 'same_day_ambiguous', 'after_first_NEC')
    comparable = sum(counts[c] for c in ('before_first_NEC', 'same_day_ambiguous', 'after_first_NEC'))
    return {'row_denominator': len(rows), 'rows_with_dated_event_and_dated_NEC': comparable,
            'counts': {c: {'rows': counts[c], 'hospitalizations': len(stays[c])} for c in kinds},
            'definition': '与住院内最早登记 NEC 日期比较，包含术前 NEC；NEC 仅日粒度，同日无法排序。无 NEC 记录不证明无 NEC。'}


def interval_audit(table, rows, index, first, missing_nec):
    sf, ef = INTERVAL_FIELDS[table]
    linked = [r for r in rows if r['hospitalizationidNEW'] in index]
    groups = defaultdict(list)
    for r in linked:
        groups[r['hospitalizationidNEW']].append(r)
    usable, checks = [], Counter()
    check_den = Counter()
    post_end_pods, span_end_pods = [], []
    for r in linked:
        m = index[r['hospitalizationidNEW']]
        s, e = r['_' + sf], r['_' + ef]
        op, adm, dis = (get_day(m, c) for c in ('cardsurgdtSHIFT', 'hospadmitdtSHIFT', 'hospdischdtSHIFT'))
        if s is not None and e is not None:
            check_den['reverse_interval'] += 1
            checks['reverse_interval'] += int(e < s)
            checks['zero_duration'] += int(e == s)
            check_den['zero_duration'] += 1
        if s is not None and adm is not None:
            check_den['start_before_hospital_admission'] += 1
            checks['start_before_hospital_admission'] += int(s.date() < adm)
        if e is not None and dis is not None:
            check_den['end_after_hospital_discharge'] += 1
            check_den['end_equals_hospital_discharge_day'] += 1
            checks['end_after_hospital_discharge'] += int(e.date() > dis)
            checks['end_equals_hospital_discharge_day'] += int(e.date() == dis)
        if s is not None and m.get('_icupacuadmitdttmSHIFT') is not None:
            for name, hit in [('start_equals_index_CICU_arrival_timestamp', s == m['_icupacuadmitdttmSHIFT']),
                              ('start_equals_index_CICU_arrival_day', s.date() == m['_icupacuadmitdttmSHIFT'].date())]:
                checks[name] += int(hit)
                check_den[name] += 1
        if s is None or (e is not None and e < s) or op is None:
            continue
        usable.append(r)
        if e is not None:
            if e.date() >= op:
                post_end_pods.append((e.date() - op).days)
            if s.date() <= op <= e.date():
                span_end_pods.append((e.date() - op).days)
    spanning = [r for r in usable if get_day(r, sf) <= get_day(index[r['hospitalizationidNEW']], 'cardsurgdtSHIFT')
                and (get_day(r, ef) is None or get_day(r, ef) >= get_day(index[r['hospitalizationidNEW']], 'cardsurgdtSHIFT'))]
    return {'source_rows': len(rows), 'linked_rows': len(linked),
            'unlinked_rows': len(rows) - len(linked),
            'hospitalizations_with_any_record': count_metric(len(groups), len(index)),
            'records_per_hospitalization': frequency([len(groups[k]) for k in index], '住院；原始记录段数，非认证临床次数'),
            'spanning_index_surgery_calendar_day': count_metric(len(spanning), len(usable), '有效起点、非逆序的记录'),
            'spanning_stays': count_metric(len({r['hospitalizationidNEW'] for r in spanning}), len(index)),
            'end_POD_all_postoperative_record_ends': frequency(post_end_pods, '术后有终点的记录'),
            'end_POD_index_spanning_records': frequency(span_end_pods, '跨索引手术日且有终点的记录'),
            'checks': {c: count_metric(checks[c], check_den[c], '可比较记录') for c in check_den},
            'all_pairs': relations(linked, sf, ef),
            'start_vs_first_NEC': nec_order(linked, sf, index, first, missing_nec),
            'end_vs_first_NEC': nec_order(linked, ef, index, first, missing_nec)}


def reinitiation(rows, index, first):
    """Recorded restart proxies: exact timestamps, strict > end, no inference of failure."""
    by = defaultdict(list)
    for r in rows:
        s, e = r['_ventstartSHIFT'], r['_ventendSHIFT']
        if r['hospitalizationidNEW'] in index and s is not None and e is not None and e >= s:
            by[r['hospitalizationidNEW']].append((s, e))
    observations, equal, overlaps = [], 0, 0
    for k, rr in by.items():
        m = index[k]
        op, dis = get_day(m, 'cardsurgdtSHIFT'), get_day(m, 'hospdischdtSHIFT')
        for s, e in rr:
            if op is None or dis is None or e.date() < op or e.date() > dis:
                continue
            others = [x for x in rr if x != (s, e)]
            later = sorted(sa for sa, ea in others if sa > e)
            equal += int(any(sa == e for sa, ea in others))
            overlaps += int(any(sa < e < ea for sa, ea in others))
            observations.append({'end': e, 'discharge_day': dis,
                                 'next': later[0] if later else None,
                                 'before_NEC': k not in first or e.date() < first[k],
                                 'boundary_end': e.date() == dis,
                                 'overlap': any(sa < e < ea for sa, ea in others)})
    out = {'postoperative_end_denominator': len(observations),
           'ends_with_equal_timestamp_restart_ambiguity': equal,
           'ends_with_another_episode_active_at_end': overlaps,
           'definition': '2/7日=48/168小时；通气再次开始为再插管代理，不能区分气管切开/计划再操作/失败拔管。出院仅有日期；完整窗口要求终点+窗口严格早于出院日00:00。分层不构成试验入选或要求未来存活。'}
    for name, rr in [('all_postoperative_record_ends', observations),
                     ('exclude_discharge_day_and_overlap', [r for r in observations if not r['boundary_end'] and not r['overlap']]),
                     ('exclude_boundary_overlap_and_NEC_not_later', [r for r in observations if not r['boundary_end'] and not r['overlap'] and r['before_NEC']])]:
        out[name] = {}
        for days in (2, 7):
            hit = lambda r: r['next'] is not None and 0 < (r['next'] - r['end']).total_seconds() <= days * 86400
            complete = lambda r: r['end'] + timedelta(days=days) < datetime.combine(r['discharge_day'], datetime.min.time())
            # Event observed despite truncated window is known positive; remaining truncated negatives unknown.
            out[name][str(days)] = {
                'observed_restart': count_metric(sum(hit(r) for r in rr), len(rr), '记录终点'),
                'complete_followup': count_metric(sum(complete(r) for r in rr), len(rr), '记录终点'),
                'restart_in_complete_followup': count_metric(sum(hit(r) and complete(r) for r in rr), sum(complete(r) for r in rr), '完整窗口记录终点'),
                'truncated_no_observed_restart': count_metric(sum(not complete(r) and not hit(r) for r in rr), len(rr), '记录终点'),
            }
    return out


def representative_timing(table, rows, index, first, missing_nec, centers, site=None):
    """One first postoperative candidate interval per stay, never chosen by future end."""
    sf, ef = INTERVAL_FIELDS[table]
    grouped = defaultdict(list)
    for r in rows:
        if site is None or r.get('ArtLineSite') == site:
            grouped[r['hospitalizationidNEW']].append(r)
    per = {c: {'eligible': 0, 'timings': [], 'boundary_ends': 0,
               'NEC_same_day_end': 0, 'no_end': 0} for c in centers}
    exclusions = Counter()
    for k, rr in grouped.items():
        if k not in index:
            exclusions['unlinked_stay'] += 1
            continue
        m = index[k]
        op, adm, dis = (get_day(m, c) for c in ('cardsurgdtSHIFT', 'hospadmitdtSHIFT', 'hospdischdtSHIFT'))
        if any(d is None for d in (op, adm, dis)) or m.get('siteidNEW') not in per:
            exclusions['missing_index_dates_or_center'] += 1
            continue
        candidates = []
        for r in rr:
            s, e = get_day(r, sf), get_day(r, ef)
            if s is None or (e is not None and e < s) or s < adm or s > dis:
                continue
            if e is not None and e < op:
                continue
            t0 = max(op, s)
            if t0 >= dis:
                continue
            candidates.append((s, r))
        if not candidates:
            exclusions['no_valid_postoperative_interval'] += 1
            continue
        candidates.sort(key=lambda x: x[0])
        s, r = candidates[0]
        if sum(x[0] == s for x in candidates) > 1:
            exclusions['tied_first_start_not_resolved'] += 1
            continue
        t0, e = max(op, s), get_day(r, ef)
        if k in missing_nec:
            exclusions['incomplete_NEC_date'] += 1
            continue
        if k in first and first[k] <= t0:
            exclusions['NEC_before_or_same_day_as_candidate_time_zero'] += 1
            continue
        c = per[m['siteidNEW']]
        c['eligible'] += 1
        if e is None:
            c['no_end'] += 1
        elif e >= dis:
            c['boundary_ends'] += 1
        elif k in first and e == first[k]:
            c['NEC_same_day_end'] += 1
        elif k in first and e > first[k]:
            exclusions['end_after_NEC_among_eligible'] += 1
        else:
            c['timings'].append((e - op).days)
    entries = []
    for label, center in enumerate(centers, 1):
        c = per[center]
        timing = summary(c['timings']) if len(c['timings']) >= 20 else {
            'n': len(c['timings']), 'unit': 'POD（日历日）', 'q1': None, 'median': None, 'q3': None}
        entries.append({'center_label': f'C{label:02d}', 'eligible_stays': c['eligible'],
                        'observed_end_stays': len(c['timings']), 'end_POD': timing,
                        'missing_end_stays': c['no_end'], 'boundary_end_stays': c['boundary_ends'],
                        'NEC_same_day_end_stays': c['NEC_same_day_end'],
                        'quantiles_suppressed_below_20_observed_stays': len(c['timings']) < 20})
    all_pods = [v for c in per.values() for v in c['timings']]
    return {'table': table, 'arterial_site_code': site,
            'definition': '每住院按起点选首个延伸至术后的有效记录，起点并列不任意选；代理时间零=max(手术日,起点日)，须早于出院且无更早/同日 NEC。不是临床准备度或正式风险集。终点须严格早于出院及首次 NEC；无NEC记录可纳入。终点不参与首段选择。',
            'center_labels': '同一审计内按原中心键排序赋 C01…；不输出中心键或映射。',
            'center_denominator': len(centers),
            'source_stay_denominator': len(grouped), 'index_stay_denominator': len(index),
            'eligible_stays': sum(c['eligible'] for c in per.values()),
            'observed_end_stays': len(all_pods), 'end_POD': summary(all_pods),
            'end_POD_frequency': frequency(all_pods, '每符合代理条件住院一个记录终点'),
            'centers_ge20_eligible': sum(c['eligible'] >= 20 for c in per.values()),
            'centers_ge20_observed': sum(len(c['timings']) >= 20 for c in per.values()),
            'center_median_distribution_ge20_observed': summary(
                [e['end_POD']['median'] for e in entries if e['observed_end_stays'] >= 20], '中心中位POD；各中心等权'),
            'exclusions_or_unobserved': dict(exclusions), 'per_center': entries,
            'warning': '描述性实践差异，非条件 positivity 认证、非工具变量；每中心<20已观察住院不发布时间分位数。'}


def sternum_extra(rows, index, groups):
    linked = [r for r in rows if r['hospitalizationidNEW'] in index]
    flags = Counter(m.get('RSOpenChestYN', '') for m in index.values())
    cross = Counter((m.get('RSOpenChestYN', ''), bool(groups['Sternum'].get(k))) for k, m in index.items())
    status = Counter(r['SternumClosed'] for r in linked)
    consistency = Counter()
    closed_dates = []
    reopen = Counter()
    reopen_stays = set()
    for r in linked:
        has = r['_SternumClosedDtSHIFT'] is not None
        consistency[('flag_' + r['SternumClosed'] + ('_dated' if has else '_undated'))] += 1
        if has and r['SternumClosed'] == '1':
            op = get_day(index[r['hospitalizationidNEW']], 'cardsurgdtSHIFT')
            if op is not None:
                closed_dates.append((get_day(r, 'SternumClosedDtSHIFT') - op).days)
    for k, rr in groups['Sternum'].items():
        opened = sorted((r for r in rr if r['_SternumDtSHIFT'] is not None), key=lambda r: r['_SternumDtSHIFT'])
        for i, r in enumerate(opened[1:], 1):
            prior = [p['_SternumClosedDtSHIFT'] for p in opened[:i]
                     if p['SternumClosed'] == '1' and p['_SternumClosedDtSHIFT'] is not None]
            d = r['_SternumDtSHIFT']
            if any(e < d for e in prior):
                kind = 'later_open_after_recorded_closure'
                reopen_stays.add(k)
            elif any(e == d for e in prior):
                kind = 'same_day_closure_and_later_open_ambiguous'
            else:
                kind = 'additional_open_without_prior_dated_closure'
            reopen[kind] += 1
    return {'OR_arrival_open_sternum_flag': {'denominator': len(index), 'counts': dict(flags)},
            'flag_vs_any_sternum_record': {'denominator': len(index), 'counts': {
                f'flag_{flag}__record_{int(has)}': n for (flag, has), n in cross.items()}},
            'closed_flag': {'denominator_rows': len(linked), 'counts': dict(status)},
            'closed_flag_vs_date': {'denominator_rows': len(linked), 'counts': dict(consistency)},
            'closure_POD_flag1_all_dated': frequency(closed_dates, '关闭标记=1且关闭日期有效的记录'),
            'reopening_proxies': {'additional_open_row_denominator': sum(max(0, len(rr) - 1) for rr in groups['Sternum'].values()),
                                  'counts': dict(reopen),
                                  'stays_with_later_open_after_closure': count_metric(len(reopen_stays), len(groups['Sternum'])),
                                  'definition': '每个后续开胸记录若有此前严格更早的已关闭日期，记为可观察复开胸代理；同日无法排序；未导出地点/原因及encounter完整关联。'}}


def cumulative_values(rr, admission, surgery, through, eod=False):
    """Independent union of recorded intervals, retaining full admission history."""
    intervals = [(r['_ventstartSHIFT'], r['_ventendSHIFT']) for r in rr
                 if r['_ventstartSHIFT'] is not None and
                 (r['_ventendSHIFT'] is None or r['_ventendSHIFT'] >= r['_ventstartSHIFT'])]
    post, off, pre = 0, 0, 0
    out = {}
    day = admission
    while day <= through:
        if eod:
            # EOD state at next midnight: a course ending during this day is off.
            next_midnight = datetime.combine(day + timedelta(days=1), datetime.min.time())
            active = int(any(s < next_midnight and (e is None or e >= next_midnight) for s, e in intervals))
        else:
            active = int(any(s.date() <= day and (e is None or e.date() >= day) for s, e in intervals))
        if day < surgery:
            pre += active
        else:
            post += active
            off += 1 - active
        out[(day - surgery).days] = (active, post, off, pre)
        day += timedelta(days=1)
    return out


def binary_audit(index, field):
    c = Counter(m.get(field, '') for m in index.values())
    return {'denominator': len(index), 'counts': dict(c),
            'known_binary': count_metric(c['0'] + c['1'], len(index))}


def field_numeric_coverage(index, field):
    values = [number(m.get(field, '')) for m in index.values()]
    return {'field': field, 'finite_nonnegative': count_metric(sum(v is not None and v >= 0 for v in values), len(index)),
            'observed_positive': count_metric(sum(v is not None and v > 0 for v in values), len(index)),
            'note': '术后首2小时峰VIS对应时点的剂量，不是入科前/入科即时剂量或最大药物剂量。'}


def synthetic_tables():
    """Small in-memory fixture; no source dates or identifiers written to disk."""
    raw = {t: [] for t in DATE_FIELDS}
    for k in ('101', '102', '103'):
        raw['IndexSurgHosp'].append({
            'hospitalizationidNEW': k, 'siteidNEW': 'SYNTHETIC',
            'cardsurgdtSHIFT': '02-Jan-26', 'hospadmitdtSHIFT': '01-Jan-26',
            'hospdischdtSHIFT': '15-Jan-26', 'icupacuadmitdttmSHIFT': '02JAN26:10:00:00',
            'OpExtubateYN': '0', 'RSOpenChestYN': '0'})
    raw['MechVent'] = [
        {'hospitalizationidNEW': '101', 'ventstartSHIFT': '01JAN26:12:00:00', 'ventendSHIFT': '03JAN26:12:00:00'},
        {'hospitalizationidNEW': '101', 'ventstartSHIFT': '05JAN26:12:00:00', 'ventendSHIFT': '06JAN26:12:00:00'},
        {'hospitalizationidNEW': '102', 'ventstartSHIFT': '02JAN26:10:00:00', 'ventendSHIFT': '03JAN26:12:00:00'},
    ]
    raw['Sternum'] = [
        {'hospitalizationidNEW': '101', 'SternumDtSHIFT': '02-Jan-26', 'SternumClosed': '1', 'SternumClosedDtSHIFT': '04-Jan-26'},
        {'hospitalizationidNEW': '101', 'SternumDtSHIFT': '06-Jan-26', 'SternumClosed': '0', 'SternumClosedDtSHIFT': '.'},
    ]
    raw['NEC'] = [{'hospitalizationidNEW': '102', 'necbelldtSHIFT': '03-Jan-26'}]
    return {t: prepare_rows(t, rr) for t, rr in raw.items()}


def synthetic_run():
    tables = synthetic_tables()
    index, first, missing, groups = build_context(tables)
    result = {'synthetic_only': True,
              'MechVent': interval_audit('MechVent', tables['MechVent'], index, first, missing),
              'Sternum': interval_audit('Sternum', tables['Sternum'], index, first, missing),
              'reinitiation': reinitiation(tables['MechVent'], index, first),
              'sternum_extra': sternum_extra(tables['Sternum'], index, groups),
              'center_timing': representative_timing('MechVent', tables['MechVent'], index, first,
                                                    missing, ['SYNTHETIC'])}
    assert result['MechVent']['records_per_hospitalization']['counts'] == {'0': 1, '1': 1, '2': 1}
    assert result['MechVent']['end_vs_first_NEC']['counts']['same_day_ambiguous']['rows'] == 1
    assert result['reinitiation']['all_postoperative_record_ends']['2']['observed_restart']['count'] == 1
    assert result['sternum_extra']['reopening_proxies']['counts']['later_open_after_recorded_closure'] == 1
    json.dumps(result, allow_nan=False)
    return result


def test_date_missing_and_invalid():
    assert parse_date('.') is None
    assert parse_date('') is None
    assert parse_date('31-Feb-26') is None
    assert parse_date('03JAN26:12:00:00') == datetime(2026, 1, 3, 12)


def test_interval_pairs_partition():
    rows = prepare_rows('MechVent', [
        {'hospitalizationidNEW': '1', 'ventstartSHIFT': '01JAN26:00:00:00', 'ventendSHIFT': '03JAN26:00:00:00'},
        {'hospitalizationidNEW': '1', 'ventstartSHIFT': '02JAN26:00:00:00', 'ventendSHIFT': '04JAN26:00:00:00'},
        {'hospitalizationidNEW': '1', 'ventstartSHIFT': '04JAN26:00:00:00', 'ventendSHIFT': '05JAN26:00:00:00'},
        {'hospitalizationidNEW': '1', 'ventstartSHIFT': '07JAN26:00:00:00', 'ventendSHIFT': '06JAN26:00:00:00'},
    ])
    out = relations(rows, 'ventstartSHIFT', 'ventendSHIFT')
    assert out['pair_denominator'] == 3
    assert out['counts']['strict_overlap']['pairs'] == 1
    assert out['counts']['equal_boundary']['pairs'] == 1
    assert out['excluded']['reverse_interval'] == 1


def test_cumulative_union_and_shape():
    tables = synthetic_tables()
    vals = cumulative_values(tables['MechVent'][:2], datetime(2026, 1, 1).date(),
                             datetime(2026, 1, 2).date(), datetime(2026, 1, 7).date())
    assert len(vals) == 7
    assert all(len(v) == 4 and all(math.isfinite(x) for x in v) for v in vals.values())
    assert vals[1] == (1, 2, 0, 1)
    assert vals[5] == (0, 4, 2, 1)


def test_causality_future_changes_leave_prefix_unchanged():
    rr = synthetic_tables()['MechVent'][:2]
    adm, op, cut = (datetime(2026, 1, d).date() for d in (1, 2, 2))
    before = cumulative_values(rr, adm, op, cut)
    changed = [dict(r) for r in rr]
    changed[0]['_ventendSHIFT'] = datetime(2026, 1, 10, 12)
    changed[1]['_ventstartSHIFT'] = datetime(2026, 1, 9, 12)
    changed[1]['_ventendSHIFT'] = datetime(2026, 1, 11, 12)
    assert cumulative_values(changed, adm, op, cut) == before


def test_end_day_semantics_are_distinct():
    rr = synthetic_tables()['MechVent'][:1]
    adm, op, end = (datetime(2026, 1, d).date() for d in (1, 2, 3))
    assert cumulative_values(rr, adm, op, end)[1][0] == 1
    assert cumulative_values(rr, adm, op, end, True)[1][0] == 0


def test_first_interval_selected_without_future_end():
    tables = synthetic_tables()
    index, first, missing, groups = build_context(tables)
    before = representative_timing('MechVent', tables['MechVent'], index, first, missing, ['SYNTHETIC'])
    changed = [dict(r) for r in tables['MechVent']]
    changed[1]['_ventendSHIFT'] = datetime(2026, 1, 14)
    after = representative_timing('MechVent', changed, index, first, missing, ['SYNTHETIC'])
    assert before == after


def test_small_center_quantiles_suppressed():
    out = synthetic_run()['center_timing']
    assert out['per_center'][0]['end_POD']['median'] is None
    assert out['centers_ge20_observed'] == 0


def test_synthetic_end_to_end():
    out = synthetic_run()
    assert out['synthetic_only']
    assert '101' not in json.dumps(out)

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(synthetic_run(), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
