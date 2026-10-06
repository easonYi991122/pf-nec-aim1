"""Exploratory Aim 1b trial scaffolds and treatment-positivity diagnostics."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import warnings
from pf_nec import config
from . import stage12 as a1


DRAFT = a1.DRAFT


INTERVAL_CARDS = {
    'arterial_5': ('ArterialLine', '5'),
    'arterial_1': ('ArterialLine', '1'),
    'arterial_4': ('ArterialLine', '4'),
    'extubation': ('MechVent', None),
    'sternal_closure': ('Sternum', None),
}


STAGE4_IDS = tuple(INTERVAL_CARDS) + ('OR_extubation', 'RSVISmilrin',
                                     'RSVISdopa', 'milrinone_ICU_entry')


PODS = tuple(range(1, 8))


GRACES = (1, 2)


SV_RULE = r'single ventricle|hypoplastic left heart|hlhs|tricuspid atresia|unbalanced.*av|univentricular'


PREOP_GROUPS = {'preop_shock': {'230', '240'}, 'preop_mech_support': {'220'},
                'preop_renal': {'450', '460'}, 'preop_vent': {'470', '600'},
                'preop_sepsis': {'380', '390'}, 'preop_heart_failure': {'670'}}


NUMERIC_STATIC = ('STATscore', 'Calcsurgage', 'Surgwtkg', 'Gestagewks',
                  'BirthWt', 'PrevOpCount')


BINARY_STATIC = ('PretermYN', 'preopECMOyn', 'RScldPre', 'ChromSyndSpecYN',
                 'ExtracardSpecYN', 'PGEsurg')


CATEGORICAL = ('center', 'STATcat', 'cardsurgtype', 'ProcPrimary')


DRUG5 = ('dopa', 'dobut', 'epi', 'norepi', 'milrin')


VIS5_COEFF = (1, 1, 100, 100, 10)


PREOP_DRUG_FIELDS = ('RSDopaPre', 'RSDobutPre', 'RSEpiPre', 'RSNorepiPre', 'RSMilrinPre')


def read_plain_table(name):
    path = config.DATA_ROOT / 'Raw CSV Files' / (name + '.csv')
    data = path.read_bytes()
    try:
        text, encoding = data.decode('utf-8-sig'), 'utf-8-sig'
    except UnicodeDecodeError:
        text, encoding = data.decode('cp1252'), 'cp1252'
    reader = csv.DictReader(io.StringIO(text, newline=''), strict=True)
    rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError('CSVStructureError')
    rows = [{k: v.strip() for k, v in r.items()} for r in rows]
    return rows, {'rows': len(rows), 'encoding': encoding, 'columns': reader.fieldnames}


def make_context(tables):
    index, first, missing, groups = a1.build_context(tables)
    centers = sorted({r['siteidNEW'] for r in index.values() if r.get('siteidNEW')})
    labels = {c: 'C%02d' % (i + 1) for i, c in enumerate(centers)}
    return {'index': index, 'first': first, 'missing_nec': missing, 'groups': groups,
            'centers': centers, 'center_labels': labels}


def preop_codes(ctx, key):
    return {r.get('preopfactor', '') for r in ctx['groups'].get('PreopRiskFactor', {}).get(key, [])
            if r.get('preopfactor', '') not in a1.MISSING}


def select_a1_intervals(ctx, cid):
    """Exact A1 representative_timing selection, including its tied-start rule.

    The selected map stays in memory. Future endpoints are not used to rank
    intervals. A1's structural validity and preoperative-end checks are retained.
    """
    table, site = INTERVAL_CARDS[cid]
    sf, ef = a1.INTERVAL_FIELDS[table]
    selected, exclusions = {}, Counter()
    source_groups = ctx['groups'][table]
    for key, rr in source_groups.items():
        rr = [r for r in rr if site is None or r.get('ArtLineSite') == site]
        if not rr:
            continue
        m = ctx['index'][key]
        op, adm, dis = (a1.get_day(m, f) for f in
                        ('cardsurgdtSHIFT', 'hospadmitdtSHIFT', 'hospdischdtSHIFT'))
        if any(x is None for x in (op, adm, dis)) or m.get('siteidNEW') not in ctx['center_labels']:
            exclusions['missing_index_dates_or_center'] += 1
            continue
        candidates = []
        for r in rr:
            start, end = a1.get_day(r, sf), a1.get_day(r, ef)
            if start is None or (end is not None and end < start) or start < adm or start > dis:
                continue
            if end is not None and end < op:
                continue
            if max(op, start) >= dis:
                continue
            candidates.append((start, r))
        if not candidates:
            exclusions['no_valid_postoperative_interval'] += 1
            continue
        candidates.sort(key=lambda x: x[0])
        start, row = candidates[0]
        if sum(x[0] == start for x in candidates) > 1:
            exclusions['tied_first_start_not_resolved'] += 1
            continue
        if key in ctx['missing_nec']:
            exclusions['incomplete_NEC_date'] += 1
            continue
        if key in ctx['first'] and ctx['first'][key] <= max(op, start):
            exclusions['NEC_before_or_same_day_as_A1_entry'] += 1
            continue
        selected[key] = row
    return selected, {'eligible_A1_entry': len(selected), 'exclusions': dict(exclusions),
                      'A1_entry_definition': 'max(手术日,首个有效区间起点日)；按stage12.py:353规则选首段，非临床准备度。'}


def positive_number(value):
    n = a1.number(value)
    return n if n is not None and n >= 0 else None


def recorded_status(rr, sf, ef, cutoff):
    """Day-level status using only completed-day starts and ends.

    An absent endpoint and an endpoint after cutoff are indistinguishable in
    the history. Do not create a future-end-missingness covariate.
    """
    d = cutoff.date()
    active = 0
    ages = []
    for r in rr:
        start, end = a1.get_day(r, sf), a1.get_day(r, ef)
        if start is None or start >= d:
            continue
        if end is not None and end < d:
            continue
        active += 1
        ages.append((d - start).days)
    return float(active > 0), float(active), float(max(ages)) if ages else 0.0


def last_snapshot(ctx, key, cutoff):
    """Last actual observation strictly before decision, age <=48h; no LOCF series."""
    anchor = ctx['index'][key].get('_icupacuadmitdttmSHIFT')
    allowed = []
    for r in ctx['groups']['RiskSurgVIS'].get(key, []):
        stamp = r.get('_RSVISdttmSHIFT')
        if (stamp is not None and anchor is not None and
                r.get('_icupacuadmitdttmSHIFT') == anchor and anchor <= stamp < cutoff and
                (cutoff - stamp).total_seconds() <= 48 * 3600):
            allowed.append((stamp, r))
    if not allowed:
        return None
    latest = max(t for t, r in allowed)
    rr = [r for t, r in allowed if t == latest]
    return rr[0] if len(rr) == 1 else None


def predecision_covariates(ctx, key, cutoff, mode='daily'):
    """Whitelist; no labels, final durations, discharge, or exposure-window values."""
    m = ctx['index'][key]
    x = {'center': ctx['center_labels'][m['siteidNEW']]}
    for f in NUMERIC_STATIC:
        val = positive_number(m.get(f))
        if f == 'PrevOpCount' and val in (99, 999):
            val = None
        x[f] = val
    for f in BINARY_STATIC:
        x[f] = float(m[f]) if m.get(f) in ('0', '1') else None
    for f in CATEGORICAL[1:]:
        x[f] = m.get(f) if m.get(f) not in a1.MISSING else '__missing__'
    text = m.get('FundDiagnosistxt', '')
    x['single_ventricle_proxy'] = float(bool(re.search(SV_RULE, text, re.I))) if text else None
    codes = preop_codes(ctx, key)
    for name, members in PREOP_GROUPS.items():
        x[name] = float(bool(codes & members)) if codes else None
    pre = [positive_number(m.get(f)) for f in PREOP_DRUG_FIELDS]
    x['preop_vis5'] = sum(c * v for c, v in zip(VIS5_COEFF, pre)) if all(v is not None for v in pre) else None
    # Intraoperative decision lacks an exact timestamp. Only completed preop
    # days are allowed for support; surgery descriptors are known at surgery end.
    support_cutoff = datetime.combine(a1.get_day(m, 'cardsurgdtSHIFT'), datetime.min.time()) if mode == 'OR' else cutoff
    for table, tag in (('MechVent', 'vent'), ('Sternum', 'sternum')):
        sf, ef = a1.INTERVAL_FIELDS[table]
        status, count, age = recorded_status(ctx['groups'][table].get(key, []), sf, ef, support_cutoff)
        x[tag + '_recorded_active_prior_day'] = status
        x[tag + '_active_record_count_prior_day'] = count
        x[tag + '_elapsed_days_prior_day'] = age
    for site in ('1', '4', '5'):
        rr = [r for r in ctx['groups']['ArterialLine'].get(key, []) if r.get('ArtLineSite') == site]
        status, count, age = recorded_status(rr, *a1.INTERVAL_FIELDS['ArterialLine'], support_cutoff)
        x['arterial_' + site + '_recorded_active_prior_day'] = status
        x['arterial_' + site + '_count_prior_day'] = count
    snap = None if mode == 'OR' else last_snapshot(ctx, key, cutoff)
    for drug in DRUG5:
        x['last_' + drug] = positive_number(snap.get('RSVIS' + drug)) if snap else None
    vals = [x['last_' + drug] for drug in DRUG5]
    x['last_vis5'] = sum(c * v for c, v in zip(VIS5_COEFF, vals)) if all(v is not None for v in vals) else None
    x['snapshot_age_hours'] = (cutoff - snap['_RSVISdttmSHIFT']).total_seconds() / 3600 if snap else None
    if mode == 'OR':
        # Structurally unavailable postoperative covariates are omitted, not
        # median-filled from other decisions or from subsequent ICU data.
        for f in ['last_' + drug for drug in DRUG5] + ['last_vis5', 'snapshot_age_hours']:
            del x[f]
    return x


def eligibility_reason(ctx, key, day):
    m = ctx['index'][key]
    op, adm, dis = (a1.get_day(m, f) for f in
                    ('cardsurgdtSHIFT', 'hospadmitdtSHIFT', 'hospdischdtSHIFT'))
    if any(t is None for t in (op, adm, dis)) or m.get('siteidNEW') not in ctx['center_labels']:
        return 'missing_dates_or_center'
    if not adm <= op <= day <= dis:
        return 'not_in_hospital_at_decision_day_or_invalid_chronology'
    if key in ctx['missing_nec']:
        return 'NEC_record_with_undated_event'
    if preop_codes(ctx, key) & {'320', '330'}:
        return 'prior_NEC_in_preop_risk_factors'
    if key in ctx['first'] and ctx['first'][key] < day:
        return 'NEC_before_decision_day'
    return None


def event_inventory(ctx, entries):
    """Only cohort totals: never read A or stratify outcomes by exposure."""
    counts = Counter()
    for key, day in entries:
        m = ctx['index'][key]
        op, dis = a1.get_day(m, 'cardsurgdtSHIFT'), a1.get_day(m, 'hospdischdtSHIFT')
        nec = ctx['first'].get(key)
        if nec is None or nec < day or nec > op + timedelta(days=31):
            continue
        if nec > dis:
            counts['recorded_event_after_discharge_not_eligible'] += 1
        elif nec == day:
            counts['same_decision_day_order_unknown'] += 1
        elif nec == dis:
            counts['same_discharge_day_order_unknown'] += 1
        else:
            counts['strictly_after_decision_before_discharge'] += 1
    return {'denominator_eligible_stays': len(entries),
            **{k: counts[k] for k in ('strictly_after_decision_before_discharge',
               'same_decision_day_order_unknown', 'same_discharge_day_order_unknown',
               'recorded_event_after_discharge_not_eligible')},
            'possible_total_including_order_ambiguity': sum(counts[k] for k in (
                'strictly_after_decision_before_discharge', 'same_decision_day_order_unknown',
                'same_discharge_day_order_unknown')),
            'window': '决策时点至POD31，院内首次登记NEC；同日不强行排序；仅此资格队列总事件，未按策略分组。'}


def unresolved_window(ctx, key, day, last_day):
    """Do not call deaths, discharge or grace-period NEC a deferred strategy."""
    m = ctx['index'][key]
    nec, dis = ctx['first'].get(key), a1.get_day(m, 'hospdischdtSHIFT')
    if nec is not None and day <= nec <= last_day:
        return 'NEC_in_decision_or_grace_window_order_or_adherence_unresolved'
    if dis <= last_day:
        return 'death_or_discharge_in_window_adherence_unresolved'
    return None


def interval_cohort(ctx, cid, selected, pod, grace):
    sf, ef = a1.INTERVAL_FIELDS[INTERVAL_CARDS[cid][0]]
    entries, features, labels, unknown, eligible_centers = [], [], [], Counter(), Counter()
    excluded = Counter()
    for key, row in selected.items():
        op = a1.get_day(ctx['index'][key], 'cardsurgdtSHIFT')
        day = op + timedelta(days=pod)
        why = eligibility_reason(ctx, key, day)
        if why:
            excluded[why] += 1
            continue
        start, end = a1.get_day(row, sf), a1.get_day(row, ef)
        if start >= day:
            excluded['not_started_by_previous_day_end'] += 1
            continue
        if end is not None and end < day:
            excluded['selected_interval_already_ended'] += 1
            continue
        entries.append((key, day))
        eligible_centers[ctx['center_labels'][ctx['index'][key]['siteidNEW']]] += 1
        last_day = day + timedelta(days=grace - 1)
        why = unresolved_window(ctx, key, day, last_day)
        if why is None and end is None:
            why = 'missing_endpoint_cannot_certify_strategy'
        if why:
            unknown[why] += 1
            continue
        features.append(predecision_covariates(ctx, key, datetime.combine(day, datetime.min.time())))
        labels.append(int(end <= last_day))
    return {'entries': entries, 'features': features, 'labels': labels,
            'unknown': dict(unknown), 'eligible_centers': eligible_centers,
            'exclusions': dict(excluded), 'candidate_id': cid, 'decision_POD': pod,
            'grace_calendar_days': grace, 'source_denominator': len(selected),
            'arms': ['A=在宽限期内记录终点', 'B=有效终点在宽限期后（暂保留记录）'],
            'time_zero': 'POD%d日始；A1首段在前一日结束时仍活动。不是首次达到临床安全准备度。' % pod}


def point_cohorts(ctx, cid):
    """All source stays determine the denominator, including absent snapshots.

    Nominal 24h is deliberately anchored to the recorded ICU arrival, with exact
    timestamp validation. It is not relabeled as surgery+24h or fixed POD1.
    """
    buckets = {}
    excluded = Counter()
    for key, m in ctx['index'].items():
        op = a1.get_day(m, 'cardsurgdtSHIFT')
        anchor = m.get('_icupacuadmitdttmSHIFT')
        if op is None or (cid != 'OR_extubation' and anchor is None):
            excluded['missing_time_anchor'] += 1
            continue
        stamp = (datetime.combine(op, datetime.min.time()) if cid == 'OR_extubation' else
                 anchor + timedelta(hours=24) if cid in ('RSVISmilrin', 'RSVISdopa') else anchor)
        day, pod = stamp.date(), (stamp.date() - op).days
        why = eligibility_reason(ctx, key, day)
        if why:
            excluded[why] += 1
            continue
        if pod > 31:
            excluded['decision_after_POD31'] += 1
            continue
        if pod not in buckets:
            buckets[pod] = {'entries': [], 'features': [], 'labels': [], 'unknown': Counter(),
                            'eligible_centers': Counter(), 'exclusions': {},
                            'candidate_id': cid, 'decision_POD': pod,
                            'grace_calendar_days': None, 'source_denominator': len(ctx['index'])}
        b = buckets[pod]
        b['entries'].append((key, day))
        b['eligible_centers'][ctx['center_labels'][m['siteidNEW']]] += 1
        end_of_window = (stamp + timedelta(hours=2)).date() if cid == 'milrinone_ICU_entry' else day
        why = unresolved_window(ctx, key, day, end_of_window)
        value = None
        if cid == 'OR_extubation':
            value = int(m['OpExtubateYN']) if m.get('OpExtubateYN') in ('0', '1') else None
            b['arms'] = ['A=OR/on-arrival合并标记1', 'B=合并标记0']
            b['time_zero'] = 'A1手术结束麻醉评估点不可精确定位；诊断按POD0、术前完整日历史及手术结束时已知类别，排除之后ICU变量。'
        elif cid == 'milrinone_ICU_entry':
            dose = positive_number(m.get('RSMilrin'))
            value = int(dose > 0) if dose is not None else None
            b['arms'] = ['A=入科后首2h峰VIS对应米力农剂量>0', 'B=该峰值时点米力农剂量=0']
            b['time_zero'] = '沿A1入ICU前分配点；入科后首2h峰VIS对应剂量仅为事后测量代理，不能识别预防给药分配。'
        else:
            rr = [r for r in ctx['groups']['RiskSurgVIS'].get(key, []) if r.get('RSVISpoint') == '24']
            if len(rr) != 1:
                why = why or ('nominal24_missing' if not rr else 'nominal24_duplicate')
            elif rr[0].get('_icupacuadmitdttmSHIFT') != anchor or rr[0].get('_RSVISdttmSHIFT') != stamp:
                why = why or 'nominal24_actual_timestamp_or_encounter_mismatch'
            else:
                dose = positive_number(rr[0].get(cid))
                value = int(dose > 0) if dose is not None else None
            b['arms'] = ['A=名义24h且实际时间匹配的剂量>0', 'B=同点剂量=0']
            b['time_zero'] = 'A1预定24h概念的显式诊断锚点=索引ICU入科+24h；分实际POD列，不等同手术+24h；动作仍不可观察。'
        if why is None and value is None:
            why = 'missing_or_invalid_exposure_value'
        if why:
            b['unknown'][why] += 1
            continue
        b['features'].append(predecision_covariates(ctx, key, stamp, 'OR' if cid == 'OR_extubation' else 'point'))
        b['labels'].append(value)
    return [buckets[k] for k in sorted(buckets)], dict(excluded)


def propensity_diagnostics(features, labels):
    """One un-tuned, in-sample logistic treatment fit; only aggregates returned."""
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    n = len(labels)
    counts = Counter(labels)
    base = {'fit_denominator': n, 'A_count': counts[1], 'B_count': counts[0],
            'model': 'L2 logistic, C=1, liblinear, tol=1e-7, max_iter=500; no tuning; in-sample',
            'weights': 'q/p for A; (1-q)/(1-p) for B; q=A/n; only numerical p guard [1e-8,1-1e-8], no clinical trimming',
            'interpretation': '仅已判定代理暴露子集的重叠筛查；不是CCW权重、正式因果权重或条件positivity证明。'}
    if n < 30 or min(counts[0], counts[1]) < 5:
        return {**base, 'status': 'not_fit_small_or_single_arm',
                'reason': '预设至少30条且每臂至少5条；未将单臂或小样本拟合伪装为有重叠。',
                'propensity': None, 'ess': None, 'smd': []}
    numeric = sorted(set(features[0]) - set(CATEGORICAL))
    transformed = [dict() for _ in features]
    missingness = {}
    for f in numeric:
        raw = np.array([np.nan if x.get(f) is None else x[f] for x in features], dtype=float)
        good = np.isfinite(raw)
        median = float(np.median(raw[good])) if good.any() else 0.0
        filled = np.where(good, raw, median)
        scale = float(filled.std()) or 1.0
        centered = (filled - filled.mean()) / scale
        missingness[f] = {'missing': int((~good).sum()), 'denominator': n}
        for i in range(n):
            transformed[i][f] = float(centered[i])
            transformed[i][f + '__missing'] = float(not good[i])
    for f in CATEGORICAL:
        for i, row in enumerate(features):
            transformed[i][f] = str(row.get(f) or '__missing__')
    vectorizer = DictVectorizer(sparse=True, sort=True)
    design = vectorizer.fit_transform(transformed)
    y = np.asarray(labels, dtype=int)
    model = LogisticRegression(C=1.0, solver='liblinear', max_iter=500, tol=1e-7,
                               random_state=2605, fit_intercept=True)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(design, y)
        p = model.predict_proba(design)[:, 1]
    if not np.isfinite(p).all():
        raise ValueError('NonfiniteTreatmentPropensity')
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    clipped = np.clip(p, 1e-8, 1 - 1e-8)
    q = float(y.mean())
    weights = np.where(y == 1, q / clipped, (1 - q) / (1 - clipped))

    def distribution(v):
        return {name: float(np.quantile(v, quantile)) for name, quantile in (
            ('min', 0), ('p01', .01), ('p05', .05), ('p25', .25),
            ('p50', .5), ('p75', .75), ('p95', .95), ('p99', .99), ('max', 1))}

    def ess(v):
        return float(v.sum() ** 2 / np.dot(v, v)) if len(v) else None

    dense = design.toarray()
    before1, before0 = dense[y == 1], dense[y == 0]
    pooled_sd = np.sqrt((before1.var(axis=0) + before0.var(axis=0)) / 2)
    diff = before1.mean(axis=0) - before0.mean(axis=0)
    wdiff = (np.average(before1, axis=0, weights=weights[y == 1]) -
             np.average(before0, axis=0, weights=weights[y == 0]))
    detail = defaultdict(list)
    for j, name in enumerate(vectorizer.get_feature_names_out()):
        main = name.split('=', 1)[0]
        if pooled_sd[j] <= 1e-12:
            before = after = 0.0 if abs(diff[j]) < 1e-12 and abs(wdiff[j]) < 1e-12 else None
        else:
            before, after = abs(float(diff[j] / pooled_sd[j])), abs(float(wdiff[j] / pooled_sd[j]))
        detail[main].append((before, after))
    smd = []
    for f, vals in sorted(detail.items()):
        smd.append({'covariate': f, 'expanded_columns': len(vals),
                    'abs_SMD_before': max((a for a, b in vals if a is not None), default=None),
                    'abs_SMD_after': max((b for a, b in vals if b is not None), default=None),
                    'undefined_zero_pooled_SD_columns': sum(a is None for a, b in vals)})
    return {**base, 'status': 'fitted' if converged else 'fit_not_converged',
            'n_features_after_encoding': int(design.shape[1]),
            'iterations': int(model.n_iter_.max()),
            'numeric_missingness': missingness,
            'propensity': {'quantiles': distribution(p), 'denominator': n,
                           'p_lt_0_05_count': int((p < .05).sum()),
                           'p_gt_0_95_count': int((p > .95).sum()),
                           'p_lt_0_05_fraction': float((p < .05).mean()),
                           'p_gt_0_95_fraction': float((p > .95).mean()),
                           'extreme_fraction': float(((p < .05) | (p > .95)).mean()),
                           'numerical_guard_clipped_count': int((p != clipped).sum()),
                           'numerical_guard_small_observed_arm_probability_count': int(
                               (((y == 1) & (p < 1e-8)) | ((y == 0) & (p > 1 - 1e-8))).sum())},
            'stabilized_weights': distribution(weights),
            'ess': {'total': ess(weights), 'A': ess(weights[y == 1]), 'B': ess(weights[y == 0]),
                    'fraction_of_fitted_n': ess(weights) / n},
            'smd_definition': '绝对均值差/未加权两臂总体方差均值的平方根；加权后沿用该分母；类别报告各哑变量最大值。缺失另列；常数同值=0，完全分离零方差=null并计数。',
            'smd': smd,
            'smd_max_before': max((r['abs_SMD_before'] for r in smd if r['abs_SMD_before'] is not None), default=None),
            'smd_max_after': max((r['abs_SMD_after'] for r in smd if r['abs_SMD_after'] is not None), default=None),
            'smd_undefined_columns': sum(r['undefined_zero_pooled_SD_columns'] for r in smd),
            'smd_covariates_above_0_1_after': sum(r['abs_SMD_after'] is not None and r['abs_SMD_after'] > .1 for r in smd)}


def aggregate_cohort(ctx, cohort):
    n = len(cohort['entries'])
    a_count, b_count = cohort['labels'].count(1), cohort['labels'].count(0)
    unknown = dict(cohort['unknown'])
    assert a_count + b_count + sum(unknown.values()) == n
    counts = defaultdict(Counter)
    for x, a in zip(cohort['features'], cohort['labels']):
        counts[x['center']][a] += 1
    per_center = []
    for label in ctx['center_labels'].values():
        eligible = cohort['eligible_centers'][label]
        ca, cb = counts[label][1], counts[label][0]
        per_center.append({'center_label': label, 'eligible': eligible, 'A': ca, 'B': cb,
                           'unclassified': eligible - ca - cb, 'both_ge5': ca >= 5 and cb >= 5})
    return {k: cohort[k] for k in ('candidate_id', 'decision_POD', 'grace_calendar_days',
                                  'source_denominator', 'arms', 'time_zero', 'exclusions')} | {
        'eligible': n, 'A': a_count, 'B': b_count, 'classified': a_count + b_count,
        'unclassified': sum(unknown.values()), 'unclassified_reasons': unknown,
        'classified_fraction': (a_count + b_count) / n if n else None,
        'center_denominator': len(per_center), 'centers_with_eligible': sum(r['eligible'] > 0 for r in per_center),
        'centers_both_ge5': sum(r['both_ge5'] for r in per_center),
        'centers_single_observed_arm': sum((r['A'] > 0) != (r['B'] > 0) for r in per_center),
        'per_center': per_center, 'events_available_total_only': event_inventory(ctx, cohort['entries']),
        'overlap': propensity_diagnostics(cohort['features'], cohort['labels'])}


def run_diagnostics(tables, expected_a1=None, pods=PODS, graces=GRACES):
    ctx = make_context(tables)
    diagnostics, entry_audits, point_exclusions = [], {}, {}
    for cid in INTERVAL_CARDS:
        selected, audit = select_a1_intervals(ctx, cid)
        if expected_a1 is not None:
            expected = expected_a1['center_timing'][cid]['eligible_stays']
            audit['A1_published_eligible'] = expected
            audit['A1_entry_count_exact_match'] = len(selected) == expected
            if not audit['A1_entry_count_exact_match']:
                raise ValueError('A1EntryDenominatorMismatch')
        entry_audits[cid] = audit
        for pod in pods:
            for grace in graces:
                diagnostics.append(aggregate_cohort(ctx, interval_cohort(ctx, cid, selected, pod, grace)))
    for cid in STAGE4_IDS[len(INTERVAL_CARDS):]:
        cohorts, excluded = point_cohorts(ctx, cid)
        point_exclusions[cid] = excluded
        diagnostics.extend(aggregate_cohort(ctx, c) for c in cohorts)
    return {'index_stays': len(ctx['index']), 'center_count': len(ctx['centers']),
            'A1_entry_reconciliation': entry_audits, 'point_source_exclusions': point_exclusions,
            'diagnostics': diagnostics}


CARD_FEATURES = {
    'arterial_5': ['ArterialLine_5', 'ArterialLine_5_since_index_days_cumsum'],
    'arterial_1': ['ArterialLine_1', 'ArterialLine_1_since_index_days_cumsum'],
    'arterial_4': ['ArterialLine_4'],
    'extubation': ['mechvent', 'mechvent_since_index_days_cumsum', 'offmechvent_since_index_days_cumsum'],
    'sternal_closure': ['closedsternum_since_index_days_cumsum', 'opensternum',
                        'opensternum_since_index_days_cumsum', 'suppS_sternum_current_run_days',
                        'suppS_sternum_days_since_weaning'],
    'OR_extubation': ['opextubateyn'],
    'RSVISmilrin': ['rsmilrin', 'rsvismilrin', 'supp_VIS__milrin_last_observed'],
    'RSVISdopa': ['rsdopa', 'rsvisdopa'],
    'RSVISepi': ['rsepi', 'rsvisepi'],
    'RSVISnorepi': ['rsnorepi', 'rsvisnorepi'],
    'RSVISdobut': ['rsdobut', 'rsvisdobut'],
    'RSVISvasopress': ['rsvasopress', 'rsvisvasopress'],
    'milrinone_ICU_entry': ['rsmilrin', 'rsvismilrin'],
}


S_RISK = {
    'arterial_5': '高：导管依赖循环、灌注/采血需求与病情改善共同决定拔线和NEC；替代线未认证。',
    'arterial_1': '高：恢复速度、侵入监测需求及并存导管；拔一根不等于取消全部监测。',
    'arterial_4': '很高：特殊解剖、术式和重症支持决定部位选择；code4不是中心静脉线。',
    'extubation': '很高：呼吸/循环准备度、镇静和残余病变未测；通气终点可含观察边界。',
    'sternal_closure': '很高：水肿、出血、循环稳定性和残余病变未测；关闭是治疗响应标记。',
    'OR_extubation': '很高：术式、麻醉与拔管准备度选择；合并标记还受术后立即失败影响。',
    'RSVISmilrin': '很高：此前剂量、低灌注、肾功能、LCOS与治疗响应；在用不能表示启动/减量。',
    'RSVISdopa': '很高：中心照护、适应证、此前剂量及联合用药；无用药者可根本无治疗需要。',
    'RSVISepi': '很高：救治指征、低心排与联合支持；低归因不代表无临床意义。',
    'RSVISnorepi': '很高：血管张力、感染、灌注和救治选择；未测适应证不能靠倾向模型补足。',
    'RSVISdobut': '很高：心功能、血流动力学和替代药选择；快照不是随机救治分配。',
    'RSVISvasopress': '很高：救治响应及单位冲突；不能用不可靠剂量定义可执行策略。',
    'milrinone_ICU_entry': '很高：首2h峰VIS剂量在入科后形成，混入治疗响应与择峰过程。',
}


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def stage3_cards(cards, summary, t6_csv, trajectories, permutation):
    """Read R10 aggregates only; do not copy outcome-conditioned counts/rates."""
    assert summary['repeats'] == [1, 2, 3, 4, 5]
    assert {r['candidate_id'] for r in t6_csv} == {r['candidate_id'] for r in summary['T6_candidates']}
    for row in t6_csv:
        source = next(c for c in summary['T6_candidates'] if c['candidate_id'] == row['candidate_id'])
        for key in ('features', 'contribution_and_stability', 'POD_lead_profile', 'trajectory_support'):
            if json.loads(row[key]) != source[key]:
                raise ValueError('R10CSVSummaryMismatch')
    # Exact feature mapping: never transfer the peripheral-line attribution to
    # umbilical or central lines. An unavailable feature has null, not zero.
    out = []
    keep_fields = ('feature', 'mean_abs', 'normalized_share', 'selection_frequency',
                   'repeat_rank_range', 'top10_frequency', 'top20_frequency', 'per_repeat')
    for card in cards:
        features = CARD_FEATURES[card['id']]
        contribution, profiles, perm = {}, {}, []
        for model, data in summary['models'].items():
            by_feature = {r['feature']: r for r in data['overall']['post']['top20']}
            for source in summary['T6_candidates']:
                for row in source['contribution_and_stability'].get(model, []):
                    by_feature[row['feature']] = row
            lag_means = defaultdict(float)
            for row in data['overall']['post'].get('lag_profile', []):
                lag_means[row['feature']] += row['mean_abs']
            contribution[model] = []
            for feature in features:
                if feature in by_feature:
                    row = by_feature[feature]
                    contribution[model].append({k: row.get(k) for k in keep_fields} | {
                        'status': 'R10_full_contribution_and_stability',
                        'included_in_any_model': row.get('included_in_any_model')})
                else:
                    contribution[model].append({'feature': feature, 'mean_abs': lag_means.get(feature),
                         'normalized_share': None, 'top20_frequency': None, 'repeat_rank_range': None,
                         'status': 'R10_lag_sum_only_stability_not_tabulated' if feature in lag_means else
                                   'not_tabulated_no_zero_imputation'})
            profiles[model] = {}
            for axis in ('pod', 'lead'):
                profiles[model][axis] = {}
                for stratum, layer in data[axis].items():
                    selected = {r['feature']: {'feature': r['feature'], 'share': r['normalized_share']}
                                for r in layer['top20'] if r['feature'] in features}
                    for source in summary['T6_candidates']:
                        source_layer = source['POD_lead_profile'].get(model, {}).get(axis, {}).get(stratum, {})
                        for r in source_layer.get('features', []):
                            if r['feature'] in features:
                                selected[r['feature']] = {k: r.get(k) for k in ('feature', 'share', 'status')}
                    profiles[model][axis][stratum] = {
                        'stable_ranking_allowed': bool(layer['rank_ci_support_ok']),
                        'features': [selected.get(f, {'feature': f, 'share': None,
                                                     'status': 'not_tabulated'}) for f in features]}
        for row in permutation:
            mapped = json.loads(row['features']) if row['features'] else (
                [row['group'].split(':', 1)[1]] if row['group'].startswith('feature:') else [])
            if set(mapped) & set(features):
                perm.append({'model': row['model'], 'group': row['group'], 'features': mapped,
                             'mean_auc_loss': a1.number(row['mean_auc_loss']),
                             'ci95': json.loads(row['uncertainty']).get('ci95') if row['uncertainty'] else None,
                             'status': row['status'], 'interpretation': '仅R10原预测任务置换损失，不是效应或当前试验性能。'})
        support = defaultdict(Counter)
        for row in trajectories:
            if (row['feature'] in features and row['alignment'] == 'surgery' and
                    int(row['day']) in (0, 3, 7, 14)):
                key = (row['feature'], int(row['day']))
                # Pool the census measurement denominators over case/control;
                # do not carry either outcome group, its estimates or events.
                for f in ('eligible_rows', 'nonmissing', 'unknown_codes', 'nan_rows', 'unavailable_rows',
                          'adjacent_calendar_pairs', 'available_pairs', 'missingness_switches'):
                    support[key][f] += int(row[f])
        trajectory = [{'feature': f, 'POD': pod, **dict(c)} for (f, pod), c in sorted(support.items())]
        out.append({**card, 'S_risk': S_RISK[card['id']], 'stage3_feature_mapping': features,
                    'prediction_contribution_and_stability': contribution,
                    'when_it_matters_POD_lead': profiles, 'permutation_post': perm,
                    'trajectory_measurement_support_pooled': trajectory,
                    'trajectory_status': '已合并病例/对照测量分母；不复制结局分组估计。' if trajectory else
                                         'R10未专门给出此卡原始轨迹支持；缺席不是零。',
                    'stage3_caveat': '重要性≠效应；低SHAP不删除临床行动。lead是回顾性病例条件归因；1/2/>31天不足稳定排名。药物早期剂量与后续快照不可互换。',
                    'stage4_included': card['id'] in STAGE4_IDS})
    assert len(out) == 13
    return out


def trial_skeleton(card):
    cid = card['id']
    daily = cid in INTERVAL_CARDS
    strategy = {
        'arterial_5': '指定首根脐动脉线在宽限期内真实拔除 vs 保留至宽限期结束；换其他部位线是否允许须预先指定。',
        'arterial_1': '指定首根外周动脉线在宽限期内真实拔除 vs 暂保留；这是单根动作，不能改称撤除全部侵入性监测。',
        'arterial_4': '指定首根中央动脉线（code4，腋/股等动脉，非中心静脉线）拔除 vs 暂保留。',
        'extubation': '按确认方案计划拔管 vs 延后评估；须排除气管切开脱机、非计划拔管及仅结束观察的记录。',
        'sternal_closure': '在宽限期内真实关闭胸骨 vs 暂保留开胸；具体关闭方式、复开胸和手术救治另定。',
        'OR_extubation': '同一手术结束且未拔管评估点：在OR/PACU内拔管 vs 转ICU后按协议评估；地点、转运和失败重插规则待定。',
        'RSVISmilrin': '共同既往米力农治疗、同一指征下，在24h决定点按规定阶梯减量 vs 维持；剂量阶梯和安全阈值待PI指定。',
        'RSVISdopa': '共同既往多巴胺治疗、同一指征下，在24h决定点按规定阶梯减量 vs 维持；不能把不需治疗者当对照。',
        'milrinone_ICU_entry': 'ICU入科分配前：明确预防性米力农方案 vs 临床认可的另一方案；首2h峰VIS对应剂量不能认定分配。',
    }[cid]
    if daily:
        grace = ['1个日历日：POD k日始至k日末', '2个日历日：POD k日始至k+1日末']
        measurement = ('沿A1按起点选首个延伸术后的有效区间，起点并列不任意选；A1描述性入点=max(手术日,起点日)。'
                       '本轮另取POD1–7日始，前日已开始且选定区间未结束；临床首次安全准备度无法重建。'
                       '宽限期内有效终点=A，晚于宽限期的有效终点=B；缺失、窗口内NEC/死亡/出院另列。'
                       '日期终点不是认证动作；非完整观察者仍留在资格分母及事件盘点。')
        assignment = ('拟用clone–censor–weight：在共同时间零克隆至两臂；仅在首次可判定违背分配策略时人工删失，'
                      '对截止该时点的治疗史、病情、中心拟合人工删失权重。宽限期内在尚兼容的克隆中保留NEC与竞争事件；'
                      '不能以存活、仍在ICU或有下一快照作为基线条件。同一住院跨POD重复入选须按住院聚类，正式协议可改为首次准备度。')
    elif cid == 'OR_extubation':
        grace = ['手术结束共同评估至离开OR/PACU前（不是精确24h；时间戳尚缺）']
        measurement = ('只诊断OpExtubateYN的OR/on-arrival合并标记1/0；准确手术结束时间、地点和仍未拔管资格不可观察。'
                       'POD0的同日NEC留在资格队列作为顺序歧义；拟合只用术前完整日历史和结束时已知术式，禁用ICU后VIS/状态。')
        assignment = ('拟在手术结束共同评估点clone–censor–weight，按离开OR/PACU前的真实执行删失不兼容克隆；'
                      '若最终规定瞬时分配，克隆法退化为基线分配方案。缺少准确动作地点/时间，本库不能实施该分配仿真。')
    else:
        grace = ['决定后6h内执行减量或维持（剂量/救治阈值待PI确认）'] if cid != 'milrinone_ICU_entry' else [
            '入科后2h内执行已明确的预防方案；该草案窗口不使峰值快照等同方案分配']
        measurement = ('不可观察动作，仅快照代理。24h卡的诊断锚点明确取索引ICU入科+24h，需名义24和实际时间戳严格一致；'
                       '缺快照者保留在分母。A/B仅为该点剂量>0/=0，并非减量/维持。临床24h若指手术结束+24h，必须换协议并重算。'
                       if cid != 'milrinone_ICU_entry' else
                       '不可观察动作，仅快照代理。用RSMilrin首2h峰VIS对应剂量>0/=0描述可测快照分布；'
                       '入科前协变量中禁用该值，无法辨认入科即时启动/分配。')
        assignment = ('真正动作需先补连续MAR、启动/变速医嘱及共同指征；其后才可在A1时间零clone–censor–weight，'
                      '用决定前历史预测偏离/观察删失。当前点快照A/B不是这些克隆，倾向概率及ESS不支持持续给药或减量效果。')
    return {'candidate_id': cid, 'title': card['title'], 'status': DRAFT,
            'clinical_time_zero_from_A1_unchanged': card['time_zero'],
            'clinical_eligibility_from_A1': card['eligibility'],
            'common_eligibility': '住院、无已知既往NEC（含术前风险因子320/330）、当时尚未死亡/出院，且两策略临床可接受；无NEC记录≠已认证无事件。',
            'strategies': strategy, 'grace_variants': grace,
            'measurement_and_explicit_time_mapping': measurement,
            'assignment_emulation': assignment,
            'followup': '从策略分配的时间零立即开始，至POD31日末、首次NEC或院内死亡/出院；不等宽限期结束再随访。宽限期后按预先规定常规照护。',
            'outcome': '时间零后、POD31内首次符合PI确认Bell标准的术后NEC；本轮仅盘点NEC表最早登记日期，未临床重判。',
            'competing_events': '死亡、出院为院内终点的竞争终止事件；此前NEC/死亡/出院是入选限制。死亡仅有出院状态2及出院日期代理，精确死亡时刻缺失。',
            'same_day_policy': '同日动作、NEC、死亡/出院顺序不能认证。本轮保留资格总分母并给事件可用数范围；正式协议需事件先/动作先两种顺序敏感性，不能直接删除早发事件。',
            'causal_contrast_draft_only': '拟议两策略从时间零至POD31的院内首次NEC累积风险差（总效应）；本轮无任何效应、风险差、hazard或臂别NEC率计算。72h不是本轮目标。',
            'rescue': card['rescue'], 'S_risk': S_RISK[cid],
            'freeze_blockers': ['床旁准备度、血流动力学趋势、肠灌注未测，交换性尚不可信。',
                                '须确认实际动作、救治例外、记录可见时间和同日顺序。',
                                '诊断按可判定代理暴露子集拟合；不能将完成观察的选择过程当作真实随机分配。']}


def draft_dag(card):
    cid = card['id']
    action_short = {'arterial_5': '脐动脉线拔除', 'arterial_1': '外周动脉线拔除',
                    'arterial_4': '中央动脉线拔除', 'extubation': '计划拔管',
                    'sternal_closure': '胸骨关闭', 'OR_extubation': 'OR内拔管',
                    'RSVISmilrin': '米力农减量/维持', 'RSVISdopa': '多巴胺减量/维持',
                    'milrinone_ICU_entry': '入科预防米力农方案'}[cid]
    mediators = {'extubation': '后续呼吸/镇静/循环', 'sternal_closure': '关闭后灌注/感染/循环',
                 'OR_extubation': '转运后呼吸/循环', 'milrinone_ICU_entry': '用药后心排/血压/灌注',
                 'RSVISmilrin': '用药后心排/血压/灌注', 'RSVISdopa': '用药后心排/血压/灌注'}.get(cid, '后续监测/替代线/灌注')
    nodes = {
        'B': 'B 决定前基线\n中心；STAT/术式\n年龄/体重/早产\n单心室代理/术前风险',
        'H': 'H 既往治疗\n既往通气/导管/药物',
        'L': ('L 决定前状态\n术前用药/完整日支持\n禁用随后ICU信息' if cid in ('OR_extubation', 'milrinone_ICU_entry') else
              'L 决定前状态\n既往VIS5快照\n前一完整日支持状态'),
        'U': 'U 未测共同原因\n床旁准备度\n血流动力学趋势/肠灌注',
        'A': 'A 本次行动（草案）\n' + action_short,
        'M': 'M 决定后中介\n' + mediators,
        'Y': 'Y 首次术后NEC\n时间零至POD31',
        'K': 'K 后续竞争事件\n死亡 / 出院',
        'N': 'L+ 后续病情\n受既往A影响',
        'A_next': 'A+ 后续救治\n再插管/再置线/变速',
        'P': 'P 时间零前事件\nNEC / 死亡 / 出院',
        'E': 'E 时间零资格\n限制P未发生\n另需临床安全条件',
        'O': 'O 后续可被观察\n仍在ICU/记录完整',
        'Z': 'Z 登记暴露代理\n终点/合并标记/快照\n不等同实际A',
    }
    main = [('B', 'L'), ('B', 'A'), ('B', 'Y'), ('B', 'K'),
            ('H', 'L'), ('H', 'A'), ('H', 'Y'),
            ('U', 'L'), ('U', 'A'), ('U', 'Y'), ('U', 'K'),
            ('L', 'A'), ('L', 'Y'), ('L', 'K'),
            ('A', 'M'), ('A', 'Y'), ('A', 'K'), ('M', 'Y'), ('M', 'K')]
    longitudinal = [('A', 'N'), ('N', 'A_next'), ('N', 'Y'), ('A_next', 'Y')]
    selection = [('B', 'P'), ('H', 'P'), ('U', 'P'), ('P', 'E'),
                 ('A', 'O'), ('U', 'O'), ('K', 'O'), ('A', 'Z'), ('O', 'Z')]
    return {'candidate_id': cid, 'title': card['title'], 'status': DRAFT,
            'time_zero': card['time_zero'], 'nodes': nodes,
            'edge_groups': {'本次决策': main, '时变反馈': longitudinal, '资格与记录选择': selection},
            'all_edges_are_unconfirmed_clinical_hypotheses': True,
            'measured_adjustment_candidates': ['STATscore/STATcat/cardsurgtype/ProcPrimary（结束后才可用）',
                'Calcsurgage/Surgwtkg/Gestagewks/BirthWt/PretermYN',
                'FundDiagnosistxt单心室文本代理（rebuild_v26.py:100规则）',
                'PreopRiskFactor及preopECMOyn/RScldPre等术前字段',
                '前一完整日的通气/开胸/分部位动脉线记录状态；药物用实际时刻严格早于决定的最近快照及距今小时',
                'siteidNEW中心固定效应（报告仅C01等匿名中心标签）'],
            'unmeasured_confounders': ['床旁拔管/拔线/关闭/给药准备度', '血流动力学趋势、乳酸/血压、心室功能',
                                       '肠灌注及未显性NEC前驱征象', '具体动作的适应证、镇静/麻醉、残余病变及救治理由'],
            'do_not_adjust': ['M决定后中介与未来L+/A+不作为本次决定的基线调整项',
                              'O是治疗/病情/竞争事件共同影响的观察选择，不能要求未来仍在ICU或有快照',
                              '累计最终时长、未来出院、未来NEC和决定后峰值剂量不能进倾向模型'],
            'time_varying_note': 'H→L→A→L+→A+；今日L可受既往治疗影响，普通调整所有每日值不能代替纵向g方法。',
            'strategy_specific_confounding': card['additional_confounds'],
            'measurement_note': '倾向模型拟合Z的已观察分布，不证明A可干预、无未测混杂或Z=A。每条箭头均为草案。'}


RANK_ORDER = ('sternal_closure', 'arterial_1', 'arterial_5', 'extubation', 'arterial_4',
              'OR_extubation', 'RSVISmilrin', 'RSVISdopa', 'milrinone_ICU_entry',
              'RSVISepi', 'RSVISnorepi', 'RSVISdobut', 'RSVISvasopress')


RANK_REASONS = {
    'arterial_1': '条件性备选候选：保留原keep，指定部位动作可修改、终点可测，资格/事件资源较多；但固定参照存在明显倾向尾部与加权后不平衡，需认证真拔除/替代线及共同适应证，不能直接推进效应估计。',
    'extubation': '保留但后置：临床动作和预测证据都强，中心双臂多；然而早期一日宽限的未截尾ESS严重坍缩，不能因SHAP高而升主/备。准备度、镇静及记录边界仍须解锁。',
    'arterial_5': '继续保留keep：临床导管动作值得讨论，不因SHAP较弱而降出候选库。首段留置较早结束，后期风险集/事件与中心双臂支持需逐日看。',
    'sternal_closure': '优先讨论主候选：临床关闭动作可修改、日期可测，固定参照的两种宽限均有较好已测重叠；事件较外周线少，尚不能保证功效。POD1表现较差，不冻结最佳日；准备度、水肿/出血及复开胸造成很高S风险。',
    'arterial_4': '保留maybe：部位选择受特殊解剖和重症支持影响，可比范围可能更窄；不可借用外周线SHAP或解释为中心静脉线。',
    'OR_extubation': '后置：临床可改变，但合并OR/on-arrival标记不能认证动作地点及共同时间零；即使中心有两种标记也不是纯OR策略支持。',
    'RSVISmilrin': '仅快照对照：动作不可观察；须补24h减量/维持、共同既往剂量及适应证。当前p反映快照持续性，不是随机减量可行性。',
    'RSVISdopa': '仅快照对照：动作不可观察且中心用药差异显著；既往剂量、适应证及观测选择可支配重叠。',
    'milrinone_ICU_entry': '独立保留、当前不推进：首2h峰值形成于分配后，不能当入科即时治疗；不可与24h问题合并。',
    'RSVISepi': '临床题目保留：需先定义共同救治阈值与可替代方案，低SHAP不是排除理由。',
    'RSVISnorepi': '临床题目保留：缺少动作轨迹、指征和血流动力学趋势，尚不能认证策略。',
    'RSVISdobut': '临床题目保留：原快照测量与临床共同指征需补证；部分稳定性未专门汇总不记为零。',
    'RSVISvasopress': '临床题目保留：先解决单位冲突及真实动作定义；当前不得推进剂量策略。',
}


def rank_cards(cards, diagnostics):
    by_id = {c['id']: c for c in cards}
    ranked = []
    for rank, cid in enumerate(RANK_ORDER, 1):
        rows = [r for r in diagnostics if r['candidate_id'] == cid]
        # POD3 is a prespecified display reference, not the empirically "best"
        # intervention day. Point decisions retain their own actual POD strata.
        reference = next((r for r in rows if r['decision_POD'] == 3 and r['grace_calendar_days'] == 1), None)
        if reference is None and cid not in INTERVAL_CARDS:
            reference = max(rows, key=lambda r: r['eligible'], default=None)
        evidence = None
        if reference:
            evidence = {k: reference[k] for k in ('decision_POD', 'grace_calendar_days', 'eligible',
                         'A', 'B', 'unclassified', 'centers_both_ge5', 'center_denominator', 'events_available_total_only')}
            evidence['overlap'] = {k: reference['overlap'].get(k) for k in (
                'status', 'propensity', 'ess', 'smd_max_before', 'smd_max_after', 'smd_undefined_columns')}
        ranked.append({**by_id[cid], 'screening_rank': rank, 'recommendation_reason': RANK_REASONS[cid],
                       'representative_diagnostic': evidence,
                       'ranking_status': '讨论顺序草案，非最终主问题、非按疗效/事件数择优；原keep/maybe不改。'})
    return ranked


def synthetic_tables(n=120):
    """Only fabricated data, created in memory; identifiers never serialized."""
    op = datetime(2026, 1, 2)
    raw = {name: [] for name in ('IndexSurgHosp', 'MechVent', 'Sternum', 'ArterialLine', 'RiskSurgVIS', 'NEC')}
    raw['PreopRiskFactor'] = []
    def sas(value):
        return value.strftime('%d%b%y:%H:%M:%S').upper()
    for i in range(n):
        key = 'SYNTHETIC_PRIVATE_STAY_%04d' % i
        anchor = op + timedelta(hours=12)
        dis = op + timedelta(days=2 if i % 31 == 0 else 22)
        master = {'hospitalizationidNEW': key, 'siteidNEW': 'SYNTHETIC_CENTER_%d' % (i % 6),
                  'cardsurgdtSHIFT': sas(op), 'icupacuadmitdttmSHIFT': sas(anchor),
                  'hospadmitdtSHIFT': sas(op - timedelta(days=2)), 'hospdischdtSHIFT': sas(dis),
                  'Hospdischstat': '2' if i % 31 == 0 else '1', 'OpExtubateYN': str(int(i % 5 == 0)),
                  'RSOpenChestYN': str(i % 2), 'STATscore': str(1 + i % 7 / 3),
                  'STATcat': str(1 + i % 5), 'cardsurgtype': str(1 + i % 3), 'ProcPrimary': str(100 + i % 4),
                  'Calcsurgage': str(3 + i % 20), 'Surgwtkg': str(2.5 + i % 12 / 10),
                  'Gestagewks': str(34 + i % 7), 'BirthWt': str(2.2 + i % 13 / 10),
                  'PrevOpCount': '0', 'FundDiagnosistxt': 'Single ventricle' if i % 4 == 0 else 'VSD',
                  'RSMilrin': str(.5 if i % 3 else 0)}
        master.update({f: str(int(i % 7 == 0)) for f in BINARY_STATIC})
        master.update({f: '0' for f in PREOP_DRUG_FIELDS})
        raw['IndexSurgHosp'].append(master)
        raw['PreopRiskFactor'].append({'hospitalizationidNEW': key, 'preopfactor': '230' if i % 9 == 0 else '10'})
        raw['MechVent'].append({'hospitalizationidNEW': key, 'ventstartSHIFT': sas(op),
                               'ventendSHIFT': sas(op + timedelta(days=1 + i % 9))})
        if i % 2:
            raw['Sternum'].append({'hospitalizationidNEW': key, 'SternumDtSHIFT': sas(op),
                'SternumClosedDtSHIFT': sas(op + timedelta(days=1 + i % 7)), 'SternumClosed': '1'})
        for site in ('1', '4', '5'):
            if site == '4' and i % 2:
                continue
            duration = 1 + (i % 5 if site == '5' else i % 11)
            raw['ArterialLine'].append({'hospitalizationidNEW': key, 'ArtLineSite': site,
                'artlinestartdtSHIFT': sas(op), 'artlineenddtSHIFT': sas(op + timedelta(days=duration))})
        for point in (6, 12, 18, 24, 48, 72, 96, 120, 144, 168):
            if point == 24 and i % 29 == 0:
                continue
            row = {'hospitalizationidNEW': key, 'icupacuadmitdttmSHIFT': sas(anchor),
                   'RSVISpoint': str(point), 'RSVISdttmSHIFT': sas(anchor + timedelta(hours=point))}
            row.update({'RSVIS' + drug: '0' for drug in DRUG5})
            row['RSVISmilrin'] = str(.5 if (i + point // 6) % 3 else 0)
            row['RSVISdopa'] = str(3 if (i + point // 6) % 4 < 2 else 0)
            raw['RiskSurgVIS'].append(row)
        if i % 19 == 0:
            raw['NEC'].append({'hospitalizationidNEW': key, 'necbelldtSHIFT': sas(op + timedelta(days=6))})
    return {name: a1.prepare_rows(name, rows) if name in a1.DATE_FIELDS else rows for name, rows in raw.items()}

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(run_diagnostics(synthetic_tables(), pods=(1, 3), graces=(1, 2)),
                     ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
