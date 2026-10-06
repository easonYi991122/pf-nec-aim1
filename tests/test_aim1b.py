from copy import deepcopy
from datetime import datetime, timedelta
import json
import pytest
from explore.aim1b import stage34 as s


@pytest.fixture
def tables():
    return s.synthetic_tables(120)


def key(i):
    return 'SYNTHETIC_PRIVATE_STAY_%04d' % i


def test_reuse_date_parser_and_exact_a1_entry(tables):
    ctx = s.make_context(tables)
    for cid, (table, site) in s.INTERVAL_CARDS.items():
        selected, audit = s.select_a1_intervals(ctx, cid)
        original = s.a1.representative_timing(table, tables[table], ctx['index'], ctx['first'],
                                             ctx['missing_nec'], ctx['centers'], site)
        assert len(selected) == original['eligible_stays'] == audit['eligible_A1_entry']
    assert s.a1.parse_date('02JAN26:12:00:00') == datetime(2026, 1, 2, 12)
    assert s.a1.parse_date('.') is None


def test_first_interval_not_selected_by_future_endpoint(tables):
    ctx = s.make_context(tables)
    initial, _ = s.select_a1_intervals(ctx, 'extubation')
    changed = deepcopy(tables)
    row = deepcopy(changed['MechVent'][1])
    row['_ventstartSHIFT'] = datetime(2026, 1, 15)
    row['_ventendSHIFT'] = datetime(2026, 1, 16)
    changed['MechVent'].append(row)
    again, _ = s.select_a1_intervals(s.make_context(changed), 'extubation')
    assert initial == again
    # A same-start duplicate is ambiguous, not silently assigned to one record.
    changed['MechVent'].append(deepcopy(changed['MechVent'][1]))
    tied, audit = s.select_a1_intervals(s.make_context(changed), 'extubation')
    assert key(1) not in tied
    assert audit['exclusions']['tied_first_start_not_resolved'] == 1


def test_future_data_cannot_change_past_covariates_or_diagnostic(tables):
    cutoff = datetime(2026, 1, 5)  # POD3 start
    ctx = s.make_context(tables)
    before = s.predecision_covariates(ctx, key(7), cutoff)
    selected, _ = s.select_a1_intervals(ctx, 'extubation')
    full_before = s.aggregate_cohort(ctx, s.interval_cohort(ctx, 'extubation', selected, 3, 1))
    changed = deepcopy(tables)
    for row in changed['RiskSurgVIS']:
        if row['_RSVISdttmSHIFT'] >= cutoff:
            row['RSVISmilrin'] = '1234'
            row['RSVISdopa'] = '4321'
    for row in changed['MechVent']:
        if row['_ventendSHIFT'].date() > cutoff.date():
            row['_ventendSHIFT'] += timedelta(days=1)
    later = s.make_context(changed)
    assert s.predecision_covariates(later, key(7), cutoff) == before
    selected2, _ = s.select_a1_intervals(later, 'extubation')
    full_after = s.aggregate_cohort(later, s.interval_cohort(later, 'extubation', selected2, 3, 1))
    assert full_after == full_before  # Includes fitted propensity and SMD.


def test_future_vs_missing_endpoint_same_baseline_state():
    r = {'_ventstartSHIFT': datetime(2026, 1, 2), '_ventendSHIFT': datetime(2026, 1, 15)}
    cutoff = datetime(2026, 1, 4)
    assert s.recorded_status([r], 'ventstartSHIFT', 'ventendSHIFT', cutoff) == (
        s.recorded_status([{**r, '_ventendSHIFT': None}], 'ventstartSHIFT', 'ventendSHIFT', cutoff))
    assert s.recorded_status([{**r, '_ventstartSHIFT': cutoff}], 'ventstartSHIFT', 'ventendSHIFT', cutoff) == (0, 0, 0)


def test_actual_snapshot_time_and_encounter_are_required(tables):
    ctx = s.make_context(tables)
    cutoff = ctx['index'][key(1)]['_icupacuadmitdttmSHIFT'] + timedelta(hours=24)
    last = s.last_snapshot(ctx, key(1), cutoff)
    assert last['RSVISpoint'] == '18'
    changed = deepcopy(tables)
    row = next(r for r in changed['RiskSurgVIS'] if r['hospitalizationidNEW'] == key(1) and r['RSVISpoint'] == '24')
    row['_RSVISdttmSHIFT'] += timedelta(hours=1)
    cohorts, _ = s.point_cohorts(s.make_context(changed), 'RSVISmilrin')
    assert sum(c['unknown'].get('nominal24_actual_timestamp_or_encounter_mismatch', 0) for c in cohorts) == 1
    row['_RSVISdttmSHIFT'] -= timedelta(hours=1)
    row['_icupacuadmitdttmSHIFT'] += timedelta(days=1)
    cohorts, _ = s.point_cohorts(s.make_context(changed), 'RSVISmilrin')
    assert sum(c['unknown'].get('nominal24_actual_timestamp_or_encounter_mismatch', 0) for c in cohorts) == 1


def test_snapshot_missing_keeps_time_zero_denominator(tables):
    ctx = s.make_context(tables)
    before, _ = s.point_cohorts(ctx, 'RSVISmilrin')
    changed = deepcopy(tables)
    changed['RiskSurgVIS'] = [r for r in changed['RiskSurgVIS'] if not
                             (r['hospitalizationidNEW'] == key(1) and r['RSVISpoint'] == '24')]
    after, _ = s.point_cohorts(s.make_context(changed), 'RSVISmilrin')
    assert sum(len(c['entries']) for c in after) == sum(len(c['entries']) for c in before)
    assert sum(len(c['labels']) for c in after) == sum(len(c['labels']) for c in before) - 1


def test_or_and_icu_baseline_do_not_use_postdecision_peak_or_status(tables):
    ctx = s.make_context(tables)
    op, icu = datetime(2026, 1, 2), datetime(2026, 1, 2, 12)
    orig_or = s.predecision_covariates(ctx, key(1), op, 'OR')
    orig_icu = s.predecision_covariates(ctx, key(1), icu, 'point')
    for field in ('RSMilrin', 'RSDopa', 'RSVIS', 'OpExtubateYN', 'RSOpenChestYN', 'RSfeedPostop'):
        ctx['index'][key(1)][field] = '9999'
    assert s.predecision_covariates(ctx, key(1), op, 'OR') == orig_or
    assert s.predecision_covariates(ctx, key(1), icu, 'point') == orig_icu
    assert not any(f.startswith('last_') for f in orig_or)


def test_grace_nec_death_discharge_not_coded_deferred_or_dropped(tables):
    changed = deepcopy(tables)
    for row in changed['MechVent']:
        if row['hospitalizationidNEW'] in (key(1), key(2), key(3)):
            row['_ventendSHIFT'] = datetime(2026, 1, 12)
    changed['NEC'].append({'hospitalizationidNEW': key(1), '_necbelldtSHIFT': datetime(2026, 1, 5)})
    for row in changed['IndexSurgHosp']:
        if row['hospitalizationidNEW'] in (key(2), key(3)):
            row['_hospdischdtSHIFT'] = datetime(2026, 1, 5)
            row['Hospdischstat'] = '2' if row['hospitalizationidNEW'] == key(2) else '1'
    ctx = s.make_context(changed)
    selected, _ = s.select_a1_intervals(ctx, 'extubation')
    cohort = s.interval_cohort(ctx, 'extubation', selected, 3, 1)
    assert {key(1), key(2), key(3)} <= {k for k, d in cohort['entries']}
    assert cohort['unknown']['NEC_in_decision_or_grace_window_order_or_adherence_unresolved'] >= 1
    assert cohort['unknown']['death_or_discharge_in_window_adherence_unresolved'] >= 2
    report = s.aggregate_cohort(ctx, cohort)
    assert report['eligible'] == report['A'] + report['B'] + report['unclassified']
    assert report['events_available_total_only']['same_decision_day_order_unknown'] == 1


def test_nec_before_decision_and_preop_history_excluded(tables):
    changed = deepcopy(tables)
    changed['NEC'].append({'hospitalizationidNEW': key(1), '_necbelldtSHIFT': datetime(2026, 1, 3)})
    changed['PreopRiskFactor'].append({'hospitalizationidNEW': key(2), 'preopfactor': '320'})
    ctx = s.make_context(changed)
    assert s.eligibility_reason(ctx, key(1), datetime(2026, 1, 5).date()) == 'NEC_before_decision_day'
    assert s.eligibility_reason(ctx, key(2), datetime(2026, 1, 5).date()) == 'prior_NEC_in_preop_risk_factors'


def test_event_inventory_depends_on_eligibility_never_arm(tables):
    ctx = s.make_context(tables)
    selected, _ = s.select_a1_intervals(ctx, 'extubation')
    cohort = s.interval_cohort(ctx, 'extubation', selected, 3, 2)
    before = s.event_inventory(ctx, cohort['entries'])
    cohort['labels'] = [1 - a for a in cohort['labels']]
    assert s.event_inventory(ctx, cohort['entries']) == before
    assert before['denominator_eligible_stays'] == len(cohort['entries'])


def test_propensity_shapes_missingness_smd_and_single_arm():
    features = [{'center': 'C01' if i % 2 else 'C02', 'STATcat': str(i % 3),
                 'cardsurgtype': '1', 'ProcPrimary': '10', 'age': None if i % 13 == 0 else i % 11,
                 'all_missing': None, 'constant': 1.0} for i in range(160)]
    labels = [int(i % 4 < 2) for i in range(160)]
    d = s.propensity_diagnostics(features, labels)
    assert d['status'] == 'fitted'
    assert d['fit_denominator'] == 160
    assert 0 < d['ess']['total'] <= 160 + 1e-7
    assert d['propensity']['denominator'] == 160
    assert d['numeric_missingness']['all_missing']['missing'] == 160
    assert next(r for r in d['smd'] if r['covariate'] == 'constant')['abs_SMD_after'] == 0
    json.dumps(d, allow_nan=False)
    assert s.propensity_diagnostics(features, [1] * 160)['status'] == 'not_fit_small_or_single_arm'
    assert s.propensity_diagnostics([], [])['fit_denominator'] == 0


def test_stabilized_ess_balanced_identical_covariates():
    x = [{'center': 'C01', 'STATcat': '1', 'cardsurgtype': '1', 'ProcPrimary': '10', 'constant': 1.0}] * 100
    result = s.propensity_diagnostics(x, [0, 1] * 50)
    assert result['ess']['total'] == pytest.approx(100)
    assert result['ess']['A'] == pytest.approx(50)
    assert result['ess']['B'] == pytest.approx(50)
    assert result['smd_max_after'] == pytest.approx(0)
