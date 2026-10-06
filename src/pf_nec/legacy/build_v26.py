"""Frozen v2.6 population, label and feature assembly functions."""
import numpy as np
import pandas as pd


H = 'hospitalizationidnew'


EXCEL0 = pd.Timestamp('1899-12-30')


DUPLICATES = {
    'any_AllOperations_binary': 'any_cardsurgtype',
    'any_AllOperations_binary_prior_index': 'any_cardsurgtype_prior_index',
    'any_AllOperations_binary_sum_beforeindex': 'any_cardsurgtype_sum_beforeindex',
    'any_AllOperations_binary_post_index': 'any_cardsurgtype_post_index',
    'supp_MechVent__recorded_active_day': 'mechvent',
    'supp_Sternum__recorded_active_day': 'opensternum',
}


def raw_dt(s):
    s = pd.Series(s).astype(str)
    return pd.to_datetime(s, format='%d-%b-%y', errors='coerce').fillna(
        pd.to_datetime(s, format='%d%b%y:%H:%M:%S', errors='coerce'))


def group(r):
    if np.isnan(r.nec_pod):
        return "none"
    if r.nec_doa < 0:
        return "err"
    if r.nec_pod < 0:
        return "c"
    if r.nec_pod <= 3:
        return "b"
    if r.nec_pod <= 31:
        return "a"
    return "d"


def build_stays(idx, nec, pr, clean_ids):
    nec["d"] = raw_dt(nec.necbelldtSHIFT)
    first = nec.groupby("hospitalizationidNEW").d.min()
    icu = raw_dt(idx.icupacuadmitdttmSHIFT)
    st = pd.DataFrame({H: idx.hospitalizationidNEW.astype(float), "site": idx.siteidNEW,
                       "surg": (raw_dt(idx.cardsurgdtSHIFT) - EXCEL0).dt.days,
                       "adm": (raw_dt(idx.hospadmitdtSHIFT) - EXCEL0).dt.days,
                       "dis": (raw_dt(idx.hospdischdtSHIFT) - EXCEL0).dt.days,
                       "died": (idx.Hospdischstat == 2).astype(int),
                       "icu_serial": (icu - EXCEL0).dt.total_seconds() / 86400})
    st["nec"] = (idx.hospitalizationidNEW.map(first) - EXCEL0).dt.days.values
    st["nec_pod"] = st.nec - st.surg
    st["nec_doa"] = st.nec - st.adm
    st["preop_nec_code"] = st[H].isin(pr.loc[pr.preopfactor.isin([320, 330]), "hospitalizationidNEW"].astype(float)).astype(int)

    st["group"] = st.apply(group, axis=1)
    st["in_clean"] = st[H].isin(clean_ids).astype(int)
    return st


def assemble_rows(R, st, populated):
    rows = R.merge(st, on=H, how="inner")
    rows = rows[(rows.actual_date < rows.dis) & ~(rows.actual_date >= rows.nec)]   # still at risk after the day
    formal_ok = rows.in_clean.eq(1) & rows.group.isin(["a", "b", "none"])
    rows["set"] = np.select(
        [formal_ok & (rows.day <= 31),
         rows.in_clean.eq(1) & rows.group.eq("c") & (rows.actual_date < rows.surg),
         rows.in_clean.eq(1) & rows.group.eq("d") & (rows.day <= 31),
         rows.in_clean.eq(1) & rows.group.isin(["d", "none"]) & (rows.day > 31)],
        ["formal", "c_aux", "d_window", "late_aux"], default="drop")
    rows = rows[rows.set != "drop"].copy()

    # ------------------------------------------------------------------ labels and time
    rows["y3"] = (rows.nec - rows.actual_date).between(1, 3).astype(int)
    rows["lead"] = np.where(rows.y3 == 1, rows.nec - rows.actual_date, np.nan)
    rows["y30"] = (rows.nec - rows.actual_date).between(1, 30).astype(int)   # DEC-005: Task B daily / landmark label
    rows["doa"] = rows.actual_date - rows.adm
    rows["post_surg"] = (rows.actual_date >= rows.surg).astype(int)
    rows["pod_raw"] = rows.actual_date - rows.surg

    # ------------------------------------------------------------------ features
    EXCLUDE = {H, "cardsurgdtshift_index", "icupacuadmitdttmshift", "hospadmitdtshift", "hospdischdtshift", "hospdischstat",
               "necbelldtshift", "actual_date", "outcome_3d", "diff", "preopNEC", "rsfeedpostop", "cardsurgdtshift", "day"}
    base = [c for c in populated if c not in EXCLUDE and c in rows.columns]
    SUPP_OK = ("_recorded_active_day", "_records_starting_today", "VIS__recorded_samples_today", "VIS__has_sample_today",
               "_last_observed", "_observation_age_days", "__recorded_on_day", "__recorded_by_day")
    supp = [c for c in rows.columns if c.startswith("supp_") and c.endswith(SUPP_OK) and "NEC" not in c]
    rows = rows.sort_values([H, "actual_date"]).reset_index(drop=True)
    g = rows.groupby(H, sort=False)
    new = {}
    for hcol in [c for c in base if c.endswith("_history")]:
        on = rows[hcol].eq(1)
        first_on = rows.actual_date.where(on).groupby(rows[H]).transform("min")
        new["z_days_since_" + hcol.replace("_history", "")] = np.where(on, rows.actual_date - first_on, np.nan)
    new["z_dol"] = rows.calcsurgage + rows.pod_raw          # age in days on this day (known from birth)
    new["z_doa"] = rows.doa
    new["z_post_surg"] = rows.post_surg
    new["z_pod"] = np.where(rows.post_surg == 1, rows.pod_raw, np.nan)
    for c in ["supp_MechVent__recorded_active_day", "supp_ArterialLine__recorded_active_day", "supp_Sternum__recorded_active_day"]:
        new["z_d1_" + c] = rows[c] - g[c].shift(1)
    rows = pd.concat([rows, pd.DataFrame(new, index=rows.index)], axis=1)

    # ------------------------------------------------------------------ availability masking
    SURGERY_TIME = [c for c in base if c in (
        "calcsurgage", "surgwtkg", "pgesurg", "cardsurgtype", "xclamptime", "opextubateyn", "statcat", "statscore", "rscldpre",
        "rsfeedpreop", "rsvispre", "rsdopa", "rsdobut", "rsepi", "rsnorepi", "rsmilrin", "rsvasopress", "surgdiagprim")
        or c.startswith(("procprimary_", "Surgdiag_", "PreopRiskFactor_"))]
    # v2.5 (DEC-005): FundDiagnosis is treated as known at admission (echo/SpO2-based anatomic diagnosis; 67% antenatal),
    # so it is NOT masked preoperatively; the strict variant masks it at run time for the sensitivity analysis.
    TWO_HOUR = [c for c in base if c in ("rsdopa", "rsdobut", "rsepi", "rsnorepi", "rsmilrin", "rsvasopress")]
    rows.loc[rows.post_surg == 0, SURGERY_TIME] = np.nan
    # v2.4: on the surgery day itself, surgery-time fields are known only if the patient reached the ICU by the end of the
    # day (180 stays arrive after midnight); an implausible ICU time falls back to the surgery date.
    icu_plaus = (rows.icu_serial - rows.surg).between(0, 2)
    # v2.5: an implausible ICU time no longer falls back to the surgery date; surgery-time fields open only from POD1
    done_by_eod = np.where(icu_plaus, rows.actual_date + 1 >= rows.icu_serial, rows.pod_raw >= 1)
    rows.loc[(rows.pod_raw == 0) & ~done_by_eod, SURGERY_TIME] = np.nan
    # v2.3: the 2-hour postoperative summary is observable only after ICU arrival + 2 h (full timestamp, any date);
    # a missing or implausible ICU time (not within 0-2 days after the surgery date) keeps it unknown on every row.
    icu_ok = (rows.icu_serial - rows.surg).between(0, 2)
    avail = icu_ok & (rows.actual_date + 1 >= rows.icu_serial + 2 / 24)
    rows.loc[~avail, TWO_HOUR] = np.nan
    unknown = [c for c in rows if c.endswith('_unknown') and not c.startswith('evr_')]
    blocks = {prefix: [c for c in rows if c.startswith(prefix + '_')] for prefix in ('dxg', 'lnb', 'evr')}
    unknown_masked = [c + '_unknown' for c in SURGERY_TIME if c + '_unknown' in unknown]
    dx_masked = [c for c in blocks['dxg'] if c.startswith(('dxg_surg_', 'dxg_proc_'))]
    extra_masked = unknown_masked + dx_masked
    rows.loc[rows.post_surg == 0, extra_masked] = np.nan
    rows.loc[(rows.pod_raw == 0) & ~done_by_eod, extra_masked] = np.nan
    base_features = base + supp + list(new) + unknown
    for dropped, kept in DUPLICATES.items():
        if dropped in base_features:
            if not rows[dropped].equals(rows[kept]):
                raise AssertionError(f'Semantic duplicate no longer identical: {dropped}')
            base_features.remove(dropped)
    feats = base_features + [c for block in blocks.values() for c in block]
    # stay-level fundamental diagnosis for the lenient sensitivity analysis, taken from the master table (every stay, not
    # affected by NEC truncation); stored as meta, never in `feats`
    FD = [c for c in base if c.startswith("funddiagnosis_")]
    fd_stay = R.groupby(H)[FD].first().add_prefix("lenient_")

    meta = [H, "site", "surg", "adm", "dis", "died", "nec", "nec_pod", "group", "preop_nec_code", "set", "actual_date",
            "pod_raw", "doa", "post_surg", "y3", "lead", "y30"]
    out = rows[meta + feats].merge(fd_stay, left_on=H, right_index=True, how="left")
    out[feats] = out[feats].astype("float32")
    spec = {"version": "v2.6", "features": base_features, "base": base_features,
            "blocks": blocks, "all_features": feats,
            "surgery_time_masked_preop": SURGERY_TIME + extra_masked,
            "two_hour_masked": TWO_HOUR,
            "removed_duplicates": DUPLICATES,
            "default_blocks": [],
            "feature_selection": "features is base-only; append blocks.dxg/lnb/evr independently"}
    return out, spec
