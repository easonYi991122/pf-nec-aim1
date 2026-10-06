"""v2.6 raw as-of rebuild, copied from the frozen v2.5 rebuild/common code.

Only the new module writes through build_v26.py. It never imports frozen modules
with import-time filesystem side effects. Legacy interval/category semantics are
retained; optional line burden explicitly distinguishes EOD from any-time-in-day.
"""
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import re
import warnings

import numpy as np
import pandas as pd

from .. import config
RAW = config.DATA_ROOT / "Raw CSV Files"
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=120000)
def serial(v):
    if v in ('', '.', 'NA', 'NaN', 'nan', None):
        return float('nan')
    for fmt in ['%d-%b-%y', '%d%b%y:%H:%M:%S']:
        try:
            return (datetime.strptime(v, fmt) - datetime(1899, 12, 30)).total_seconds() / 86400
        except ValueError:
            pass
    return float('nan')


def hid(v):
    return re.sub(r'\.0+$', '', str(v).strip())


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else float('nan')
    except (ValueError, TypeError):
        return float('nan')


def code(v):
    return '' if str(v).strip() in ('', '.', 'nan', 'None') else hid(v)


def read_raw(raw_dir=RAW):
    out = {}
    for p in sorted(Path(raw_dir).glob('*.csv')):
        try:
            p.read_text(encoding='utf-8-sig')
            enc = 'utf-8-sig'
        except UnicodeDecodeError:
            enc = 'cp1252'
        d = pd.read_csv(p, encoding=enc, dtype=str, keep_default_na=False)
        d.insert(0, 'source_record_number', np.arange(2, len(d) + 2))
        d.insert(1, 'hid', d.hospitalizationidNEW.map(hid))
        for c in list(d):
            if 'SHIFT' in c:
                d[c + '__serial'] = d[c].map(serial)
        out[p.stem] = d
    return out


DRUGS=['dopa','dobut','epi','norepi','milrin','vasopress']
EVENTS={'Arrest':('cardarrestdttmSHIFT','arrest'),'CABSI':('CABSIDtSHIFT','clabsi'),'DSSI':('WoundInfDtSHIFT','dssi'),'Stroke':('StrokeDtTmSHIFT','stroke'),'UTI':('UTIDtSHIFT','uti')}
INTERVALS={'ArterialLine':('artlinestartdtSHIFT','artlineenddtSHIFT'),'IntracardLine':('IntracardLineStartDtSHIFT','IntracardLineEndDtSHIFT'),'MechVent':('ventstartSHIFT','ventendSHIFT'),'ECMO':('MechCircSuppInitDtTmSHIFT','MechCircSuppDiscDtTmSHIFT'),'Sternum':('SternumDtSHIFT','SternumClosedDtSHIFT')}
COMPS={'compreopbleed':('CompReopBleed','CompReopBleedDtTmSHIFT'),'lcospost':('LCOSpost','LCOSpostDtTmSHIFT'),'lcospostvis':('LCOSpostVIS','LCOSpostDtTmSHIFT'),'lcosposttriple':('LCOSpostTriple','LCOSpostDtTmSHIFT'),'lcospostavo':('LCOSpostAVO','LCOSpostDtTmSHIFT')}
THERAPY={'crrtarf':('CRRTarf','CRRTarfDtTmSHIFT',None),'crrtpd':('CRRTPD','CRRTPDstartDtSHIFT','CRRTPDendDtSHIFT'),'crrthemodial':('CRRTHemodial','CRRTHemodialStartDtSHIFT','CRRTHemodialEndDtSHIFT'),'crrtcvvh':('CRRTCVVH','CRRTCVVHstartDtSHIFT','CRRTCVVHendDtSHIFT')}

# Explicit field whitelist: a numeric 9 in a dose, age, duration or operation
# category must never be recoded. PGEsurg is a binary field with undocumented 9s.
MASTER_BINARY = ['PretermYN', 'RaceCaucasian', 'RaceBlack', 'RaceAsian',
                 'RaceNativeAm', 'RaceNativePI', 'RaceOther', 'Ethnicity',
                 'AntenatalDiag', 'ChromSyndSpecYN', 'ExtracardSpecYN', 'PGEsurg',
                 'OpExtubateYN', 'RScldPre', 'RSfeedPreop', 'RSVISpre']
CATH_BINARY = ['InterventionalYN', 'DiagnosticYN']
PULSE_BASES = {v[1] for v in EVENTS.values()} | {'any_catheterizations'}

# Fixed, overlapping text rules; no labels, frequencies or outcomes are used.
# Procedure memberships are proxies for the named anatomic target, not proof
# that the child has that anatomy or any particular physiology.
DX_RULES = {
    'single_ventricle': r'single ventricle|hypoplastic left heart|hlhs|tricuspid atresia|unbalanced.*av|univentricular',
    'arch_obstruction': r'coarctation|aortic arch hypoplasia|interrupted aortic arch|arch interruption|arch obstruction',
    'systemic_outflow_obstruction': r'hypoplastic left heart|hlhs|aortic (?:valve )?(?:atresia|stenosis)|coarctation|aortic arch hypoplasia|interrupted aortic arch|subaortic stenosis|supravalvar aortic stenosis',
    'pulmonary_outflow_obstruction': r'pulmonary atresia|pulmonary stenosis|pulmonic stenosis|tricuspid atresia|\btof\b|tetralogy',
    'tga_dorv': r'\btga\b|transposition|\bdorv\b|double outlet',
    'anomalous_pulmonary_venous_return': r'tapvc|papvc|tapvr|papvr|anomalous pulmonary ven',
    'shunt_septal_pda': r'\bvsd\b|\basd\b|avsd|\bavc\b|septal defect|\bpfo\b|patent ductus|\bpda\b|ap window|truncus arteriosus',
    'valve': r'valv|ebstein|mitral|tricuspid|aortic regurgitation',
    'heterotaxy': r'heterotax|isomerism|situs',
    'coronary': r'coronary|alcapa',
    'rhythm': r'arrhythm|heart block|tachycard|bradycard',
}
PROC_EXTRA = {
    'single_ventricle': r'norwood|hybrid.*stage (?:1|i\b)|damus|stansel|fontan|glenn|cavopulmonary',
    'arch_obstruction': r'aortic arch|arch repair|arch reconstruction',
    'systemic_outflow_obstruction': r'norwood|damus|stansel|lvoto|lvot obstruction',
    'pulmonary_outflow_obstruction': r'rvoto|rvot obstruction',
    'tga_dorv': r'arterial switch|\baso\b|nikaidoh|senning|mustard',
    'shunt_septal_pda': r'shunt.*systemic.*pulmonary|blalock|central shunt|atrial.*fenestration|septostomy|pa band|pulmonary artery band|rpa.*lpa bands',
    'rhythm': r'pacemaker|ablation',
}
DX_GROUPS = list(DX_RULES) + ['other']


def diagnosis_mapping(raw):
    """Ship every observed raw code with the literal rule that assigned it."""
    mapped = []
    for source, table, field, text_field in [
        ('Surgdiag', 'Surgdiag', 'surgdiag', 'surgdiagTxt'),
        ('FundDiagnosis', 'IndexSurgHosp', 'FundDiagnosis', 'FundDiagnosistxt'),
        ('ProcPrimary', 'IndexSurgHosp', 'ProcPrimary', 'ProcPrimaryTxt'),
    ]:
        pairs = raw[table][[field, text_field]].copy()
        pairs['code'] = pairs[field].map(code)
        for c, entries in pairs[pairs.code.ne('')].groupby('code', sort=True):
            labels = sorted({str(v).strip() for v in entries[text_field] if code(v)})
            label = ' / '.join(labels)
            rules = {g: p + ('|' + PROC_EXTRA[g] if source == 'ProcPrimary' and g in PROC_EXTRA else '')
                     for g, p in DX_RULES.items()}
            groups = [g for g, p in rules.items() if re.search(p, label, re.I)] or ['other']
            mapped.append(dict(source=source, clinical_code=c, label=label,
                               families=';'.join(groups), interpretation='anatomic proxy',
                               mapping_status='missing_label' if not label else ('fallback_other' if groups == ['other'] else 'text_rule'),
                               label_variants=len(labels),
                               matched_rules=';'.join(g + '=' + rules[g] for g in groups if g != 'other')))
    return pd.DataFrame(mapped)


def unique_records(records, id_field):
    """Deduplicate by stay-local raw ID, rejecting conflicting versions."""
    seen = {}
    for r in records:
        payload = tuple(sorted((k, str(v)) for k, v in r.items()
                               if k != 'source_record_number' and not k.endswith('__serial')))
        key = code(r.get(id_field, '')) or payload
        if key in seen and seen[key][0] != payload:
            raise ValueError(f'Conflicting raw records for {id_field}; adjudication required')
        seen[key] = (payload, r)
    return [v[1] for v in seen.values()]


def is_index_operation(record, master):
    """The raw operative ID is authoritative; its date must agree as well."""
    match = code(record.get('operativeidNEW')) == code(master.operativeidNEW)
    if match and record['cardsurgdtSHIFT__serial'] != master.cardsurgdtSHIFT__serial:
        raise ValueError('Index operative ID has a conflicting date')
    return match


def line_burden(dates, records, start, end, id_field):
    """EOD state, placement counts and inclusive calendar line-days since admission.

    A line removed on t is absent at EOD t but contributes a calendar line-day.
    Missing ends remain open; missing starts / end-before-start records are ignored.
    """
    result = np.zeros((len(dates), 4), dtype=np.float32)
    for r in unique_records(records, id_field):
        st, en = r.get(start + '__serial', np.nan), r.get(end + '__serial', np.nan)
        if not math.isfinite(st) or (math.isfinite(en) and en < st):
            continue
        st = math.floor(st)
        en = math.floor(en) if math.isfinite(en) else np.inf
        result[:, 0] += (dates >= st) & (dates < en)
        result[:, 1] += dates == st
        result[:, 2] += dates >= st
        result[:, 3] += np.maximum(0, np.minimum(dates, en) - max(st, dates[0]) + 1)
    return result


def event_recency(dates, event_dates):
    """One count per unique raw event (multiple events on one day count twice)."""
    t = np.sort(np.floor([v for v in event_dates if math.isfinite(v)]))
    t = t[t >= dates[0]]  # match admission-anchored legacy histories
    right = np.searchsorted(t, dates, side='right')
    count7 = right - np.searchsorted(t, dates - 6, side='left')
    age = np.full(len(dates), np.nan)
    seen = right > 0
    age[seen] = dates[seen] - t[right[seen] - 1]
    return age, count7, seen.astype(float)


class Rebuilder:
    def __init__(self, raw=None, cols=None, defs=None, addchecks=None):
        self.raw = read_raw() if raw is None else raw
        self.master = self.raw['IndexSurgHosp'].set_index('hid', drop=False)
        if not self.master.index.is_unique:
            raise ValueError('IndexSurgHosp must have one row per stay')
        self.by = {t: {k: g.to_dict('records') for k, g in d.groupby('hid', sort=False)}
                   for t, d in self.raw.items()}
        self.cols = cols
        if cols is None or defs is None or addchecks is None:
            raise ValueError("Supply locally regenerated code mappings; no implicit metadata lookup")
        self.ci = {c: i for i, c in enumerate(self.cols)}
        self.defs = defs
        self.addchecks = addchecks
        self.rules = {c: {'column': c, 'label': self.defs.get(c, {}).get('label', c),
                         'status': 'unresolved_not_populated', 'source': '', 'rule': '', 'availability': ''}
                      for c in self.cols}
        self.direct = {c.lower(): c for c in self.master.columns if not c.endswith('__serial')}
        self.dx_mapping = diagnosis_mapping(self.raw)
        self.dx_lookup = {(r.source, r.clinical_code): r.families.split(';')
                          for r in self.dx_mapping.itertuples()}

    def rule(self,c,source,description,status='dictionary_reconstruction',availability='截至当日结束的登记信息；仅日精度不保证日内先后'):
        self.rules[c].update(source=source,rule=description,status=status,availability=availability)

    def build(self,k):
        m=self.master.loc[k];s=m.cardsurgdtSHIFT__serial;a=m.hospadmitdtSHIFT__serial;z=m.hospdischdtSHIFT__serial
        days=np.arange(int(a-s),int(z-s)+1);dates=days+s;n=len(days);X=np.full((n,len(self.cols)),np.nan)
        def setc(c,v,src,desc,status='dictionary_reconstruction',availability='截至当日结束的登记信息；仅日精度不保证日内先后'):
            if c not in self.ci:return
            X[:,self.ci[c]]=v;self.rule(c,src,desc,status,availability)
        setc('hospitalizationidnew',float(k),'IndexSurgHosp.hospitalizationidNEW','住院键仅做整数格式对齐','direct_source','identifier_not_feature')
        for c,rc in [('cardsurgdtshift_index','cardsurgdtSHIFT'),('icupacuadmitdttmshift','icupacuadmitdttmSHIFT'),('hospadmitdtshift','hospadmitdtSHIFT'),('hospdischdtshift','hospdischdtSHIFT')]:
            setc(c,m[rc+'__serial'],'IndexSurgHosp.'+rc,'解析日期为1899-12-30起序列日；不修正疑似错年','direct_source','出院日期为结局信息；其余依实际记录时点')
        setc('day',days,'IndexSurgHosp','当前日期减索引手术日','direct_source','calendar')
        setc('actual_date',dates,'IndexSurgHosp','逐日展开完整登记住院区间','direct_source','calendar')
        # Source-compatible recodings; unknown is never coerced to a true negative.
        special={'gender':{'1':1,'2':0},'hospdischstat':{'1':1,'2':0},'cardsurgtype':{'1':1,'2':0},'pretermyn':{'0':0,'1':1}}
        for c in self.cols[25:58]+['hospdischstat']:
            if c not in self.direct:continue
            rc=self.direct[c];v=special[c].get(m[rc],np.nan) if c in special else num(m[rc])
            if rc in MASTER_BINARY and num(m[rc]) == 9:
                v = np.nan
            setc(c,v,'IndexSurgHosp.'+rc,'按raw原值；性别/离院/手术类型按已核对的clean重编码；未知类别不合并为0','direct_source','术前/术中/术后2小时/全住院摘要混合，见原字典；不是自动可用特征')
        for prefix,rc in [('funddiagnosis_','FundDiagnosis'),('procprimary_','ProcPrimary')]:
            for c in [x for x in self.cols if x.startswith(prefix)]:
                members=[str(x['code']) for x in self.defs[c].get('members',[])]
                val=float(m[rc] in members) if m[rc] not in ('','.') else np.nan
                setc(c,val,'IndexSurgHosp.'+rc,'采用旧并组映射并重新验证；raw代码原样另存','empirical_group_mapping','手术诊断/术式记录后')
        rr=self.by['PreopRiskFactor'].get(k,[]);risk={r['preopfactor'] for r in rr}
        for c,v in self.addchecks['risk_minimum_compatible_groups'].items():
            covers=v['minimal_compatible_code_sets'];val=float(bool(risk.intersection(covers[0]))) if rr and not risk<=set(['','.']) else np.nan
            setc(c,val,'PreopRiskFactor.preopfactor','旧样本中唯一最小OR相容组；非原生成脚本，原始代码完整保留','empirical_group_mapping','术前记录')
        di=self.by['Surgdiag'].get(k,[]);di=[r for r in di if r['operativeidNEW']==m.operativeidNEW];dc={r['surgdiag'] for r in di}
        for c in [x for x in self.cols if x.startswith('Surgdiag_')]:
            setc(c,float(c.split('_')[-1] in dc) if dc and not dc<=set(['','.']) else np.nan,'Surgdiag.surgdiag','索引手术诊断代码的精确多热编码；全缺失不写全零','dictionary_reconstruction','索引手术诊断记录后')
        setc('surgdiagprim',max([num(r['SurgDiagPrim']) for r in di],default=np.nan),'Surgdiag.SurgDiagPrim','索引手术条目中主要诊断标记最大值','dictionary_reconstruction','索引手术诊断记录后')
        necrows=self.by['NEC'].get(k,[]);necdates=[r['necbelldtSHIFT__serial'] for r in necrows if math.isfinite(r['necbelldtSHIFT__serial']) and a<=r['necbelldtSHIFT__serial']<=z]
        postoperative=sorted(t-s for t in necdates if t>s)
        setc('preopNEC',int(any(t<s for t in necdates)),'NEC.necbelldtSHIFT','只使用落在登记住院区间内、术前的NEC；异常日期另存','dictionary_reconstruction','label_review_not_feature')
        # Legacy selected event date, diff, and outcome_3d intentionally unfilled.
        # A descriptive date/state supplement is supplied without committing to a task label.
        supp={
            'hid':np.repeat(k,n),'day':days,'actual_date':dates,
            'registered_postop_NEC_on_day':np.isin(days,postoperative).astype(int),
            'registered_postop_NEC_already_occurred':np.array([int(any(t<=d for t in postoperative)) for d in days]),
            'first_registered_postop_NEC_day':np.repeat(postoperative[0] if postoperative else np.nan,n),
            'has_abnormal_NEC_date':np.repeat(int(any(not math.isfinite(r['necbelldtSHIFT__serial']) or not a<=r['necbelldtSHIFT__serial']<=z for r in necrows)),n),
        }
        for rc in MASTER_BINARY:
            if rc.lower() in self.ci and rc in m:
                supp['v26__' + rc.lower() + '_unknown'] = np.repeat(float(num(m[rc]) == 9), n)
        def rows(t):return self.by[t].get(k,[])
        def point(rr,dc):
            out=np.zeros(n)
            for r in rr:
                t=r.get(dc+'__serial',np.nan)
                if math.isfinite(t):out[np.floor(t)==dates]=1
            return out  # as-of: undated records are ignored instead of making every day unknown
        def interval(rr,sc,ec):
            yes=np.zeros(n,dtype=bool);uncertain=np.zeros(n,dtype=bool)
            for r in rr:
                st=r.get(sc+'__serial',np.nan);en=r.get(ec+'__serial',np.nan) if ec else np.nan
                if not math.isfinite(st) or (math.isfinite(en) and en<st):continue  # undated/invalid: ignored (as-of)
                elif not math.isfinite(en):yes|=(dates>=math.floor(st))  # no end recorded: in place from start (as-of)
                else:yes|=(dates>=math.floor(st))&(dates<=math.floor(en))
            res=yes.astype(float);res[uncertain&~yes]=np.nan
            return res
        def history(base, post_values=None):
            b=X[:,self.ci[base]]; finite=np.isfinite(b)
            inclusive = base in PULSE_BASES or post_values is not None
            for suffix,mask in [('_prior_index',days<0),('_post_index',days>=0 if inclusive else days>0)]:
                if base+suffix in self.ci:
                    event = post_values if suffix == '_post_index' and post_values is not None else b
                    y=np.maximum.accumulate(np.where(mask&finite,event==1,0)).astype(float)
                    uncertain=np.maximum.accumulate(mask&~finite);y[uncertain&(y==0)]=np.nan
                    setc(base+suffix,y,self.rules[base]['source'],'截至本日的术前/术后历史；脉冲含POD0，操作排除索引手术；状态列保留v2.5规则')
            for suffix,mask in [('_before_index_days_cumsum',days<0),('_since_index_days_cumsum',days>=0),('_sum_beforeindex',days<0)]:
                if base+suffix in self.ci:
                    y=np.cumsum(np.where(mask&finite,b,0)).astype(float);y[np.maximum.accumulate(mask&~finite)]=np.nan
                    setc(base+suffix,y,self.rules[base]['source'],'从登记入院日起累计本日及以前标记日数；完整住院累计，不假装与截断面板相同')
        for t,(dc,c) in EVENTS.items():setc(c,point(rows(t),dc),t+'.'+dc,'事件发生日期的脉冲；无该表记录表示未登记该事件，不证明未发生');history(c)
        cr=rows('Catheterizations')
        setc('any_catheterizations',point(cr,'cardcathdtSHIFT'),'Catheterizations','当天有心导管记录；无日期记录不回填');history('any_catheterizations')
        for c in self.cols[109:124]:
            rc=next((x for x in self.raw['Catheterizations'] if x.lower()==c),None)
            if rc:
                vals=np.zeros(n)
                unknown = np.zeros(n)
                for r in cr:
                    t=r['cardcathdtSHIFT__serial'];v=num(r[rc])
                    if math.isfinite(t):
                        ix=np.where(dates==math.floor(t))[0]
                        if rc in CATH_BINARY and v == 9:
                            unknown[ix] = 1
                            v = 0
                        if len(ix): vals[ix]=np.maximum(vals[ix],v) if math.isfinite(v) else np.nan
                setc(c,vals,'Catheterizations.'+rc,'当日已知阳性；0/1/9字段的未知另列，不作为阳性')
                if rc in CATH_BINARY:
                    supp['v26__' + c + '_unknown'] = unknown
                    supp['v26__' + c + '_history_unknown'] = np.maximum.accumulate(unknown)
        op=rows('AllOperations')
        for prefix,rc in [('AllOperations_cardsurgtype_','cardsurgtype'),('AllOperations_procloc_','ProcLoc')]:
            for c in [x for x in self.cols if x.startswith(prefix)]:setc(c,point([r for r in op if r[rc]==c.split('_')[-1]],'cardsurgdtSHIFT'),'AllOperations.'+rc,'同代码当日手术记录的多热指示；未冒充未经证明的并组')
        other_op = [r for r in op if not is_index_operation(r, m)]
        for c in ['any_cardsurgtype','any_AllOperations_binary']:
            setc(c,point(op,'cardsurgdtSHIFT'),'AllOperations.cardsurgdtSHIFT','当天有任意手术记录，含索引手术')
            history(c, post_values=point(other_op, 'cardsurgdtSHIFT'))
        for t,(sc,ec) in INTERVALS.items():
            rr=rows(t);active=interval(rr,sc,ec)
            supp[t+'__recorded_active_day']=active
            supp[t+'__records_starting_today']=np.array([sum(math.isfinite(r.get(sc+'__serial',np.nan)) and math.floor(r[sc+'__serial'])==d for r in rr) for d in dates])
            supp[t+'__source_records']=np.repeat(len(rr),n)
            supp[t+'__missing_start_records']=np.repeat(sum(not math.isfinite(r.get(sc+'__serial',np.nan)) for r in rr),n)
            supp[t+'__missing_end_records']=np.repeat(sum(not math.isfinite(r.get(ec+'__serial',np.nan)) for r in rr),n)
            if t in ['ArterialLine','IntracardLine']:
                prefixes=[('ArterialLine_','ArtLineSite')] if t=='ArterialLine' else [('IntracardLineSite_','IntraCardLineSite'),('IntracardLineType_','IntraCardLineType')]
                for prefix,rc in prefixes:
                    for c in [x for x in self.cols if re.fullmatch(re.escape(prefix)+r'\d+',x)]:
                        setc(c,interval([r for r in rr if r[rc]==c.split('_')[-1]],sc,ec),t+'.'+rc,'同代码起止日期均计入；缺少终点时从开始日起持续在用');history(c)
            elif t in ['MechVent','Sternum']:
                c='mechvent' if t=='MechVent' else 'opensternum';setc(c,active,t,'起止日期均计入；无终点视为从开始日起持续在用');history(c)
                inverse='offmechvent_since_index_days_cumsum' if t=='MechVent' else 'closedsternum_since_index_days_cumsum'
                y=np.cumsum(np.where(days>=0,1-active,0));setc(inverse,y,t,'完整住院术日起累计未记录活动的日数；未知会传播，非精确小时')
        for c,(rc,dc) in COMPS.items():
            vals = point([r for r in rows('Complications') if num(r[rc]) == 1], dc)
            unknown = point([r for r in rows('Complications') if num(r[rc]) == 9], dc)
            setc(c,vals,'Complications.'+rc+';'+dc,'事件日已知阳性=1；未知另列，不作事件')
            setc(c+'_history',np.maximum.accumulate(vals),'Complications.'+rc,'截至本日是否出现已知阳性；未知不能压过后来阳性')
            supp['v26__' + c + '_unknown'] = unknown
            supp['v26__' + c + '_history_unknown'] = np.maximum.accumulate(unknown)
        for c,(rc,sc,ec) in THERAPY.items():
            rr=[r for r in rows('Therapies') if r[rc]=='1'];vals=interval(rr,sc,ec) if ec else point(rr,sc)
            setc(c,vals,'Therapies.'+rc,'肯定条目日期/区间；无终点不虚构停药/停机日')
            setc(c+'_history',np.maximum.accumulate(vals),'Therapies.'+rc,'截至本日的已知阳性历史；未知单独记录')
            unknown = point([r for r in rows('Therapies') if num(r[rc]) == 9], sc)
            supp['v26__' + c + '_unknown'] = unknown
            supp['v26__' + c + '_history_unknown'] = np.maximum.accumulate(unknown)
        vr=rows('RiskSurgVIS');timed=[r for r in vr if math.isfinite(r['RSVISdttmSHIFT__serial'])]
        supp['VIS__recorded_samples_today']=np.array([sum(math.floor(r['RSVISdttmSHIFT__serial'])==d for r in timed) for d in dates])
        supp['VIS__has_sample_today']=(supp['VIS__recorded_samples_today']>0).astype(int)
        for drug in DRUGS:
            rc='RSVIS'+drug;vals=np.full(n,np.nan)
            for i,d in enumerate(dates):
                vv=[num(r[rc]) for r in timed if math.floor(r['RSVISdttmSHIFT__serial'])==d and math.isfinite(num(r[rc]))]
                if vv:vals[i]=sum(vv)/len(vv)
            setc('rsvis'+drug,vals,'RiskSurgVIS.'+rc,'当日已记录定点剂量算术平均；无采样不作0；不同于全天暴露量')
            supp['VIS__'+drug+'_last_observed']=np.full(n,np.nan)
            supp['VIS__'+drug+'_observation_age_days']=np.full(n,np.nan)
            for i,d in enumerate(dates):
                past=[r for r in timed if r['RSVISdttmSHIFT__serial']<d+1 and math.isfinite(num(r[rc]))]
                if past:
                    last=max(past,key=lambda r:r['RSVISdttmSHIFT__serial']);supp['VIS__'+drug+'_last_observed'][i]=num(last[rc]);supp['VIS__'+drug+'_observation_age_days'][i]=d+1-last['RSVISdttmSHIFT__serial']
        setc('is',X[:,self.ci['rsvisdopa']]+X[:,self.ci['rsvisdobut']]+100*X[:,self.ci['rsvisepi']],'RiskSurgVIS','三药日均值的IS公式；任一缺失则缺失')
        # Dictionary has conflicting vasopressin units: neither legacy nor standard VIS is silently invented.
        for t,dc,c in [('Complications','SepsisDtSHIFT','sepsis'),('Complications','SupWoundInfDtSHIFT','superficial_wound_infection')]:
            flag='CompSepsis' if c=='sepsis' else 'CompSupWoundInf'
            vv=point([r for r in rows(t) if r[flag]=='1'],dc);supp[c+'__recorded_on_day']=vv;supp[c+'__recorded_by_day']=np.maximum.accumulate(vv)
            unknown = point([r for r in rows(t) if num(r[flag]) == 9], dc)
            supp['v26__supp_' + c + '_unknown'] = unknown
            supp['v26__supp_' + c + '_history_unknown'] = np.maximum.accumulate(unknown)
        for t in ['AllOperations','Catheterizations','RiskSurgVIS','Complications','Therapies']:
            supp[t+'__source_records']=np.repeat(len(rows(t)),n)
        supp.update({'v26__' + c: v for c, v in self.optional_blocks(k, dates, m).items()})
        return days,X,supp

    def optional_blocks(self, k, dates, master):
        new = {}
        for source, prefix, values in [
            ('FundDiagnosis', 'fund', {code(master.FundDiagnosis)} - {''}),
            ('ProcPrimary', 'proc', {code(master.ProcPrimary)} - {''}),
            ('Surgdiag', 'surg', {code(r['surgdiag']) for r in self.by['Surgdiag'].get(k, [])
                                 if code(r['operativeidNEW']) == code(master.operativeidNEW)} - {''}),
        ]:
            counts = dict.fromkeys(DX_GROUPS, 0)
            for value in sorted(values):
                for family in self.dx_lookup[(source, value)]:
                    counts[family] += 1
            for family in DX_GROUPS:
                new[f'dxg_{prefix}_{family}'] = np.repeat(counts[family] if values else np.nan, len(dates))
            if prefix != 'proc':
                new[f'dxg_{prefix}_n_diagnoses'] = np.repeat(len(values) if values else np.nan, len(dates))
        for table, prefix, id_field in [('ArterialLine', 'arterial', 'artlineidNEW'),
                                        ('IntracardLine', 'intracardiac', 'intracardlineidNEW')]:
            start, end = INTERVALS[table]
            values = line_burden(dates, self.by[table].get(k, []), start, end, id_field)
            for j, name in enumerate(['in_place_eod', 'placed_today', 'ever_placed', 'line_days']):
                new[f'lnb_{prefix}_{name}'] = values[:, j]

        events = {}
        ids = {'Arrest': 'cardarrestidNEW', 'Stroke': 'strokeidNEW', 'CABSI': 'cabsiidNEW',
               'DSSI': 'woundinfidNEW', 'UTI': 'utiidNEW'}
        for table, (dc, name) in EVENTS.items():
            rr = unique_records(self.by[table].get(k, []), ids[table])
            events[name] = ([r[dc + '__serial'] for r in rr], None)
        cr = unique_records(self.by['Complications'].get(k, []), 'complicationidNEW')
        for name, flag, dc in [('sepsis', 'CompSepsis', 'SepsisDtSHIFT'),
                               ('lcos', 'LCOSpost', 'LCOSpostDtTmSHIFT'),
                               ('reop_bleed', 'CompReopBleed', 'CompReopBleedDtTmSHIFT'),
                               ('superficial_wound_infection', 'CompSupWoundInf', 'SupWoundInfDtSHIFT')]:
            events[name] = ([r[dc + '__serial'] for r in cr if num(r[flag]) == 1],
                            [r[dc + '__serial'] for r in cr if num(r[flag]) == 9])
        cats = unique_records(self.by['Catheterizations'].get(k, []), 'cathidNEW')
        # A recorded catheterization is an event even if its subtype is unknown.
        events['catheterization'] = ([r['cardcathdtSHIFT__serial'] for r in cats], None)
        ops = unique_records(self.by['AllOperations'].get(k, []), 'operativeidNEW')
        events['other_operations'] = ([r['cardsurgdtSHIFT__serial'] for r in ops
                                       if not is_index_operation(r, master)], None)
        inf = ['clabsi', 'dssi', 'uti', 'sepsis', 'superficial_wound_infection']
        # A sum of source-specific episodes: two different infections on a day count twice.
        events['infections'] = ([t for name in inf for t in events[name][0]],
                                [t for name in inf for t in (events[name][1] or [])])
        for name, (positive, unknown) in events.items():
            for suffix, values in zip(['days_since_recent', 'count_7d', 'ever'], event_recency(dates, positive)):
                new[f'evr_{name}_{suffix}'] = values
            if unknown is not None:
                for suffix, values in zip(['days_since_recent_unknown', 'count_7d_unknown', 'unknown'],
                                          event_recency(dates, unknown)):
                    new[f'evr_{name}_{suffix}'] = values
        return new


def rebuild_all(builder, progress=False):
    """Same all-day float32 reconstruction as v2.5; all output is caller-owned."""
    parts, supplements = [], []
    for i, k in enumerate(builder.master.index):
        _, values, supp = builder.build(k)
        parts.append(values.astype(np.float32))
        supplements.append(pd.DataFrame({c: np.asarray(v, dtype=np.float32)
                                         for c, v in supp.items() if c not in ('hid', 'day', 'actual_date')}))
        if progress and i % 2000 == 0:
            print('rebuilt', i, flush=True)
    matrix = pd.DataFrame(np.vstack(parts), columns=builder.cols)
    extra = pd.concat(supplements, ignore_index=True)
    extra.columns = [c.removeprefix('v26__') if c.startswith('v26__') else 'supp_' + c for c in extra]
    return pd.concat([matrix, extra], axis=1)
