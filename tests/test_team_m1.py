import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from pf_nec import contract as c, windows
from explore.team import sequence as old, sequence_m1 as s
from test_team import rows_fixture, extract, m1_config as config, m1_configured as configured, build_m1_apply as build_apply


@pytest.mark.parametrize("k", (10, 29, 50))
def test_parameter_formulas(k):
    assert s.parameter_count(s.GRUD(k)) == 226*k+3329
    assert s.parameter_count(s.CausalMSCNN(k)) == 64*k+11209
    assert s.parameter_count(s.GRUD(k)) <= 30000
    assert s.parameter_count(s.CausalMSCNN(k)) <= 30000


@pytest.mark.parametrize("learner", ("GRUD-WINDOW", "CAUSAL-MS-CNN"))
def test_architecture_bias_and_finite(learner):
    model = s.make_model(configured(learner), .125)
    x = torch.randn(3, 7, 101)
    x[..., 50:100] = torch.randint(0, 2, (3, 7, 50)).float()
    x[..., -1] = 1
    assert model(x).shape == (3,)
    assert torch.isfinite(model(x)).all()
    assert model.head.bias.item() == pytest.approx(np.log(.125/.875))
    assert not any(isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)) for m in model.modules())
    if learner == "GRUD-WINDOW":
        assert model.drop.p == .15
        assert all(torch.equal(g.bias, torch.zeros(32)) for g in model.gates.values())
        assert torch.equal(model.a_x, torch.full((50,), .1))
        assert torch.allclose(model.hidden_decay.weight, torch.full((32, 50), .1/50))
        assert all(layer.bias is None for layer in model.recurrent.values())
    else:
        assert model.receptive_field == 19 and len(model.blocks) == 3
        assert [[conv.kernel_size[0] for conv in block.branches] for block in model.blocks] == [[1,3,7]]*3
        assert model.input.in_features == 101
        assert all(block.norm.normalized_shape == (32,) for block in model.blocks)


@pytest.mark.parametrize("learner", ("GRUD-WINDOW", "CAUSAL-MS-CNN"))
@pytest.mark.parametrize("training", (False, True))
def test_future_causality_including_decay_and_gradients(learner, training):
    model = s.make_model(configured(learner, 10), .25).train(training)
    x = torch.randn(4, 7, 21)
    x[..., 10:20] = torch.randint(0, 2, (4,7,10)).float()
    x[..., -1] = 1
    changed = x.clone()
    changed[:, 4:, :10] += 300
    changed[:, 4:, 10:20] = 1 - changed[:, 4:, 10:20]
    changed[:, 4:, -1] = 0
    torch.manual_seed(73)
    before = model.sequence_logits(x)
    torch.manual_seed(73)
    after = model.sequence_logits(changed)
    assert torch.equal(before[:, :4], after[:, :4])
    if learner == "GRUD-WINDOW":
        _, d1 = model.sequence_logits(x, return_decay=True)
        _, d2 = model.sequence_logits(changed, return_decay=True)
        assert torch.equal(d1[:, :4], d2[:, :4])
    differentiable = x.clone().requires_grad_()
    model.sequence_logits(differentiable)[:, 3].sum().backward()
    assert (differentiable.grad[:, 4:] == 0).all()


@pytest.mark.parametrize("learner", ("GRUD-WINDOW", "CAUSAL-MS-CNN"))
def test_cross_stay_and_call_isolation(learner):
    model = s.make_model(configured(learner, 10), .25).eval()
    x = torch.randn(3,7,21)
    x[..., 10:20], x[..., -1] = 1, 1
    baseline = model(x[:1])
    changed = x.clone()
    changed[1:] += 100
    assert torch.allclose(model(changed)[:1], baseline, atol=1e-6)
    model(torch.randn(9,14,21))
    assert torch.equal(model(x[:1]), baseline)
    assert torch.allclose(model.sequence_logits(x[:, :4]), model.sequence_logits(x)[:, :4], atol=1e-6)


def test_grud_hand_calculated_delta_and_unavailable_row():
    model = s.GRUD(2, np.array([.4, -.5], np.float32)).eval()
    x = torch.zeros(1,5,5)
    x[..., -1] = 1
    x[0,:,2:4] = torch.tensor([[0,1],[1,1],[1,0],[0,0],[0,0]])
    x[0,3,-1] = 0
    logits, delta = model.sequence_logits(x, return_decay=True)
    expected = torch.tensor([[[0,0],[1,1],[2,2],[3,1],[4,2]]], dtype=torch.float32)
    assert torch.equal(delta, expected)
    value_gamma = torch.exp(-torch.relu(model.a_x*delta+model.b_x))
    hidden_gamma = torch.exp(-torch.relu(model.hidden_decay(delta)))
    assert ((value_gamma>=0)&(value_gamma<=1)).all()
    assert ((hidden_gamma>=0)&(hidden_gamma<=1)).all()
    assert torch.isfinite(logits).all()
    padding = torch.zeros(1,3,5)
    padding[..., 2:4] = 1
    assert torch.equal(model.sequence_logits(padding), model.head.bias.expand(1,3))
    legal = padding.clone(); legal[..., -1] = 1
    assert not torch.equal(model.sequence_logits(legal), model.sequence_logits(padding))


def test_grud_missing_bits_ignore_imputed_values_and_reset_window():
    model = s.GRUD(2, np.array([.3,.7], np.float32)).eval()
    x = torch.zeros(1,4,5); x[..., 2:4] = 1; x[..., -1] = 1
    changed = x.clone(); changed[..., :2] = 200
    assert torch.equal(model(x), model(changed))
    model(torch.tensor([[[900.,800.,0.,0.,1.]]]))
    assert torch.equal(model(x), model(changed))


def test_grud_reset_before_equation():
    model = s.GRUD(2).eval()
    x = torch.tensor([[[1.,2.,0.,0.,1.],[3.,4.,0.,0.,1.]]])
    h = torch.zeros(1,32)
    for slot in range(2):
        delta = torch.full((1,2), float(slot))
        decayed = torch.exp(-torch.relu(model.hidden_decay(delta)))*h
        u = torch.cat((x[:,slot,:2], torch.ones(1,2), torch.ones(1,1)), dim=-1)
        z = torch.sigmoid(model.gates['z'](u)+model.recurrent['z'](decayed))
        r = torch.sigmoid(model.gates['r'](u)+model.recurrent['r'](decayed))
        n = torch.tanh(model.gates['n'](u)+model.recurrent['n'](r*decayed))
        h = (1-z)*decayed+z*n
    assert torch.equal(model(x), model.head(h).squeeze(-1))


@pytest.mark.parametrize("learner", s.LEARNERS)
def test_future_feature_rows_do_not_change_past_package(learner, tmp_path):
    rows, _, _ = rows_fixture()
    cfg = config(learner)
    first, _ = s._pack_rows(rows, cfg)
    changed = copy.deepcopy(rows)
    future = changed.keys[c.DATE].eq(103).to_numpy()
    changed.values.iloc[future] += 200
    second, _ = s._pack_rows(changed, cfg)
    past = rows.keys[c.DATE].le(101).to_numpy()
    assert np.array_equal(first['X'][past], second['X'][past], equal_nan=True)
    assert np.array_equal(first['availability'][past], second['availability'][past])
    other = copy.deepcopy(rows)
    other.values.iloc[4:] += 500
    third, _ = s._pack_rows(other,cfg)
    assert np.array_equal(first['X'][:4], third['X'][:4], equal_nan=True)


@pytest.mark.parametrize("learner", s.LEARNERS)
def test_apply_forbidden_fields_and_train_stay_overlap(learner, tmp_path):
    rows,y,post=rows_fixture()
    info=build_apply(tmp_path/'apply.zip',rows,config(learner))
    folder=extract(info['path'],tmp_path/'apply')
    cfg,arrays=s.b.load_package(folder)
    assert set(arrays)=={'X','row_key'}
    for field in ('y','y30','outcome','NEC_date','discharge','lead','final_los','case','pod'):
        with pytest.raises(s.SequenceError):
            s._validate_arrays(cfg, dict(arrays, **{field:np.zeros(24)}))
    for field in ('label','outcome','nec','lead','discharge','stay_length'):
        bad=copy.deepcopy(cfg);bad['features'][0]=field
        with pytest.raises(s.SequenceError):s._validate_arrays(bad,arrays)
    with pytest.raises(s.SequenceError,match='overlap'):
        s.build_fit_package(tmp_path/'bad.zip',rows,y,config(learner,'inner'),
            validation_rows=rows,validation_labels=y,validation_score_mask=post)


def test_tabm_member_loss_probability_mean_and_version_guard(monkeypatch):
    logits=torch.tensor([[10.,-1.]*8,[-4.,3.]*8],requires_grad=True)
    target=torch.tensor([0.,1.])
    loss=s.member_loss(logits,target)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,target[:,None].expand_as(logits))
    assert torch.equal(loss,expected)
    assert not torch.isclose(loss,torch.nn.functional.binary_cross_entropy_with_logits(logits.mean(1),target))
    class Members(nn.Module):
        def forward(self,x):return logits[:len(x)]
    pp=s.Preprocessor.fit(np.array([[0.]*10,[1.]*10],np.float32))
    raw=np.zeros((2,34),np.float32);raw[:,30:33]=1;raw[:,-1]=3
    actual=s._predict(Members(),raw,pp,torch.device('cpu'))
    assert np.allclose(actual,logits.sigmoid().mean(1).detach().numpy())
    monkeypatch.setattr(s.importlib.metadata,'version',lambda name:'0.0.4')
    with pytest.raises(s.SequenceError,match='0.0.3'):s.TabMR1(34)
    with pytest.raises(s.SequenceError,match='800'):s.TabMR1(801)


@pytest.mark.parametrize("learner", s.LEARNERS)
def test_portable_cpu_micro_fit_apply_determinism_and_outer_independence(learner,tmp_path,monkeypatch):
    if learner == "TABM-R1":
        try: importlib.metadata.version("tabm")
        except importlib.metadata.PackageNotFoundError:
            if os.environ.get("PF_M1_REQUIRE_TABM") == "1":
                pytest.fail("TabM micro-smoke requires tabm==0.0.3")
            pytest.skip("TabM exact-library micro-smoke pending dependency")
    train,y,_=rows_fixture(); test,_,_=rows_fixture(offset=50)
    fit=s.build_fit_package(tmp_path/'fit.zip',train,y,config(learner,'refit'))
    fitdir=extract(fit['path'],tmp_path/'fit')
    fitted,_=s.b.load_package(fitdir)
    apply=s.build_apply_package(tmp_path/'apply.zip',test,config(learner),
        preprocessing=fitted['preprocessing'],fit_pool_sha256=fitted['fit_pool_sha256'])
    applydir=extract(apply['path'],tmp_path/'apply')
    result=subprocess.run([sys.executable,'-B',str(fitdir/'sequence.py'),'fit','--package',str(fitdir),
        '--output',str(tmp_path/'models'),'--device','cpu','--synthetic-smoke'],cwd=tmp_path,
        capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    hashes={p.name:s.file_sha256(p) for p in (tmp_path/'models').glob('init*.pt')}
    result=subprocess.run([sys.executable,'-B',str(applydir/'sequence.py'),'apply','--package',str(applydir),
        '--models',str(tmp_path/'models'),'--output',str(tmp_path/'probability.npz'),'--device','cpu'],
        cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    with np.load(tmp_path/'probability.npz',allow_pickle=False) as p:
        assert len(p['probability'])==24 and np.isfinite(p['probability']).all()
        assert np.array_equal(p['row_key'],test.keys[c.SOURCE_ROW])
    assert hashes=={p.name:s.file_sha256(p) for p in (tmp_path/'models').glob('init*.pt')}
    def forbidden_fit(*args,**kwargs):raise AssertionError('GPU runner/apply must not refit preprocessing')
    monkeypatch.setattr(s.Preprocessor,'fit',forbidden_fit)
    second=tmp_path/'models2';s.fit_package(fitdir,second,device='cpu',synthetic_smoke=True)
    for init in range(3):
        a=torch.load(tmp_path/'models'/f'init{init}.pt',weights_only=True)
        b=torch.load(second/f'init{init}.pt',weights_only=True)
        assert a['preprocessing']==b['preprocessing']
        assert all(torch.equal(v,b['state_dict'][key]) for key,v in a['state_dict'].items())
    outer=copy.deepcopy(test);outer.values.iloc[outer.keys[c.DATE].eq(103).to_numpy()] += 200
    info=s.build_apply_package(tmp_path/'altered.zip',outer,config(learner),
        preprocessing=fitted['preprocessing'],fit_pool_sha256=fitted['fit_pool_sha256'])
    changed_probability=s.apply_package(extract(info['path'],tmp_path/'altered'),tmp_path/'models',tmp_path/'altered.npz',device='cpu')
    with np.load(tmp_path/'probability.npz',allow_pickle=False) as data:
        past=test.keys[c.DATE].le(101).to_numpy()
        assert np.array_equal(data['probability'][past],changed_probability[past])
    assert hashes=={p.name:s.file_sha256(p) for p in (tmp_path/'models').glob('init*.pt')}


def test_real_cpu_fallback_and_single_class_fail_closed(tmp_path):
    cfg=config();cfg['synthetic']=False
    with pytest.raises(s.SequenceError,match='CPU'):s._device(cfg,'cpu')
    with pytest.raises(s.SequenceError,match='Single-class'):s.make_model(configured(),0.)
    rows,y,post=rows_fixture(); val,vy,vp=rows_fixture(offset=50)
    with pytest.raises(s.SequenceError,match='Single-class training'):
        s.build_fit_package(tmp_path/'bad.zip',rows,y*0,config(stage='refit'))
    with pytest.raises(s.SequenceError,match='Single-class validation'):
        s.build_fit_package(tmp_path/'bad2.zip',rows,y,config(stage='inner'),
            validation_rows=val,validation_labels=vy*0,validation_score_mask=vp)


@pytest.mark.parametrize('learner',('GRUD-WINDOW','CAUSAL-MS-CNN'))
def test_maximum_shape_backward_optimizer_and_easy_loss_decreases(learner):
    cfg=configured(learner)
    cfg['stage']='refit';cfg['chosen_epochs']=20
    # This is a CPU shape probe, not GPU throughput or clinical performance.
    values=np.zeros((24,50),np.float32)
    values[:12]=-3;values[12:]=3
    pp=s.Preprocessor.fit(values)
    raw=np.zeros((24,14,101),np.float32)
    raw[:,:,:50]=values[:,None,:]
    raw[:,:,-1]=1
    arrays=dict(train_y=np.array([0]*12+[1]*12),train_X=pp.r4(raw))
    model,receipt=s._fit_one(cfg,arrays,pp,0,torch.device('cpu'))
    losses=[r['loss'] for r in receipt['history']]
    assert losses[-1]<losses[0]
    assert np.isfinite(losses).all()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(p.device.type=='cpu' for p in model.parameters())
    assert np.isfinite(s._predict(model,pp.r4(raw),pp,torch.device('cpu'))).all()


@pytest.mark.parametrize('learner',('GRUD-WINDOW','CAUSAL-MS-CNN'))
def test_inner_patience_tolerance_best_state_restore_and_single_class_batch(learner,tmp_path,monkeypatch):
    rows,y,_=rows_fixture();valid,vy,post=rows_fixture(offset=50)
    package=s.build_fit_package(tmp_path/'inner.zip',rows,y,config(learner,'inner'),
        validation_rows=valid,validation_labels=vy,validation_score_mask=post)
    cfg,arrays=s.b.load_package(extract(package['path'],tmp_path/'inner'))
    pp=s.Preprocessor.from_state(cfg['preprocessing'])
    states=[]
    original=s._predict
    def tracked(model,*args):
        states.append({k:v.clone() for k,v in model.state_dict().items()})
        return original(model,*args)
    monkeypatch.setattr(s,'_predict',tracked)
    trajectory=iter([.6,.60005,.61]+[.61]*8)
    monkeypatch.setattr(s.b,'_binary_metric',lambda *args:{'auc':next(trajectory)})
    model,receipt=s._fit_one(cfg,arrays,pp,0,torch.device('cpu'))
    assert receipt['best_epoch']==3 and receipt['epochs_ran']==11
    assert all(torch.equal(v,states[2][key]) for key,v in model.state_dict().items())
    # A minibatch without positives remains in ordinary unit-weight BCE.
    assert torch.isfinite(s.member_loss(torch.tensor([.1,.2],requires_grad=True),torch.zeros(2)))


def test_train_only_mu_and_unique_source_statistics():
    values=np.array([[0.,np.nan],[2.,np.nan],[100.,np.nan]],np.float32)
    pp=s.Preprocessor.fit(values)
    transformed=pp.transform(values)
    assert pp.mu[0]==pytest.approx(transformed[:,0].mean()) and pp.mu[1]==0
    overlapped=np.repeat(values,[1,2,8],axis=0)
    repeated=s.Preprocessor.fit(overlapped)
    assert not np.allclose(pp.mean,repeated.mean)
    baseline=copy.deepcopy(pp.state())
    pp.r4(np.full((5,7,5),1e9,np.float32))
    assert pp.state()==baseline


@pytest.mark.parametrize('field',('availability','stay','actual_date','train_y','validation_score_mask','final_los'))
def test_minimal_apply_metadata_white_list(field,tmp_path):
    rows,_,_=rows_fixture()
    info=build_apply(tmp_path/'minimal.zip',rows,config())
    cfg,arrays=s.b.load_package(extract(info['path'],tmp_path/'minimal'))
    with pytest.raises(s.SequenceError,match='fields'):
        s._validate_arrays(cfg,dict(arrays,**{field:np.zeros(len(rows.keys))}))
    assert not {'source_values','phase','case','stay','actual_date'} & set(arrays)


def test_join_requires_exact_int64_complete_keys_and_rejects_corruption(tmp_path):
    targets=pd.DataFrame(dict(repeat=[4,4],row_key=[1234567890123456,1234567890123457],
                              stay=['s1','s2'],y=[0,1]))
    path=tmp_path/'predictions.npz'
    np.savez_compressed(path,repeat=np.array([4,4],np.int64),row_key=targets.row_key.to_numpy(np.int64),
                        probability=np.array([.2,.8],np.float64))
    joined=s.join_predictions(path,targets)
    assert joined.drop(columns='probability').equals(targets)
    for key in (np.array([1,2],np.int64),targets.row_key.to_numpy(np.float64)):
        np.savez_compressed(path,repeat=np.array([4,4],np.int64),row_key=key,probability=np.array([.2,.8],np.float64))
        with pytest.raises(s.SequenceError):s.join_predictions(path,targets)


def test_packager_stats_fitted_only_to_train_even_when_validation_changes(tmp_path):
    train,y,_=rows_fixture();valid,vy,post=rows_fixture(offset=50)
    first=s.build_fit_package(tmp_path/'one.zip',train,y,config(stage='inner'),
        validation_rows=valid,validation_labels=vy,validation_score_mask=post)
    altered=copy.deepcopy(valid);altered.values.iloc[:]+=200
    second=s.build_fit_package(tmp_path/'two.zip',train,y,config(stage='inner'),
        validation_rows=altered,validation_labels=1-vy,validation_score_mask=post)
    cfg1,a1=s.b.load_package(extract(first['path'],tmp_path/'one'))
    cfg2,a2=s.b.load_package(extract(second['path'],tmp_path/'two'))
    assert cfg1['preprocessing']==cfg2['preprocessing']
    assert cfg1['fit_pool_sha256']==cfg2['fit_pool_sha256']
    assert np.array_equal(a1['train_X'],a2['train_X'])
    assert not np.array_equal(a1['validation_X'],a2['validation_X'])
