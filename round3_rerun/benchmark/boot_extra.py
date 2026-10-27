import pickle, json, numpy as np, pandas as pd
st = pickle.load(open('oof_out/state.pkl','rb'))
df = pd.read_csv('/home/claude/data/4efe2522-arp_15parts.csv.gz', usecols=['label','partition'])
pos = df.groupby('partition').cumcount().values; tgt = np.where(pos>=19)[0]
y = (df.label.values[tgt]=='MITM-ArpSpoofing').astype(int); part = df.partition.values[tgt]
P = {q: np.where(part==q)[0] for q in np.unique(part)}
def counts(pred):  # per-partition tp, fp, fn
    return np.array([[((pred[i]==1)&(y[i]==1)).sum(), ((pred[i]==1)&(y[i]==0)).sum(), ((pred[i]==0)&(y[i]==1)).sum()] for i in P.values()])
cb = counts((st['oof']['XGB, T = 1']>=.5).astype(int))
f1 = lambda c: 2*c[0]/(2*c[0]+c[1]+c[2])
res = json.load(open('oof_out/round3_rerun_results.json'))
rng = np.random.default_rng(42); draws = rng.integers(0, 15, size=(2000, 15))
for c in st['oof']:
    if c == 'XGB, T = 1' or c in res['cluster_bootstrap_xgb_minus']: continue
    cc = counts((st['oof'][c]>=.5).astype(int))
    d = np.array([f1(cb[s].sum(0)) - f1(cc[s].sum(0)) for s in draws])
    fav = int(sum(f1(cb[i]) > f1(cc[i]) for i in range(15)))
    res['cluster_bootstrap_xgb_minus'][c] = dict(dF1=f1(cb.sum(0))-f1(cc.sum(0)), ci_low=float(np.percentile(d,2.5)), ci_high=float(np.percentile(d,97.5)), ci90_low=float(np.percentile(d,5)), ci90_high=float(np.percentile(d,95)), partitions_favouring_xgb=fav)
for k,v in res['mcnemar_vs_xgb_t1'].items(): v['p_bonf_m19'] = min(1.0, v['p']*19)
json.dump(res, open('oof_out/round3_rerun_results.json','w'), indent=1, default=float)
print(pd.DataFrame(res['cluster_bootstrap_xgb_minus']).T.round(4).sort_values('dF1'))
