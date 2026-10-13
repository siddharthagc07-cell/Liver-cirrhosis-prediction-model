import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder

BASE = '/drive/siddharth/Liver_Cirrhosis_ML'
ft = pd.read_csv(f'{BASE}/ml/feature_table.tsv', sep='\t', skiprows=1, index_col=0).T
meta = pd.read_csv(f'{BASE}/docs/metadata.tsv', sep='\t', index_col=0)
common = ft.index.intersection(meta.index)
X = ft.loc[common]
X = X.div(X.sum(axis=1), axis=0)
m = meta.loc[common]
y = LabelEncoder().fit_transform(m['phase1_label'])
ds = m['dataset']

for d in ds.unique():
    mask = (ds == d).to_numpy()
    Xd = X[mask].to_numpy(dtype=np.float32)
    yd = y[mask]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                class_weight='balanced', random_state=42, n_jobs=-1)
    r = cross_validate(rf, Xd, yd, cv=cv,
                       scoring={'auc': 'roc_auc', 'bal': 'balanced_accuracy'})
    print('\n====', d, '====')
    print('counts (Cirrhosis, Healthy):', np.bincount(yd, minlength=2).tolist())
    print('within-cohort AUROC: %.3f +/- %.3f' % (r['test_auc'].mean(), r['test_auc'].std()))
    print('within-cohort balanced accuracy: %.3f +/- %.3f' % (r['test_bal'].mean(), r['test_bal'].std()))

