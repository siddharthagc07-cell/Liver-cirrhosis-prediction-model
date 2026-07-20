import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("  PHASE 2 LODO ONLY")
print("="*80)

ft = pd.read_csv('feature_table.tsv', sep='\t', skiprows=1, index_col=0).T
meta = pd.read_csv('/home/siddharth/liver_cirrhosis/docs/metadata.tsv', sep='\t', index_col=0)

meta_p2 = meta[meta['phase2_label'].isin(['Compensated', 'Decompensated'])].copy()
common = ft.index.intersection(meta_p2.index)
X_raw = ft.loc[common]
y_raw = meta_p2.loc[common, 'phase2_label']
datasets_p2 = meta_p2.loc[common, 'dataset']

le = LabelEncoder()
y = le.fit_transform(y_raw)

n_neg, n_pos = int(np.sum(y == 0)), int(np.sum(y == 1))
scale_pos_weight_p2 = round(n_neg / n_pos, 2)

p2_results = {}

for test_dataset in datasets_p2.unique():
    test_mask = (datasets_p2 == test_dataset).values
    train_mask = ~test_mask

    if test_mask.sum() == 0 or train_mask.sum() == 0:
        continue

    X_train = X_raw.loc[train_mask].div(X_raw.loc[train_mask].sum(axis=1), axis=0).to_numpy(dtype=np.float32)
    X_test  = X_raw.loc[test_mask].div(X_raw.loc[test_mask].sum(axis=1), axis=0).to_numpy(dtype=np.float32)
    y_train = y[train_mask]
    y_test  = y[test_mask]

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test,  counts_test  = np.unique(y_test,  return_counts=True)

    print(f"\nTesting on {test_dataset}:")
    print(f"  Train: {train_mask.sum()} samples | Test: {test_mask.sum()} samples")
    print(f"  Train classes: {dict(zip(le.inverse_transform(unique_train), counts_train))}")
    print(f"  Test classes:  {dict(zip(le.inverse_transform(unique_test),  counts_test))}")

    if len(unique_train) < 2:
        print(f"  SKIPPING: only one class in TRAIN set, SMOTE cannot run")
        p2_results[test_dataset] = {'Accuracy': None, 'ROC-AUC': None, 'F1': None, 'note': 'single-class train set'}
        continue

    if len(unique_test) < 2:
        print(f"  SKIPPING: only one class in TEST set, AUC undefined")
        p2_results[test_dataset] = {'Accuracy': None, 'ROC-AUC': None, 'F1': None, 'note': 'single-class test set'}
        continue

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('xgb', XGBClassifier(
            learning_rate=0.1, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, n_estimators=200,
            scale_pos_weight=scale_pos_weight_p2,
            eval_metric='logloss', random_state=42, verbosity=0
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred       = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1  = f1_score(y_test, y_pred, zero_division=0)

    p2_results[test_dataset] = {'Accuracy': acc, 'ROC-AUC': auc, 'F1': f1}
    print(f"  Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f} | F1: {f1:.4f}")

print("\n" + "-"*80)
print("PHASE 2 LODO Summary:")
for ds, res in sorted(p2_results.items()):
    if res['Accuracy'] is None:
        print(f"  {ds}: SKIPPED - {res['note']}")
    else:
        print(f"  {ds}: Acc={res['Accuracy']:.4f}  AUC={res['ROC-AUC']:.4f}  F1={res['F1']:.4f}")
print("="*80 + "\n")
