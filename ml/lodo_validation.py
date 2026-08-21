import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

os.makedirs('/drive/siddharth/Liver_Cirrhosis_ML/ml/results', exist_ok=True)

print("Loading data...")
ft = pd.read_csv('/drive/siddharth/Liver_Cirrhosis_ML/ml/feature_table.tsv',
                 sep='\t', skiprows=1, index_col=0)
ft = ft.T

meta = pd.read_csv('/drive/siddharth/Liver_Cirrhosis_ML/docs/metadata.tsv',
                   sep='\t', index_col=0)

common = ft.index.intersection(meta.index)
X_raw = ft.loc[common]
y_raw = meta.loc[common, 'phase1_label']
dataset = meta.loc[common, 'dataset']

X = X_raw.div(X_raw.sum(axis=1), axis=0)
le = LabelEncoder()
y = le.fit_transform(y_raw)

datasets = dataset.unique()
print(f"Datasets: {datasets}")
print(f"Total samples: {len(common)}")

models = {
    'Random Forest': RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                             class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost':       XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                    eval_metric='logloss', random_state=42, verbosity=0),
}

print("\n" + "="*65)
print("  LODO VALIDATION — Leave One Dataset Out")
print("="*65)

all_results = []
for test_ds in datasets:
    train_mask = dataset != test_ds
    test_mask  = dataset == test_ds

    X_train = X[train_mask].to_numpy(dtype=np.float32)
    X_test  = X[test_mask].to_numpy(dtype=np.float32)
    y_train = y[train_mask]
    y_test  = y[test_mask]

    print(f"\nTest dataset: {test_ds}")
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"  Test class dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = float('nan')
        f1 = f1_score(y_test, y_pred, zero_division=0,
                      pos_label=le.transform(['Healthy'])[0])

        print(f"  {name}: Accuracy={acc:.3f} | ROC-AUC={auc:.3f} | F1={f1:.3f}")
        all_results.append({
            'Test_Dataset': test_ds, 'Model': name,
            'Accuracy': acc, 'ROC_AUC': auc, 'F1': f1,
            'Train_N': len(X_train), 'Test_N': len(X_test)
        })

results_df = pd.DataFrame(all_results)
results_df.to_csv('/drive/siddharth/Liver_Cirrhosis_ML/ml/results/lodo_results.csv', index=False)

print("\n" + "="*65)
print("  LODO SUMMARY")
print("="*65)
pivot = results_df.pivot_table(index='Test_Dataset', columns='Model',
                                values='ROC_AUC')
print(pivot.round(3).to_string())
print("\nFull results saved to: ml/results/lodo_results.csv")
print("="*65)
