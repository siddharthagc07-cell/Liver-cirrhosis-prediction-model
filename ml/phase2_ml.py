import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")

# ── 1. Feature table ───────────────────────────────────────────────────
ft = pd.read_csv('/home/siddharth/liver_cirrhosis/ml/feature_table.tsv',
                 sep='\t', skiprows=1, index_col=0)
ft = ft.T

# ── 2. Metadata ────────────────────────────────────────────────────────
meta = pd.read_csv('/home/siddharth/liver_cirrhosis/docs/metadata.tsv',
                   sep='\t', index_col=0)

# ── 3. Filter to Cirrhosis samples only with phase2 labels ────────────
meta_p2 = meta[meta['phase2_label'].isin(['Compensated', 'Decompensated'])]
common = ft.index.intersection(meta_p2.index)
X_raw = ft.loc[common].copy()
y_raw = meta_p2.loc[common, 'phase2_label'].copy()

print(f"Phase 2 samples: {len(common)}")
print(f"Class distribution:\n{y_raw.value_counts().to_string()}")

# ── 4. TSS normalization ───────────────────────────────────────────────
X = X_raw.div(X_raw.sum(axis=1), axis=0).to_numpy(dtype=np.float32)
print(f"TSS normalization done. Shape: {X.shape}")

# ── 5. Encode labels ───────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 6. Class weight for XGBoost ────────────────────────────────────────
n_neg = int(np.sum(y == 0))
n_pos = int(np.sum(y == 1))
spw   = round(n_neg / n_pos, 4)
print(f"XGBoost scale_pos_weight: {spw}")

# ── 7. Model definitions ───────────────────────────────────────────────
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=500,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        class_weight='balanced',
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
}

# ── 8. Stratified 5-fold CV ────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

decomp_label = int(le.transform(['Decompensated'])[0])
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'roc_auc':  make_scorer(roc_auc_score, needs_proba=True),
    'f1':       make_scorer(f1_score, pos_label=decomp_label, zero_division=0),
}

# ── 9. Run and report ──────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  PHASE 2 — Compensated vs Decompensated  |  5-Fold CV")
print("=" * 62)

all_results = {}
for name, model in models.items():
    print(f"\nRunning {name}...")
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    all_results[name] = {
        'Accuracy':     res['test_accuracy'].mean(),
        'Accuracy_std': res['test_accuracy'].std(),
        'ROC_AUC':      res['test_roc_auc'].mean(),
        'ROC_AUC_std':  res['test_roc_auc'].std(),
        'F1':           res['test_f1'].mean(),
        'F1_std':       res['test_f1'].std(),
    }
    r = all_results[name]
    print(f"  Accuracy : {r['Accuracy']:.3f} +/- {r['Accuracy_std']:.3f}")
    print(f"  ROC-AUC  : {r['ROC_AUC']:.3f} +/- {r['ROC_AUC_std']:.3f}")
    print(f"  F1       : {r['F1']:.3f} +/- {r['F1_std']:.3f}")

# ── 10. Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SUMMARY")
print("=" * 62)
summary = pd.DataFrame(all_results).T[['Accuracy','ROC_AUC','F1']]
summary.columns = ['Accuracy', 'ROC-AUC', 'F1']
print(summary.round(3).to_string())

# ── 11. Save ───────────────────────────────────────────────────────────
pd.DataFrame(all_results).T.to_csv(
    '/home/siddharth/liver_cirrhosis/ml/results/phase2_cv_results.csv')
print("\nFull results saved to: ml/results/phase2_cv_results.csv")
print("=" * 62)
