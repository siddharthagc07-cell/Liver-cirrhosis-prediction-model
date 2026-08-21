import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")

# ── 1. Feature table ───────────────────────────────────────────────────
ft = pd.read_csv('/drive/siddharth/Liver_Cirrhosis_ML/ml/feature_table.tsv',
                 sep='\t', skiprows=1, index_col=0)
ft = ft.T  # samples x genera

# ── 2. Metadata ────────────────────────────────────────────────────────
meta = pd.read_csv('/drive/siddharth/Liver_Cirrhosis_ML/docs/metadata.tsv',
                   sep='\t', index_col=0)

# ── 3. Align samples ───────────────────────────────────────────────────
common = ft.index.intersection(meta.index)
X_raw = ft.loc[common].copy()
y_raw = meta.loc[common, 'phase1_label'].copy()

print(f"Samples: {len(common)}  |  Genera: {X_raw.shape[1]}")
print(f"Class distribution:\n{y_raw.value_counts().to_string()}")

# ── 4. TSS normalization ────────────────────────────────────────────────
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

healthy_label = int(le.transform(['Healthy'])[0])

# ── 7. Outer CV for final evaluation (same as before — fair comparison) ─
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Inner CV used INSIDE GridSearchCV to pick best hyperparameters
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'roc_auc':  make_scorer(roc_auc_score, needs_proba=True),
    'f1':       make_scorer(f1_score, pos_label=healthy_label, zero_division=0),
}

# ── 8. Define models + hyperparameter grids ─────────────────────────────
model_grids = {
    'Random Forest': (
        RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        {
            'n_estimators': [200, 300, 500],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 'log2'],
        }
    ),
    'XGBoost': (
        XGBClassifier(scale_pos_weight=spw, eval_metric='logloss', random_state=42, verbosity=0),
        {
            'n_estimators': [200, 300, 400],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.03, 0.05, 0.1],
        }
    ),
    'SVM': (
        Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
        ]),
        {
            'svc__C': [0.1, 1.0, 5.0],
            'svc__gamma': [0.001, 0.01, 0.1],
        }
    ),
    'Logistic Regression': (
        LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42),
        {
            'C': [0.01, 0.1, 1.0, 10.0],
        }
    ),
}

# ── 9. Run GridSearchCV (hyperparameter tuning) + outer CV (evaluation) ─
print("\n" + "=" * 62)
print("  PHASE 1 (TUNED) — Healthy vs Cirrhosis  |  GridSearchCV + 5-Fold CV")
print("=" * 62)

all_results = {}
best_params_log = {}

for name, (model, grid) in model_grids.items():
    print(f"\nTuning {name}...")

    # Step 1: find best hyperparameters using inner CV, optimizing F1
    f1_scorer = make_scorer(f1_score, pos_label=healthy_label, zero_division=0)
    search = GridSearchCV(model, grid, scoring=f1_scorer, cv=inner_cv, n_jobs=-1)
    search.fit(X, y)

    best_model = search.best_estimator_
    best_params_log[name] = search.best_params_
    print(f"  Best params: {search.best_params_}")

    # Step 2: evaluate the tuned model fairly using outer 5-fold CV
    res = cross_validate(best_model, X, y, cv=outer_cv, scoring=scoring, n_jobs=-1)
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

# ── 10. Summary table ──────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SUMMARY (TUNED)")
print("=" * 62)
summary = pd.DataFrame(all_results).T[['Accuracy','ROC_AUC','F1']]
summary.columns = ['Accuracy', 'ROC-AUC', 'F1']
print(summary.round(3).to_string())

print("\n" + "=" * 62)
print("  BEST HYPERPARAMETERS FOUND")
print("=" * 62)
for name, params in best_params_log.items():
    print(f"{name}: {params}")

# ── 11. Save ───────────────────────────────────────────────────────────
out = pd.DataFrame(all_results).T
out.to_csv('/drive/siddharth/Liver_Cirrhosis_ML/ml/results/phase1_tuned_cv_results.csv')

params_out = pd.DataFrame(best_params_log).T
params_out.to_csv('/drive/siddharth/Liver_Cirrhosis_ML/ml/results/phase1_best_params.csv')

print("\nTuned results saved to: ml/results/phase1_tuned_cv_results.csv")
print("Best hyperparameters saved to: ml/results/phase1_best_params.csv")
print("=" * 62)
