import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

os.makedirs('/home/siddharth/liver_cirrhosis/ml/results/plots', exist_ok=True)

print("Loading data...")
ft = pd.read_csv('/home/siddharth/liver_cirrhosis/ml/feature_table.tsv',
                 sep='\t', skiprows=1, index_col=0)
ft = ft.T

meta = pd.read_csv('/home/siddharth/liver_cirrhosis/docs/metadata.tsv',
                   sep='\t', index_col=0)

common = ft.index.intersection(meta.index)
X_raw = ft.loc[common]
y_raw = meta.loc[common, 'phase1_label']
X = X_raw.div(X_raw.sum(axis=1), axis=0).to_numpy(dtype=np.float32)

le = LabelEncoder()
y = le.fit_transform(y_raw)

pos_weight = float(np.sum(y==0)) / float(np.sum(y==1))

models = {
    'Random Forest': RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                             class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost':       XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                    scale_pos_weight=pos_weight, eval_metric='logloss',
                                    random_state=42, verbosity=0),
    'SVM':           SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    'Logistic Reg':  LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

# ── ROC Curve ──────────────────────────────────────────────────────────
print("Generating ROC curves...")
plt.figure(figsize=(8, 7))

for (name, model), color in zip(models.items(), colors):
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y[test_idx], proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=color, lw=2,
             label=f'{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

plt.plot([0,1],[0,1],'k--', lw=1, label='Random (AUC = 0.500)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves — Phase 1: Healthy vs Cirrhosis\n5-Fold Cross Validation', fontsize=13)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/siddharth/liver_cirrhosis/ml/results/plots/roc_curves.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: roc_curves.png")

# ── Confusion Matrices ─────────────────────────────────────────────────
print("Generating confusion matrices...")
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, (name, model) in zip(axes, models.items()):
    y_pred_all = np.zeros_like(y)
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred_all[test_idx] = model.predict(X[test_idx])
    cm = confusion_matrix(y, y_pred_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=11)

plt.suptitle('Confusion Matrices — Phase 1: Healthy vs Cirrhosis (5-Fold CV)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/siddharth/liver_cirrhosis/ml/results/plots/confusion_matrices.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: confusion_matrices.png")
print("\nAll plots saved to: ml/results/plots/")
print("Done.")
