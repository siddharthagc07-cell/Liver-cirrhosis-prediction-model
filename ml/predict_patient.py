import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Load training data ─────────────────────────────────────────────────
ft = pd.read_csv('/home/siddharth/liver_cirrhosis/ml/feature_table.tsv',
                 sep='\t', skiprows=1, index_col=0)
ft = ft.T
meta = pd.read_csv('/home/siddharth/liver_cirrhosis/docs/metadata.tsv',
                   sep='\t', index_col=0)

common = ft.index.intersection(meta.index)
X_raw = ft.loc[common]
y_raw = meta.loc[common, 'phase1_label']
X = X_raw.div(X_raw.sum(axis=1), axis=0)

le = LabelEncoder()
y = le.fit_transform(y_raw)

# ── Train model on all 195 samples ────────────────────────────────────
rf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                             class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X.to_numpy(dtype=np.float32), y)

# ── Simulate new patient — pick random sample as demo ─────────────────
# In real deployment this would be replaced by new patient fastq → QIIME2 → genus profile
patient_id = sys.argv[1] if len(sys.argv) > 1 else X.index[np.random.randint(len(X))]

if patient_id not in X.index:
    print(f"Patient {patient_id} not found. Available: {list(X.index[:5])}...")
    sys.exit(1)

patient_X = X.loc[[patient_id]].to_numpy(dtype=np.float32)
proba = rf.predict_proba(patient_X)[0]
pred  = le.inverse_transform([rf.predict(patient_X)[0]])[0]
true  = y_raw.get(patient_id, 'Unknown')

# ── Report ─────────────────────────────────────────────────────────────
print("\n" + "="*52)
print("    LIVER CIRRHOSIS MICROBIOME PREDICTION REPORT")
print("="*52)
print(f"  Patient ID     : {patient_id}")
print(f"  True Label     : {true}")
print(f"  Prediction     : {pred}")
print(f"  Confidence     :")
for cls, p in zip(le.classes_, proba):
    bar = '█' * int(p * 30)
    print(f"    {cls:<14} {bar} {p:.1%}")

risk = "HIGH RISK" if pred == 'Cirrhosis' else "LOW RISK"
print(f"\n  Clinical Flag  : ⚠ {risk}" if pred == 'Cirrhosis' else f"\n  Clinical Flag  : ✓ {risk}")

print("\n  Top 5 microbial drivers:")
importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:5]
for i, idx in enumerate(top_idx, 1):
    genus = X.columns[idx].split(';')[-1].strip()
    abundance = X.loc[patient_id].iloc[idx]
    direction = "↑ elevated" if abundance > X.iloc[:, idx].median() else "↓ depleted"
    print(f"  {i}. {genus:<38} {direction}  ({abundance:.3f})")
print("="*52 + "\n")
