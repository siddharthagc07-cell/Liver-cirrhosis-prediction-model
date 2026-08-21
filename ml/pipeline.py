#!/usr/bin/env python3
import sys, os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import warnings
warnings.filterwarnings('ignore')

BASE        = '/drive/siddharth/Liver_Cirrhosis_ML'
FEATURE_TSV = f'{BASE}/ml/feature_table.tsv'
METADATA    = f'{BASE}/docs/metadata.tsv'

def clean_genus(name):
    parts = name.split(';')
    for part in reversed(parts):
        cleaned = part.strip().replace('g__','').replace('f__','').replace('__','').strip()
        if cleaned:
            return cleaned
    return name

def load_and_train():
    ft     = pd.read_csv(FEATURE_TSV, sep='\t', skiprows=1, index_col=0).T
    meta   = pd.read_csv(METADATA, sep='\t', index_col=0)
    common = ft.index.intersection(meta.index)
    X_raw  = ft.loc[common]
    y_raw  = meta.loc[common, 'phase1_label']
    X      = X_raw.div(X_raw.sum(axis=1), axis=0)
    X.columns = [clean_genus(c) for c in X.columns]
    le     = LabelEncoder()
    y      = le.fit_transform(y_raw)
    rf     = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                     class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X.to_numpy(dtype=np.float32), y)
    return rf, le, X, y_raw

def get_drivers(rf, X, patient_X):
    explainer  = shap.TreeExplainer(rf)
    shap_vals  = explainer.shap_values(patient_X.to_numpy(dtype=np.float32))
    shap_cirr  = shap_vals[:, :, 0][0]
    total      = np.abs(shap_cirr).sum()
    shap_pct   = np.abs(shap_cirr) / total * 100 if total > 0 else np.zeros_like(shap_cirr)
    top_idx    = np.argsort(np.abs(shap_cirr))[::-1][:3]
    drivers = []
    for idx in top_idx:
        if shap_pct[idx] < 0.01:
            continue
        abundance = float(patient_X.iloc[0, idx])
        median    = float(X.iloc[:, idx].median())
        drivers.append({
            'genus':     X.columns[idx],
            'direction': '↑ elevated' if abundance > median else '↓ depleted',
            'shap_pct':  round(float(shap_pct[idx]), 1)
        })
    return drivers

def get_model_performance():
    try:
        cv = pd.read_csv(f'{BASE}/ml/results/phase1_cv_results.csv', index_col=0)
        rf_row   = cv.loc['Random Forest']
        mean_f1  = round(rf_row['F1'], 3)
        mean_auc = round(rf_row['ROC_AUC'], 3)
        mean_acc = round(rf_row['Accuracy'], 3)
    except:
        mean_f1, mean_auc, mean_acc = 'N/A', 'N/A', 'N/A'
    return mean_f1, mean_auc, mean_acc

if __name__ == '__main__':
    if len(sys.argv) == 3:
        fastq_path = sys.argv[1]  # forward read; reverse (sys.argv[2]) confirms paired-end
    elif len(sys.argv) == 2:
        fastq_path = sys.argv[1]
    else:
        print("Usage:")
        print("  Paired-end: python3 pipeline.py <sample>_1.fastq.gz <sample>_2.fastq.gz")
        print("  Single-end: python3 pipeline.py <sample>.fastq.gz")
        sys.exit(1)
    if not os.path.exists(fastq_path):
        print(f"Error: File not found: {fastq_path}")
        sys.exit(1)

    sample_id = os.path.basename(fastq_path).split('_')[0].split('.')[0]

    print("Loading model...", flush=True)
    rf, le, X, y_raw = load_and_train()

    if sample_id not in X.index:
        print(f"Error: {sample_id} not found in feature table.")
        sys.exit(1)

    patient_X  = X.loc[[sample_id]]
    proba      = rf.predict_proba(patient_X.to_numpy(dtype=np.float32))[0]
    pred       = le.inverse_transform([rf.predict(patient_X.to_numpy(dtype=np.float32))[0]])[0]
    classes    = list(le.classes_)
    cirr_pct   = round(proba[classes.index('Cirrhosis')] * 100, 1)
    hlth_pct   = round(proba[classes.index('Healthy')]   * 100, 1)
    confidence = max(cirr_pct, hlth_pct)

    print("Computing SHAP...", flush=True)
    drivers = get_drivers(rf, X, patient_X)
    mean_f1, mean_auc, mean_acc = get_model_performance()

    W = 58
    print()
    print("=" * W)
    print("         LIVER CIRRHOSIS PREDICTION REPORT")
    print("=" * W)
    print(f"  Patient ID        : {sample_id}")
    print("-" * W)
    print(f"  Predicted Class   : {pred}")
    print(f"  Confidence Score  : {confidence}%")
    print()
    print("  Class Probabilities:")
    print(f"    Healthy         : {hlth_pct}%")
    print(f"    Cirrhosis       : {cirr_pct}%")
    print()
    print("  Key Microbial Features:")
    for d in drivers:
        print(f"    {d['genus']:<30} {d['direction']}   Contribution: {d['shap_pct']}%")
    print("-" * W)
    print("  MODEL INFORMATION")
    print("-" * W)
    print(f"  Selected Model    : Random Forest")
    print(f"  Selection Method  : Best Mean CV F1 Score")
    print()
    print(f"  Model Performance:")
    print(f"    Mean F1 Score   : {mean_f1}")
    print(f"    Mean AUROC      : {mean_auc}")
    print(f"    Mean Accuracy   : {mean_acc}")
    print("-" * W)
    print("  Interpretation:")
    if pred == 'Cirrhosis':
        print(f"  Microbiome profile shows patterns consistent with")
        print(f"  liver cirrhosis. Key genera show dysbiosis.")
    else:
        print(f"  Microbiome profile is consistent with a healthy gut.")
        print(f"  No significant dysbiosis detected.")
    print("=" * W)
    print()
