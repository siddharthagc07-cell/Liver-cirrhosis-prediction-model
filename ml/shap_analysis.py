import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Loading data...")
ft = pd.read_csv('/drive/siddharth/Liver_Cirrhosis_ML/ml/feature_table.tsv',
                 sep='\t', skiprows=1, index_col=0)
ft = ft.T

meta = pd.read_csv('/drive/siddharth/Liver_Cirrhosis_ML/docs/metadata.tsv',
                   sep='\t', index_col=0)

common = ft.index.intersection(meta.index)
X_raw = ft.loc[common]
y_raw = meta.loc[common, 'phase1_label']

X = X_raw.div(X_raw.sum(axis=1), axis=0)

def clean_genus(name):
    parts = name.split(';')
    last = parts[-1].strip()
    return last if last not in ['__', '', 'g__'] else parts[-2].strip()

X.columns = [clean_genus(c) for c in X.columns]

le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                             class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X, y)

print("Computing SHAP values (1-2 mins)...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

os.makedirs('/drive/siddharth/Liver_Cirrhosis_ML/ml/results/plots', exist_ok=True)

# Beeswarm plot
print("Saving beeswarm plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[:, :, 0], X, max_display=20, show=False, plot_type='dot')
plt.title('SHAP Summary — Cirrhosis vs Healthy\n(positive SHAP = pushes toward Cirrhosis)', fontsize=12)
plt.tight_layout()
plt.savefig('/drive/siddharth/Liver_Cirrhosis_ML/ml/results/plots/shap_beeswarm.png',
            dpi=150, bbox_inches='tight')
plt.close()

# Bar plot
print("Saving bar plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[:, :, 0], X, max_display=20, show=False, plot_type='bar')
plt.title('Mean Absolute SHAP — Top 20 Genera', fontsize=12)
plt.tight_layout()
plt.savefig('/drive/siddharth/Liver_Cirrhosis_ML/ml/results/plots/shap_bar.png',
            dpi=150, bbox_inches='tight')
plt.close()

# Print directions
shap_matrix = shap_values[:, :, 0]
summary = pd.DataFrame({
    'genus': X.columns,
    'mean_abs_shap': np.abs(shap_matrix).mean(axis=0),
    'mean_shap': shap_matrix.mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False).head(20)

print("\nTop 20 genera — direction:\n")
for _, row in summary.iterrows():
    direction = "→ Cirrhosis" if row['mean_shap'] > 0 else "→ Healthy  "
    print(f"  {row['genus']:<45} {direction}  shap={row['mean_shap']:+.4f}")

summary.to_csv('/drive/siddharth/Liver_Cirrhosis_ML/ml/results/shap_top20.csv', index=False)
print("\nPlots saved to: ml/results/plots/")
print("Done.")
