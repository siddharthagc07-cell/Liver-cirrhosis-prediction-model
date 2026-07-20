import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ft = pd.read_csv('/home/siddharth/liver_cirrhosis/ml/feature_table.tsv',
                 sep='\t', skiprows=1, index_col=0)
genus_names = ft.index.tolist()
ft = ft.T

meta = pd.read_csv('/home/siddharth/liver_cirrhosis/docs/metadata.tsv',
                   sep='\t', index_col=0)

common = ft.index.intersection(meta.index)
X_raw = ft.loc[common]
y_raw = meta.loc[common, 'phase1_label']

X = X_raw.div(X_raw.sum(axis=1), axis=0)

le = LabelEncoder()
y = le.fit_transform(y_raw)

rf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                             class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
top20 = importances.sort_values(ascending=False).head(20)

print("Top 20 genera by Random Forest importance:\n")
for i, (genus, imp) in enumerate(top20.items(), 1):
    parts = genus.split(';')
    name = parts[-1].strip() if parts[-1].strip() not in ['__', ''] else parts[-2].strip()
    print(f"  {i:2d}. {name:<50} importance={imp:.4f}")

out = pd.DataFrame({'genus': X.columns, 'importance': rf.feature_importances_})
out = out.sort_values('importance', ascending=False)
out.to_csv('/home/siddharth/liver_cirrhosis/ml/results/feature_importance.csv', index=False)
print("\nFull table saved to results/feature_importance.csv")
