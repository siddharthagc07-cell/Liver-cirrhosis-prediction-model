import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")

# 1. Feature table (same load as phase1_ml_gridsearch.py)
ft = pd.read_csv('/home/siddharth/liver_cirrhosis/ml/feature_table.tsv',
                 sep='\t', skiprows=1, index_col=0)
ft = ft.T  # samples x genera

# 2. Metadata
meta = pd.read_csv('/home/siddharth/liver_cirrhosis/docs/metadata.tsv',
                   sep='\t', index_col=0)

# 3. Align samples
common = ft.index.intersection(meta.index)
X_raw = ft.loc[common].copy()
y_raw = meta.loc[common, 'phase1_label'].copy()
print(f"Samples: {len(common)}  |  Genera: {X_raw.shape[1]}")

# 4. Lock the column order NOW, before normalization —
#    this exact order is what align_features() will match new samples against later
training_columns = X_raw.columns.tolist()

# 5. TSS normalization (identical to training script)
X = X_raw.div(X_raw.sum(axis=1), axis=0).to_numpy(dtype=np.float32)

# 6. Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 7. Train ONE final RF on ALL 195 samples, using the locked hyperparameters
rf_final = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    max_features='log2',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_final.fit(X, y)
print("Final RF model trained on full 195-sample dataset.")

# 8. Bundle model + encoder + column order together — all three are needed
#    at prediction time, so they must never drift apart from each other
bundle = {
    'model': rf_final,
    'label_encoder': le,
    'training_columns': training_columns,
}
joblib.dump(bundle, '/home/siddharth/liver_cirrhosis/ml/pipeline_model.pkl')
print("Saved bundle to pipeline_model.pkl")

