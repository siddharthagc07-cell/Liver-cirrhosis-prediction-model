# Reference Environment & Metrics

Captured from the authoritative working environment (Sonali WSL2 machine)
before any cloud migration. Compare future environments/results against
this file, not against old report figures from memory.

## Code state
- Git commit: 8e139ea6a3d2799fe0346bfef6007a70687c4cd9
- Repo: https://github.com/siddharthagc07-cell/Liver-cirrhosis-prediction-model

## Environment
- Python: 3.10.14 (conda-forge, GCC 12.3.0)
- numpy 1.26.4, pandas 2.2.2, scikit-learn 1.4.2, xgboost 3.2.0,
  shap 0.49.1, scipy 1.13.0, joblib 1.4.2
- Full conda env export: see environment.yml in repo root
- CPU: Intel Core i5-11400H, 4 threads (2 cores, hyperthreaded)
  (NOTE: different CPU architecture on a cloud VM may cause small
  numerical drift in RandomForest split-finding even with identical
  package versions and random_state fixed — see RandomForest/threading
  determinism caveat below.)

## Data checksums
- ml/feature_table.tsv: b698406cf1a774ec5defefd27cec836a (430 lines)
- docs/metadata.tsv: 200ce804512e30122801baac430e0062 (196 lines)
- ml/pipeline_model.pkl: 7f932f173502e1a47492350a3a8ae2e9

## Reference metrics (from phase1_ml_gridsearch.py, Stratified 5-Fold CV)

| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Random Forest | 0.733 | 0.802 | 0.681 |
| XGBoost | 0.692 | 0.770 | 0.673 |
| SVM | 0.697 | 0.753 | 0.609 |
| Logistic Regression | 0.692 | 0.708 | 0.602 |

Best hyperparameters:
- Random Forest: max_depth=None, max_features='log2', n_estimators=500
- XGBoost: learning_rate=0.1, max_depth=3, n_estimators=200
- SVM: C=5.0, gamma=0.001
- Logistic Regression: C=1.0

## Determinism caveat
random_state is fixed throughout (42 for outer CV / model fitting, 1 for
inner CV), but RandomForestClassifier uses n_jobs=-1 (multi-threaded).
Floating-point summation order in multi-threaded tree building can vary
slightly across different CPU architectures/thread counts, meaning
bit-for-bit identical results across machines are not guaranteed even
with identical package versions. A small numerical drift (e.g. in the
3rd decimal place) on a different CPU is expected and not a bug -
treat exact match as the target, small documented drift as acceptable.

## Validation procedure for a new environment
1. Create environment from environment.yml (or match package versions above)
2. Copy feature_table.tsv and metadata.tsv, verify checksums match
3. Run phase1_ml_gridsearch.py
4. Compare output table against the reference metrics above
5. Differences beyond ~0.5-1% in any metric warrant investigation
   (check package versions first, then CPU/threading differences)
