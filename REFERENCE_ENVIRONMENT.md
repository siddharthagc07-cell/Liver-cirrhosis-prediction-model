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

## Cloud validation — ICE Cloud (2026-08-20)
Full reproducibility test performed on ICE Cloud after migrating off the
Sonali machine (returned to owner). Environment built via mamba (conda's
solver could not resolve environment.yml's full dependency set within a
reasonable memory/time budget on this platform; mamba succeeded cleanly).

### Environment
- Instance: ICE Cloud, 24 vCPU / 12GB RAM
- Python: 3.10.14 — exact match
- numpy 1.26.4, pandas 2.2.2, scikit-learn 1.4.2, xgboost 3.2.0,
  shap 0.49.1, scipy 1.13.0, joblib 1.4.2 — all exact matches
- Data checksums verified identical to reference above before running
  (ml/feature_table.tsv, docs/metadata.tsv)

### Result
phase1_ml_gridsearch.py produced results identical to the reference
table above to three decimal places, for all four models, including
best hyperparameters found by GridSearchCV:

| Model | Accuracy | ROC-AUC | F1 | Match |
|---|---|---|---|---|
| Random Forest | 0.733 | 0.802 | 0.681 | Exact |
| XGBoost | 0.692 | 0.770 | 0.673 | Exact |
| SVM | 0.697 | 0.753 | 0.609 | Exact |
| Logistic Regression | 0.692 | 0.708 | 0.602 | Exact |

This exceeds the ~0.5-1% tolerance anticipated in the determinism
caveat above. The theoretical concern about RandomForest n_jobs=-1
floating-point drift across different CPU/thread counts (4 threads on
Sonali vs. 24 vCPU on ICE Cloud) did not manifest in this run.

## Inference pipeline validation — ICE Cloud (2026-08-21)
Full end-to-end raw-FASTQ-to-prediction test performed on ICE Cloud using
pipeline_v2.py, to confirm the complete inference path (not just the
training/CV script above) reproduces correctly on cloud infrastructure.

### Environment gotchas found and fixed
Building the mamba-based lc_qiime environment left a broken jpeg setup
that only surfaced when running pipeline_v2.py (phase1_ml_gridsearch.py
does not touch these libraries, so this was not caught by the training
validation above). Two libjpeg consumers in the same environment need
different, ABI-incompatible versions:
- R's dada2 package requires libjpeg.so.9 (from conda-forge/defaults
  package jpeg=9b=habf39ab_1)
- Python's PIL/matplotlib (pulled in via seaborn, used by QIIME2's
  demux summarize step) requires libjpeg.so.8 (from libjpeg-turbo)
Conda cannot have both jpeg and libjpeg-turbo installed as tracked
packages simultaneously (dependency conflict). Fix: install jpeg=9b
normally, then manually copy a real libjpeg.so.8.3.2 file (sourced from
a disposable temporary conda env created just to extract it) directly
into $CONDA_PREFIX/lib/ alongside the .so.9 files, so both coexist on
disk outside conda's own package tracking. If rebuilding this
environment from scratch, expect to redo this manual step - it is not
captured by environment.yml.

Separately, pipeline_v2.py expects the Silva classifier at
ml/reference/silva-138-99-nb-classifier.qza; if downloaded elsewhere
(e.g. qiime/ as instructed in README_SETUP.md), symlink it into place.

### Result
Sample SRR33459675 (paired-end, PRJNA1259947 protocol) processed
through the full pipeline (import, cutadapt, DADA2, Silva
classification, genus collapse, SHAP explanation) on ICE Cloud:

| | Original (Sonali) | ICE Cloud |
|---|---|---|
| Predicted class | Cirrhosis | Cirrhosis |
| Confidence | 86.2% | 86.2% |
| Genera matched | 69/69 | 69/69 |
| Genera dropped | 0 | 0 |

Exact match. Runtime 4.6 minutes. This confirms the full inference
pipeline, not just model training, reproduces correctly on ICE Cloud.

## Single-end inference pipeline validation — ICE Cloud (2026-09-11)
Full end-to-end test of the single-end/Deblur code path in pipeline_v2.py
(distinct from the paired-end/DADA2 path validated above), using demo
sample SRR26159183 (PRJNA1019460 protocol).

### Result
| | Original (Sonali) | ICE Cloud |
|---|---|---|
| Predicted class | Healthy | Healthy |
| Confidence | 87.8% | 87.8% |
| Genera matched | 33/33 | 33/33 |
| Genera dropped | 0 | 0 |

Exact match. This confirms both inference code paths (paired-end/DADA2
and single-end/Deblur) reproduce correctly on ICE Cloud, completing full
pipeline validation - not just model training, but the entire raw
sample-to-prediction flow for every dataset type in this project.

### Platform caveat: ICE Cloud session stability
ICE Cloud's terminal runs on Kubernetes (hostnames follow the pattern
siddharth-deploy-<id1>-<id2>-<id3>). During this validation, the
underlying pod was observed being replaced mid-session multiple times
without any user-initiated disconnect, killing in-progress jobs
(including ones backgrounded with nohup, since nohup only survives a
terminal disconnect, not the pod itself being torn down). Several
single-end test attempts were lost this way before one finally
completed on a pod that happened to stay alive long enough. For any
job expected to run more than a few minutes, check progress frequently
(every 60-90 seconds) and be prepared to restart; consider
checkpointing long jobs to persistent storage where possible.

## Training reproducibility fix — XGBoost non-determinism (2026-09-23)
Mentor requirement: training metrics must reproduce exactly across
repeated runs. RF, SVM, and Logistic Regression were already exactly
reproducible (explicit random_state set on all splitters and models).
XGBoost showed small run-to-run drift in ROC-AUC and F1 despite a
fixed random_state, traced to its default multi-threaded histogram
tree-building (n_jobs unset -> all cores), which is not guaranteed to
sum floating-point operations in the same order every run.

### Fix
Added n_jobs=1, tree_method='exact' to the XGBClassifier constructor
in phase1_ml_gridsearch.py, forcing single-threaded, deterministic
tree construction.

### Verification: two independent back-to-back runs post-fix
| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Random Forest | 0.733 | 0.802 | 0.681 |
| XGBoost | 0.692 | 0.768 | 0.677 |
| SVM | 0.697 | 0.753 | 0.609 |
| Logistic Regression | 0.692 | 0.708 | 0.602 |

Both runs produced bit-for-bit identical metrics and identical best
hyperparameters for all four models. All training metrics are now
exactly reproducible.

Note: XGBoost's ROC-AUC/F1 shifted slightly from the previously locked
values (0.770/0.673 -> 0.768/0.677). This reflects the previously
undetected multi-threading non-determinism, not a regression - the new
values are the stable, permanently reproducible ones going forward.
