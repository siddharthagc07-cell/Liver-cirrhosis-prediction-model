
# 🧬 Healthy vs Cirrhosis Classification Using Gut Microbiome Data

A bioinformatics and machine learning project that processes **16S rRNA sequencing data** to investigate differences in the gut microbiome between Healthy individuals and individuals with Cirrhosis.

The pipeline converts raw sequencing reads into a genus-level microbial abundance table and uses machine learning to classify samples into the two groups.

> **Key Result:** Random Forest achieved a ROC-AUC of **0.802** in the tuned five-fold cross-validation evaluation, making it the best-performing model among the four evaluated.
>
> **Important:** This is an exploratory research project, and not a legitimate diagnostic tool.

Built using publicly available sequencing data. No wet-lab experiments were performed.

---

## 🔍 Project Overview

The gut microbiome is connected to liver health through the gut–liver axis. Changes in microbial composition have been investigated in several liver diseases, including cirrhosis.

This project explores whether genus-level gut microbiome profiles can distinguish Healthy samples from Cirrhosis samples using machine learning.

### Objectives

- Process raw 16S rRNA sequencing data using QIIME 2.
- Generate a genus-level microbial abundance table.
- Train and compare multiple machine learning classifiers.
- Evaluate model performance using cross-validation.
- Explore model predictions using SHAP explainability.

## 📊 Dataset

The dataset combines publicly available sequencing data from three NCBI BioProjects.

| Dataset | Sequencing Platform | Read Layout | Healthy | Cirrhosis* |
|---|---|---|---:|---:|
| PRJNA1259947 | NovaSeq 6000 | Paired-end | 40 | 83 |
| PRJNA471972 | MiSeq | Paired-end | 20 | 12 |
| PRJNA1019460 | MiSeq | Single-end | 33 | 7 |
| **Total** | — | — | **93** | **102** |

**Dataset summary**
- **195 samples** in total.
- **429 bacterial genera** in the genus-level feature table.
- Data derived from human stool samples.
- 16S rRNA amplicon sequencing data.

\*The Cirrhosis group includes 12 samples from patients with acute-on-chronic liver failure (ACLF), which were grouped with Cirrhosis for this binary classification task.

### Dataset Considerations

The studies differ in sequencing platform, read layout, sequencing depth, and class distribution. These differences may influence the observed microbial profiles and should be considered when interpreting model performance.

## ⚙️ Bioinformatics Pipeline

The workflow transforms raw sequencing reads into a machine learning-ready dataset.

```text
Raw FASTQ Files
      │
      ▼
QIIME 2 Import
      │
      ▼
Primer Removal
(Cutadapt, where applicable)
      │
      ▼
Read Denoising
(DADA2 / Deblur)
      │
      ▼
ASV Feature Table
      │
      ▼
Taxonomic Classification
(SILVA 138)
      │
      ▼
Genus-Level Abundance Table
      │
      ▼
Total-Sum Scaling (TSS)
      │
      ▼
Machine Learning Models
      │
      ▼
Performance Evaluation
      │
      ▼
SHAP Explainability
```

### Pipeline Steps

| Step | Method | Purpose |
|---|---|---|
| Sequence processing | QIIME 2 | Import and process sequencing reads |
| Primer removal | Cutadapt | Remove primers where required |
| Denoising | DADA2 / Deblur | Infer amplicon sequence variants (ASVs) |
| Taxonomy | SILVA 138 | Assign taxonomic classifications |
| Feature preparation | Genus aggregation | Construct genus-level abundance features |
| Normalization | Total-Sum Scaling (TSS) | Convert counts to relative abundances |
| Classification | Scikit-learn / XGBoost | Train and compare classifiers |
| Explainability | SHAP | Examine feature contributions to predictions |

The machine learning scripts use TSS-normalized genus-level features.

## 🤖 Machine Learning

Four classifiers were evaluated:

- **Random Forest**
- **XGBoost**
- **Support Vector Machine (SVM)**
- **Logistic Regression**

### Model Evaluation

The models were evaluated using standard five-fold cross-validation. Two evaluations were performed: one with fixed model settings and another using GridSearchCV for hyperparameter tuning.

### Results

#### Fixed Model Settings

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Random Forest | 0.723 | **0.806** |
| XGBoost | 0.703 | 0.778 |
| SVM | 0.651 | 0.686 |
| Logistic Regression | 0.692 | 0.708 |

#### Hyperparameter-Tuned Models

| Model | Accuracy | ROC-AUC | F1-Score (Healthy) |
|---|---:|---:|---:|
| Random Forest | **0.733** | **0.802** | **0.681** |
| XGBoost | 0.692 | 0.768 | 0.677 |
| SVM | 0.697 | 0.753 | 0.609 |
| Logistic Regression | 0.692 | 0.708 | 0.602 |

*The F1-score in the tuned evaluation refers to the Healthy class. Healthy is encoded as 1 and Cirrhosis as 0.*

### Best-Performing Model: Random Forest

Random Forest achieved the highest ROC-AUC in both evaluations:

- **ROC-AUC:** 0.806 with fixed settings.
- **ROC-AUC:** 0.802 after hyperparameter tuning.
- **Accuracy:** 0.733 in the tuned evaluation.

Random Forest was selected as the final model based on its performance across the two evaluations.

**Interpretation:** These results indicate that the model can distinguish between the two classes to some extent under the evaluated cross-validation setup.

## 🔬 Model Explainability

SHAP (SHapley Additive exPlanations) is used to examine how microbial features contribute to model predictions.

The explanations help investigate:
- Which microbial genera influence model predictions.
- Whether particular features push predictions toward Healthy or Cirrhosis.
- How feature contributions vary across samples.

SHAP describes how the trained model uses its features. A feature's importance does not establish that the microorganism causes or protects against cirrhosis.

## 🧪 Input Requirements

The current pipeline is designed for a specific set of 16S rRNA sequencing inputs.

**Supported data**
- Human stool samples.
- Illumina MiSeq or NovaSeq 6000 data.
- Paired-end FASTQ files with the expected primers attached.
- Single-end FASTQ files that have already been primer-trimmed.

### Study-Specific Processing Settings

| BioProject | Read Layout | 16S Region | Processing |
|---|---|---|---|
| PRJNA471972 | Paired-end | V3–V4 | DADA2; forward truncation 260 bp, reverse truncation 200 bp |
| PRJNA1259947 | Paired-end | Not documented in the cited ENA description | DADA2; forward truncation 230 bp, reverse truncation 200 bp |
| PRJNA1019460 | Single-end | Not documented | Deblur; trim length 250 bp |

For PRJNA471972, the primer sequences used are:

- Forward: `CCTACGGGNGGCWGCAG`
- Reverse: `GACTACHVGGGTATCTAATCC`

For PRJNA1259947:

- Forward: `TACGGRAGGCAGCAG`
- Reverse: `AGGGTATCTAATCCT`

The pipeline also requires the appropriate SILVA 138 classifier and the downstream genus aggregation and TSS normalization steps.

**Important:** Paired-end reads without the required primers are discarded by the current preprocessing workflow. Paired-end files that have already had these primers removed are not supported by the current configuration.

Shotgun metagenomics, other sample types, other primer sets, alternative sequencing platforms, and different reference databases have not been validated for this pipeline.

## 🚀 Usage

The inference script is:

`ml/pipeline_v2.py`

It is intended to process supported sequencing inputs and produce:

- Predicted class: Healthy or Cirrhosis.
- Prediction probability.
- SHAP-based feature explanation.

Refer to [`README_SETUP.md`](README_SETUP.md) for environment setup and execution instructions.

**Note:** The scripts contain environment-specific paths that may need to be updated before running the pipeline on another machine.

## ⚠️ Limitations

- The model is trained on limited data (195 samples from three public studies).
- The model is specific to the sequencing setup it was built for: the target 16S region, primers, read layout and processing settings listed in Input Requirements. It does not work beyond them.
- The model is strictly specific to the data it was trained on.
- The project does not have a graphical user interface (GUI).

## 🔮 Future Work

- Add more data in the future, so the model is exposed to new patterns.
- Improve accuracy and the other evaluation metrics.
- Make the model available for public use once it is trained well on diverse conditions.

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Bioinformatics | QIIME 2, DADA2, Deblur |
| Taxonomy | SILVA 138 |
| Programming | Python |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| Environment | Linux, Ubuntu |

## 📁 Project Resources

| Resource | File |
|---|---|
| Setup instructions | [`README_SETUP.md`](README_SETUP.md) |
| Reference environment and results | [`REFERENCE_ENVIRONMENT.md`](REFERENCE_ENVIRONMENT.md) |
| Tuned model evaluation | [`ml/phase1_ml_gridsearch.py`](ml/phase1_ml_gridsearch.py) |
| Fixed-settings model evaluation | [`ml/phase1_ml.py`](ml/phase1_ml.py) |
| Within-study evaluation | [`validation/within_cohort.py`](validation/within_cohort.py) |
| Model saving | [`ml/save_model.py`](ml/save_model.py) |
| Prediction pipeline | [`ml/pipeline_v2.py`](ml/pipeline_v2.py) |

## 📚 Data Sources

The sequencing datasets were obtained from the NCBI Sequence Read Archive (SRA).

- [PRJNA1259947](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1259947)
- [PRJNA471972](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA471972)
- [PRJNA1019460](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1019460)

## 📌 Disclaimer

This repository is intended for research and educational purposes only. It is not a medical device, does not provide a diagnosis, and must not be used to make clinical decisions.

---

**Project focus:** Bioinformatics · Gut Microbiome · 16S rRNA Sequencing · Machine Learning · Explainable AI
