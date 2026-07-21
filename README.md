# Predicting Liver Cirrhosis from Gut Microbiome Data using 16S rRNA Sequencing and Machine Learning

A computational bioinformatics pipeline that predicts liver cirrhosis risk from gut microbiome 16S rRNA sequencing data, using machine learning.

## Overview

This project processes raw 16S rRNA FASTQ files through a QIIME2/DADA2 pipeline, generates a genus-level microbial feature table, and trains machine learning models to classify samples as Healthy or Cirrhosis. Model predictions are explained using SHAP.

This is a computational project built entirely on publicly available sequencing data — no wet-lab work was involved.

## Data

Three independent public datasets were used, all from NCBI SRA:

| Dataset | Sample Type | Sequencing |
|---|---|---|
| PRJNA471972 | Stool | Paired-end (V3–V4) |
| PRJNA1259947 | Stool | Paired-end (V3–V4) |
| PRJNA1019460 | Stool | Single-end (V3–V4) |

Combined: 195 samples, 429 bacterial genera.

## Pipeline

1. Raw FASTQ import into QIIME2
2. Quality control
3. Denoising (DADA2 for paired-end, Deblur for single-end)
4. Taxonomic classification (Silva 138)
5. Genus-level feature table generation
6. TSS normalization
7. Model training (Random Forest, XGBoost, SVM, Logistic Regression)
8. Hyperparameter tuning (GridSearchCV)
9. Evaluation (Stratified 5-Fold Cross-Validation)
10. SHAP explainability
11. Automated prediction report generation

## Results

| Model | Accuracy | AUROC | F1-Score |
|---|---|---|---|
| **Random Forest** | **73.3%** | **0.802** | **0.681** |
| XGBoost | 69.2% | 0.770 | 0.673 |
| SVM | 69.7% | 0.753 | 0.609 |
| Logistic Regression | 69.2% | 0.708 | 0.602 |

Random Forest was selected as the final model.

## Technologies Used

### Operating System
- Linux (Ubuntu)

### Bioinformatics
- QIIME2
- DADA2
- Deblur
- SILVA 138

### Programming & Data Analysis
- Python
- pandas
- NumPy

### Machine Learning
- scikit-learn
- XGBoost

### Model Explainability
- SHAP

---

## Current Limitations

- Limited sample size and class imbalance.
- Publicly available 16S rRNA datasets from multiple studies may introduce batch effects.
- 16S rRNA sequencing provides genus-level information with limited species-level and functional resolution.

---

## Future Scope

- Validate the model using larger and more diverse cohorts.
- Incorporate shotgun metagenomic sequencing for improved taxonomic and functional resolution.
- Improve model performance through advanced feature selection and external validation.


## Author

Siddharth Shivangi — M.Sc. Microbiology, MIT-WPU Pune
Bioinformatics Internship, GeneSpectrum Life Sciences LLP
