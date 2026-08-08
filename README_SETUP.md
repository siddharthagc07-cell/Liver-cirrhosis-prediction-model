# Setup Instructions — Liver Cirrhosis Prediction Pipeline

## 1. Environment

```bash
conda env create -f environment.yml
conda activate lc_qiime
```

## 2. Required large files (not in this repo)

| File | Size | Purpose | Get it from |
|------|------|---------|--------------|
| `ml/pipeline_model.pkl` | few MB | Trained RF model + label encoder + training columns | https://github.com/siddharthagc07-cell/Liver-cirrhosis-prediction-model/releases/download/v1.0-model-assets/pipeline_model.pkl |
| `ml/reference/silva-138-99-nb-classifier.qza` | 209 MB | Silva 138 taxonomy classifier | https://data.qiime2.org/classifiers/sklearn-1.4.2/silva/silva-138-99-nb-classifier.qza |

Place them at those exact paths relative to the repo root.

## 3. Running a prediction

```bash
cd ml
python3 pipeline_v2.py <sample_id> <forward.fastq.gz> <reverse.fastq.gz> <protocol>
```

`<protocol>` must be one of:
- `PRJNA471972`  — primers CCTACGGGNGGCWGCAG / GACTACHVGGGTATCTAATCC, trunc 260/200
- `PRJNA1259947` — primers TACGGRAGGCAGCAG / AGGGTATCTAATCCT, trunc 230/200

For a different V3-V4 protocol, use custom mode:
```bash
python3 pipeline_v2.py <sample_id> <fwd> <rev> custom <front_f> <front_r> <trunc_f> <trunc_r>
```

Expect **~2-8 minutes** per sample.

## 4. Important: primer trimming is required

New samples must be primer-trimmed with cutadapt (built into the pipeline)
using primers matching the training protocol. Wrong primers/truncation
lengths cause DADA2 to filter out nearly all reads at the merge step —
this was found and fixed during development.

## 5. What this fixes vs. the original `pipeline.py`

The original script only accepted sample IDs already present in the
pre-built `feature_table.tsv` — it looked up existing rows rather than
processing raw FASTQ. `pipeline_v2.py` runs new samples through the full
QIIME2 pipeline: import → cutadapt primer trim → DADA2 → Silva
classification → genus collapse → feature alignment → prediction.
