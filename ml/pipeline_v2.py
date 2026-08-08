#!/usr/bin/env python3
"""
pipeline_v2.py — Raw FASTQ -> Liver Cirrhosis Prediction

Fixes the core gap in the original pipeline.py: that script only accepted
sample IDs already present in feature_table.tsv (a lookup, not real
processing). This version actually runs new raw FASTQ through QIIME2 —
primer trimming (cutadapt), DADA2 denoising, Silva 138 taxonomy
classification, genus-level collapse — then aligns the result to the
model's 429 trained genus columns and predicts.

Usage:
    python3 pipeline_v2.py <sample_id> <forward.fastq.gz> <reverse.fastq.gz> <protocol>

    <protocol> must be one of:
      PRJNA471972    trunc-f 260, trunc-r 200, primers CCTACGGGNGGCWGCAG / GACTACHVGGGTATCTAATCC
      PRJNA1259947   trunc-f 230, trunc-r 200, primers TACGGRAGGCAGCAG / AGGGTATCTAATCCT

    Or pass explicit params:
    python3 pipeline_v2.py <sample_id> <fwd> <rev> custom <front_f> <front_r> <trunc_f> <trunc_r>

Scope: target samples must come from one of the two known protocols above,
or an explicitly specified matching protocol. Using the wrong primer/trunc
combination will cause DADA2 to filter out nearly all reads at the merge
step (forward/reverse reads won't overlap correctly). Different sequencing
setups entirely (single-end, other regions) are out of scope for this version.
"""
import sys
import os
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE = '/home/siddharth/liver_cirrhosis'
CLASSIFIER_PATH = f'{BASE}/ml/reference/silva-138-99-nb-classifier.qza'
MODEL_PATH = f'{BASE}/ml/pipeline_model.pkl'
WORK_DIR_ROOT = f'{BASE}/ml/predictions'

# Known protocols: (front_f, front_r, trunc_len_f, trunc_len_r)
PROTOCOLS = {
    'PRJNA471972': ('CCTACGGGNGGCWGCAG', 'GACTACHVGGGTATCTAATCC', 260, 200),
    'PRJNA1259947': ('TACGGRAGGCAGCAG', 'AGGGTATCTAATCCT', 230, 200),
}

# Single-end Deblur protocols - no primer trimming, single trim_length param.
# Kept SEPARATE from PROTOCOLS since paired-end and single-end processing
# functions take structurally different arguments.
SINGLE_END_PROTOCOLS = {
    'PRJNA1019460': 250,
}


def run_cmd(cmd, description):
    """Run a shell command, print status, raise on failure."""
    print(f"\n[RUNNING] {description}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"FAILED: {description}")
    print(f"[DONE] {description}")
    return result


def process_new_sample(sample_id, fwd_fastq, rev_fastq, work_dir, classifier_path,
                        front_f, front_r, trunc_len_f, trunc_len_r):
    """Raw paired-end FASTQ -> genus-level feature TSV via QIIME2.
    Includes cutadapt primer trimming before DADA2 (required — original
    training data was primer-trimmed, so new samples must match)."""
    os.makedirs(work_dir, exist_ok=True)

    manifest_path = os.path.join(work_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("sample-id\tforward-absolute-filepath\treverse-absolute-filepath\n")
        f.write(f"{sample_id}\t{os.path.abspath(fwd_fastq)}\t{os.path.abspath(rev_fastq)}\n")

    demux_qza = os.path.join(work_dir, "demux.qza")
    trimmed_qza = os.path.join(work_dir, "trimmed.qza")
    table_qza = os.path.join(work_dir, "table.qza")
    rep_seqs_qza = os.path.join(work_dir, "rep-seqs.qza")
    denoise_stats_qza = os.path.join(work_dir, "denoising-stats.qza")
    taxonomy_qza = os.path.join(work_dir, "taxonomy.qza")
    taxa_collapsed_qza = os.path.join(work_dir, "genus-table.qza")
    export_dir = os.path.join(work_dir, "exported")

    run_cmd([
        "qiime", "tools", "import",
        "--type", "SampleData[PairedEndSequencesWithQuality]",
        "--input-path", manifest_path,
        "--output-path", demux_qza,
        "--input-format", "PairedEndFastqManifestPhred33V2"
    ], "Import raw FASTQ")

    run_cmd([
        "qiime", "cutadapt", "trim-paired",
        "--i-demultiplexed-sequences", demux_qza,
        "--p-front-f", front_f,
        "--p-front-r", front_r,
        "--p-error-rate", "0.1",
        "--p-overlap", "3",
        "--p-discard-untrimmed",
        "--o-trimmed-sequences", trimmed_qza,
        "--verbose"
    ], f"Cutadapt primer trim (front-f {front_f}, front-r {front_r})")

    run_cmd([
        "qiime", "dada2", "denoise-paired",
        "--i-demultiplexed-seqs", trimmed_qza,
        "--p-trunc-len-f", str(trunc_len_f), "--p-trunc-len-r", str(trunc_len_r),
        "--p-trim-left-f", "0", "--p-trim-left-r", "0",
        "--p-max-ee-f", "2", "--p-max-ee-r", "5",
        "--p-trunc-q", "2", "--p-min-overlap", "12",
        "--p-pooling-method", "independent",
        "--p-chimera-method", "consensus",
        "--p-min-fold-parent-over-abundance", "1.0",
        "--p-n-threads", "0",
        "--o-table", table_qza,
        "--o-representative-sequences", rep_seqs_qza,
        "--o-denoising-stats", denoise_stats_qza,
        "--verbose"
    ], f"DADA2 denoise (trunc-f {trunc_len_f}, trunc-r {trunc_len_r})")

    run_cmd([
        "qiime", "feature-classifier", "classify-sklearn",
        "--i-classifier", classifier_path,
        "--i-reads", rep_seqs_qza,
        "--p-confidence", "0.7", "--p-n-jobs", "4",
        "--o-classification", taxonomy_qza
    ], "Silva taxonomy classification")

    run_cmd([
        "qiime", "taxa", "collapse",
        "--i-table", table_qza,
        "--i-taxonomy", taxonomy_qza,
        "--p-level", "6",
        "--o-collapsed-table", taxa_collapsed_qza
    ], "Collapse to genus level")

    run_cmd([
        "qiime", "tools", "export",
        "--input-path", taxa_collapsed_qza,
        "--output-path", export_dir
    ], "Export genus table")

    biom_path = os.path.join(export_dir, "feature-table.biom")
    tsv_path = os.path.join(export_dir, "feature-table.tsv")
    run_cmd(["biom", "convert", "-i", biom_path, "-o", tsv_path, "--to-tsv"],
            "Convert biom to TSV")

    return tsv_path


def process_new_sample_single_end(sample_id, fastq, work_dir, classifier_path, trim_length):
    """Raw single-end FASTQ -> genus-level feature TSV via QIIME2.
    NO cutadapt step — confirmed via provenance that PRJNA1019460's
    original processing went straight from import to Deblur (reads were
    already primer-trimmed by the original submitters before deposit).
    Uses Deblur, not DADA2, matching the original single-end protocol."""
    os.makedirs(work_dir, exist_ok=True)
    manifest_path = os.path.join(work_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("sample-id\tabsolute-filepath\n")
        f.write(f"{sample_id}\t{os.path.abspath(fastq)}\n")
    demux_qza = os.path.join(work_dir, "demux.qza")
    table_qza = os.path.join(work_dir, "table.qza")
    rep_seqs_qza = os.path.join(work_dir, "rep-seqs.qza")
    deblur_stats_qza = os.path.join(work_dir, "deblur-stats.qza")
    taxonomy_qza = os.path.join(work_dir, "taxonomy.qza")
    taxa_collapsed_qza = os.path.join(work_dir, "genus-table.qza")
    export_dir = os.path.join(work_dir, "exported")

    run_cmd([
        "qiime", "tools", "import",
        "--type", "SampleData[SequencesWithQuality]",
        "--input-path", manifest_path,
        "--output-path", demux_qza,
        "--input-format", "SingleEndFastqManifestPhred33V2"
    ], "Import raw single-end FASTQ")

    # NO quality-filter step — confirmed via provenance that Deblur's
    # input UUID matched the raw import artifact directly, with nothing
    # in between. This is unusual for a standard QIIME2 Deblur workflow
    # (quality-filter q-score is normally a required precursor), but it's
    # what the original run actually did, so we match it exactly here
    # rather than "fixing" it — matching genus columns matters more than
    # following the textbook-standard workflow.
    run_cmd([
        "qiime", "deblur", "denoise-16S",
        "--i-demultiplexed-seqs", demux_qza,
        "--p-trim-length", str(trim_length),
        "--p-sample-stats",
        "--o-table", table_qza,
        "--o-representative-sequences", rep_seqs_qza,
        "--o-stats", deblur_stats_qza
    ], f"Deblur denoise (trim-length {trim_length})")

    run_cmd([
        "qiime", "feature-classifier", "classify-sklearn",
        "--i-classifier", classifier_path,
        "--i-reads", rep_seqs_qza,
        "--p-confidence", "0.7", "--p-n-jobs", "4",
        "--o-classification", taxonomy_qza
    ], "Silva taxonomy classification")

    run_cmd([
        "qiime", "taxa", "collapse",
        "--i-table", table_qza,
        "--i-taxonomy", taxonomy_qza,
        "--p-level", "6",
        "--o-collapsed-table", taxa_collapsed_qza
    ], "Collapse to genus level")

    run_cmd([
        "qiime", "tools", "export",
        "--input-path", taxa_collapsed_qza,
        "--output-path", export_dir
    ], "Export genus table")

    biom_path = os.path.join(export_dir, "feature-table.biom")
    tsv_path = os.path.join(export_dir, "feature-table.tsv")
    run_cmd(["biom", "convert", "-i", biom_path, "-o", tsv_path, "--to-tsv"],
            "Convert biom to TSV")
    return tsv_path


def align_features(tsv_path, training_columns, sample_id):
    """Reindex new sample's genus columns against the model's 429 training columns."""
    raw = pd.read_csv(tsv_path, sep='\t', skiprows=1, index_col=0).T

    if sample_id not in raw.index:
        raise ValueError(f"Sample ID '{sample_id}' not found in {tsv_path}")
    raw = raw.loc[[sample_id]]

    n_detected = raw.shape[1]
    n_matched = len(set(raw.columns) & set(training_columns))
    n_dropped = len(set(raw.columns) - set(training_columns))

    aligned = raw.reindex(columns=training_columns, fill_value=0)
    row_sum = aligned.sum(axis=1).iloc[0]
    if row_sum == 0:
        raise ValueError("Zero total abundance after alignment — no genus overlap with training set.")

    aligned_norm = aligned.div(aligned.sum(axis=1), axis=0).astype(np.float32)
    return aligned_norm, n_detected, n_matched, n_dropped


def predict_new_sample(sample_id, fwd_fastq, rev_fastq, front_f, front_r,
                        trunc_len_f, trunc_len_r, work_dir=None, verbose=True):
    """
    End-to-end: raw FASTQ pair -> QIIME2 -> alignment -> RF prediction.
    Returns a dict, reusable from other code (e.g. a Flask endpoint).
    """
    if work_dir is None:
        work_dir = os.path.join(WORK_DIR_ROOT, sample_id)

    if verbose:
        print(f"\n{'='*60}\nProcessing sample: {sample_id}\n{'='*60}")

    tsv_path = process_new_sample(sample_id, fwd_fastq, rev_fastq, work_dir, CLASSIFIER_PATH,
                                   front_f, front_r, trunc_len_f, trunc_len_r)

    bundle = joblib.load(MODEL_PATH)
    model, le, training_columns = bundle['model'], bundle['label_encoder'], bundle['training_columns']

    aligned, n_detected, n_matched, n_dropped = align_features(tsv_path, training_columns, sample_id)

    X = aligned.to_numpy(dtype=np.float32)
    proba = model.predict_proba(X)[0]
    pred_class = le.inverse_transform([model.predict(X)[0]])[0]
    classes = list(le.classes_)
    class_probs = {c: round(float(proba[classes.index(c)]) * 100, 1) for c in classes}
    confidence = max(class_probs.values())

    result = {
        'sample_id': sample_id,
        'predicted_class': pred_class,
        'confidence_pct': confidence,
        'class_probabilities': class_probs,
        'n_genera_detected': n_detected,
        'n_genera_matched': n_matched,
        'n_genera_dropped': n_dropped,
        'feature_tsv_path': tsv_path,
    }

    if verbose:
        print(f"\n{'='*60}\nPREDICTION RESULT\n{'='*60}")
        print(f"  Sample ID        : {result['sample_id']}")
        print(f"  Predicted Class  : {result['predicted_class']}")
        print(f"  Confidence       : {result['confidence_pct']}%")
        for cls, pct in class_probs.items():
            print(f"    {cls:12s}: {pct}%")
        print(f"  Genera detected  : {n_detected}")
        print(f"  Genera matched   : {n_matched} (used in prediction)")
        print(f"  Genera dropped   : {n_dropped} (unseen in training)")
        print(f"{'='*60}\n")

    return result


def predict_new_sample_single_end(sample_id, fastq, trim_length, work_dir=None, verbose=True):
    """
    End-to-end single-end version: raw FASTQ -> QIIME2 (Deblur) -> alignment -> RF prediction.
    Mirrors predict_new_sample() but calls process_new_sample_single_end().
    """
    if work_dir is None:
        work_dir = os.path.join(WORK_DIR_ROOT, sample_id)

    if verbose:
        print(f"\n{'='*60}\nProcessing sample (single-end): {sample_id}\n{'='*60}")

    tsv_path = process_new_sample_single_end(sample_id, fastq, work_dir, CLASSIFIER_PATH, trim_length)

    bundle = joblib.load(MODEL_PATH)
    model, le, training_columns = bundle['model'], bundle['label_encoder'], bundle['training_columns']

    aligned, n_detected, n_matched, n_dropped = align_features(tsv_path, training_columns, sample_id)

    X = aligned.to_numpy(dtype=np.float32)
    proba = model.predict_proba(X)[0]
    pred_class = le.inverse_transform([model.predict(X)[0]])[0]
    classes = list(le.classes_)
    class_probs = {c: round(float(proba[classes.index(c)]) * 100, 1) for c in classes}
    confidence = max(class_probs.values())

    result = {
        'sample_id': sample_id,
        'predicted_class': pred_class,
        'confidence_pct': confidence,
        'class_probabilities': class_probs,
        'n_genera_detected': n_detected,
        'n_genera_matched': n_matched,
        'n_genera_dropped': n_dropped,
        'feature_tsv_path': tsv_path,
    }

    if verbose:
        print(f"\n{'='*60}\nPREDICTION RESULT\n{'='*60}")
        print(f"  Sample ID        : {result['sample_id']}")
        print(f"  Predicted Class  : {result['predicted_class']}")
        print(f"  Confidence       : {result['confidence_pct']}%")
        for cls, pct in class_probs.items():
            print(f"    {cls:12s}: {pct}%")
        print(f"  Genera detected  : {n_detected}")
        print(f"  Genera matched   : {n_matched} (used in prediction)")
        print(f"  Genera dropped   : {n_dropped} (unseen in training)")
        print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[3] in SINGLE_END_PROTOCOLS:
        sample_id, fastq, protocol = sys.argv[1], sys.argv[2], sys.argv[3]
        trim_length = SINGLE_END_PROTOCOLS[protocol]
        result = predict_new_sample_single_end(sample_id, fastq, trim_length)
        sys.exit(0)
    elif len(sys.argv) == 5 and sys.argv[4] in PROTOCOLS:
        sample_id, fwd, rev, protocol = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
        front_f, front_r, trunc_f, trunc_r = PROTOCOLS[protocol]
    elif len(sys.argv) == 9 and sys.argv[4] == 'custom':
        sample_id, fwd, rev = sys.argv[1], sys.argv[2], sys.argv[3]
        front_f, front_r = sys.argv[5], sys.argv[6]
        trunc_f, trunc_r = int(sys.argv[7]), int(sys.argv[8])
    else:
        print("Usage:")
        print("  python3 pipeline_v2.py <sample_id> <fastq.gz> <single_end_protocol>")
        print(f"    <single_end_protocol> one of: {list(SINGLE_END_PROTOCOLS.keys())}")
        print("  OR")
        print("  python3 pipeline_v2.py <sample_id> <fwd.fastq.gz> <rev.fastq.gz> <protocol>")
        print(f"    <protocol> one of: {list(PROTOCOLS.keys())}")
        print("  OR")
        print("  python3 pipeline_v2.py <sample_id> <fwd.fastq.gz> <rev.fastq.gz> custom <front_f> <front_r> <trunc_f> <trunc_r>")
        sys.exit(1)

    for f in (fwd, rev):
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            sys.exit(1)

    start = time.time()
    result = predict_new_sample(sample_id, fwd, rev, front_f, front_r, trunc_f, trunc_r)
    elapsed = time.time() - start
    print(f"Total runtime: {elapsed/60:.1f} minutes")
