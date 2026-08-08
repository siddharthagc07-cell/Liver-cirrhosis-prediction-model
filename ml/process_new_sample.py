import subprocess
import os

def run_cmd(cmd, description):
    """Run a shell command, print status, raise on failure."""
    print(f"\n[RUNNING] {description}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"FAILED: {description}")
    print(f"[DONE] {description}")
    return result

def process_new_sample(sample_id, fwd_fastq, rev_fastq, work_dir, classifier_path):
    """
    Takes one new paired-end raw FASTQ sample through the QIIME2 pipeline:
    import -> DADA2 denoise -> classify-sklearn -> taxa collapse (genus) -> export TSV.

    Uses the exact DADA2 parameters from the original PRJNA471972/PRJNA1259947
    processing (trunc-len-f 260, trunc-len-r 200, no left-trim).

    Returns the path to the exported genus-level feature TSV.
    """
    os.makedirs(work_dir, exist_ok=True)

    # Step 1: write manifest file (QIIME2 V2 manifest format = TAB-separated)
    manifest_path = os.path.join(work_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("sample-id\tforward-absolute-filepath\treverse-absolute-filepath\n")
        f.write(f"{sample_id}\t{os.path.abspath(fwd_fastq)}\t{os.path.abspath(rev_fastq)}\n")

    demux_qza = os.path.join(work_dir, "demux.qza")
    table_qza = os.path.join(work_dir, "table.qza")
    rep_seqs_qza = os.path.join(work_dir, "rep-seqs.qza")
    denoise_stats_qza = os.path.join(work_dir, "denoising-stats.qza")
    taxonomy_qza = os.path.join(work_dir, "taxonomy.qza")
    taxa_collapsed_qza = os.path.join(work_dir, "genus-table.qza")
    export_dir = os.path.join(work_dir, "exported")

    # Step 2: import as paired-end demultiplexed
    run_cmd([
        "qiime", "tools", "import",
        "--type", "SampleData[PairedEndSequencesWithQuality]",
        "--input-path", manifest_path,
        "--output-path", demux_qza,
        "--input-format", "PairedEndFastqManifestPhred33V2"
    ], "Import raw FASTQ as QIIME2 artifact")

    # Step 3: DADA2 denoise (exact params from original PRJNA471972 run)
    run_cmd([
        "qiime", "dada2", "denoise-paired",
        "--i-demultiplexed-seqs", demux_qza,
        "--p-trunc-len-f", "260",
        "--p-trunc-len-r", "200",
        "--p-trim-left-f", "0",
        "--p-trim-left-r", "0",
        "--p-max-ee-f", "2",
        "--p-max-ee-r", "5",
        "--p-trunc-q", "2",
        "--p-min-overlap", "12",
        "--p-pooling-method", "independent",
        "--p-chimera-method", "consensus",
        "--p-min-fold-parent-over-abundance", "1.0",
        "--p-n-threads", "0",
        "--o-table", table_qza,
        "--o-representative-sequences", rep_seqs_qza,
        "--o-denoising-stats", denoise_stats_qza,
        "--verbose"
    ], "DADA2 denoise (single sample)")

    # Step 4: taxonomy classification using full-length Silva 138 classifier
    run_cmd([
        "qiime", "feature-classifier", "classify-sklearn",
        "--i-classifier", classifier_path,
        "--i-reads", rep_seqs_qza,
        "--p-confidence", "0.7",
        "--p-n-jobs", "4",
        "--o-classification", taxonomy_qza
    ], "Taxonomy classification (Silva 138)")

    # Step 5: collapse to genus level (level 6, matches original pipeline)
    run_cmd([
        "qiime", "taxa", "collapse",
        "--i-table", table_qza,
        "--i-taxonomy", taxonomy_qza,
        "--p-level", "6",
        "--o-collapsed-table", taxa_collapsed_qza
    ], "Collapse feature table to genus level")

    # Step 6: export to TSV (biom -> tsv)
    run_cmd([
        "qiime", "tools", "export",
        "--input-path", taxa_collapsed_qza,
        "--output-path", export_dir
    ], "Export genus table")

    biom_path = os.path.join(export_dir, "feature-table.biom")
    tsv_path = os.path.join(export_dir, "feature-table.tsv")

    run_cmd([
        "biom", "convert",
        "-i", biom_path,
        "-o", tsv_path,
        "--to-tsv"
    ], "Convert biom to TSV")

    print(f"\n[SUCCESS] Genus-level feature table exported to: {tsv_path}")
    return tsv_path


if __name__ == "__main__":
    tsv_out = process_new_sample(
        sample_id="TEST_SAMPLE",
        fwd_fastq="/home/siddharth/liver_cirrhosis/data/PRJNA471972/SRR7182199_1.fastq.gz",
        rev_fastq="/home/siddharth/liver_cirrhosis/data/PRJNA471972/SRR7182199_2.fastq.gz",
        work_dir="/home/siddharth/liver_cirrhosis/ml/new_sample_test",
        classifier_path="/home/siddharth/liver_cirrhosis/ml/reference/silva-138-99-nb-classifier.qza"
    )
    print(f"Output TSV: {tsv_out}")
