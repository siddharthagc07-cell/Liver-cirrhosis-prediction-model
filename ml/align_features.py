import pandas as pd
import numpy as np

def align_features(tsv_path, training_columns, sample_id=None):
    """
    Takes a new sample's exported genus-level feature-table.tsv (raw counts)
    and reindexes it against the model's saved training_columns (429 genera,
    raw taxonomy strings, exact match to pipeline_model.pkl).

    Steps:
      1. Load new sample TSV (genera as rows, samples as columns)
      2. Transpose to samples x genera (matches training orientation)
      3. Reindex columns to training_columns exactly:
           - genera present in new sample but not in training set -> dropped
           - genera in training set but absent in new sample -> filled with 0
      4. Re-normalize via TSS (row sums to 1) AFTER alignment —
         must happen after, not before, since dropping columns changes the
         total abundance denominator
      5. Return a single-row DataFrame in the exact column order the model expects

    Returns: pandas DataFrame, shape (1, len(training_columns)), float32
    """
    # Load raw genus-level counts (skiprows=1 skips the "# Constructed from biom file" line)
    raw = pd.read_csv(tsv_path, sep='\t', skiprows=1, index_col=0)

    # raw is genera x samples -> transpose to samples x genera
    raw = raw.T

    if sample_id is not None:
        if sample_id not in raw.index:
            raise ValueError(f"Sample ID '{sample_id}' not found in {tsv_path}. "
                              f"Found: {list(raw.index)}")
        raw = raw.loc[[sample_id]]
    elif raw.shape[0] != 1:
        raise ValueError(f"Expected exactly 1 sample in {tsv_path}, found {raw.shape[0]}. "
                          f"Pass sample_id explicitly to select one.")

    n_present = len(set(raw.columns) & set(training_columns))
    n_dropped = len(set(raw.columns) - set(training_columns))
    n_missing = len(set(training_columns) - set(raw.columns))
    print(f"[align_features] New sample genera: {raw.shape[1]}")
    print(f"[align_features]   Matched to training columns: {n_present}")
    print(f"[align_features]   Dropped (unseen in training):  {n_dropped}")
    print(f"[align_features]   Filled as 0 (missing from sample): {n_missing}")

    # Reindex: this does the drop + fill-with-0 in one step
    aligned = raw.reindex(columns=training_columns, fill_value=0)

    # Re-normalize (TSS) AFTER alignment — total abundance changes once
    # unseen genera are dropped, so normalizing before would be wrong
    row_sum = aligned.sum(axis=1).iloc[0]
    if row_sum == 0:
        raise ValueError("Sample has zero total abundance after alignment — "
                          "check that genera overlap with training set at all.")
    aligned_norm = aligned.div(aligned.sum(axis=1), axis=0).astype(np.float32)

    return aligned_norm


if __name__ == "__main__":
    import joblib
    bundle = joblib.load('/home/siddharth/liver_cirrhosis/ml/pipeline_model.pkl')
    training_columns = bundle['training_columns']

    result = align_features(
        tsv_path='/home/siddharth/liver_cirrhosis/ml/new_sample_test/exported/feature-table.tsv',
        training_columns=training_columns,
        sample_id='TEST_SAMPLE'
    )
    print(f"\nAligned feature vector shape: {result.shape}")
    print(f"Row sum (should be ~1.0): {result.sum(axis=1).values[0]:.6f}")
    print(f"Non-zero genera in aligned vector: {(result.iloc[0] > 0).sum()}")
