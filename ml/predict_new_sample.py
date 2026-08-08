import joblib
import numpy as np
from process_new_sample import process_new_sample
from align_features import align_features

def predict_new_sample(sample_id, fwd_fastq, rev_fastq, work_dir,
                        classifier_path, model_path, verbose=True):
    """
    End-to-end: raw FASTQ pair -> QIIME2 processing -> feature alignment ->
    RF prediction. Ties together process_new_sample() and align_features().

    Returns a dict (not just printed output) so this can be called from
    other code, e.g. a Flask endpoint:

    {
        'sample_id': str,
        'predicted_class': str,          # 'Cirrhosis' or 'Healthy'
        'confidence_pct': float,         # confidence of predicted class
        'class_probabilities': {         # both class probabilities
            'Cirrhosis': float,
            'Healthy': float
        },
        'n_genera_detected': int,        # genera found in new sample
        'n_genera_matched': int,         # genera matched to training columns
        'n_genera_dropped': int,         # genera unseen in training, dropped
        'feature_tsv_path': str,         # path to intermediate genus TSV
    }
    """
    # Step 1: raw FASTQ -> genus-level TSV via QIIME2
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing sample: {sample_id}")
        print(f"{'='*60}")

    tsv_path = process_new_sample(
        sample_id=sample_id,
        fwd_fastq=fwd_fastq,
        rev_fastq=rev_fastq,
        work_dir=work_dir,
        classifier_path=classifier_path
    )

    # Step 2: load model bundle
    bundle = joblib.load(model_path)
    model = bundle['model']
    le = bundle['label_encoder']
    training_columns = bundle['training_columns']

    # Step 3: align new sample's genus columns to training columns
    aligned = align_features(
        tsv_path=tsv_path,
        training_columns=training_columns,
        sample_id=sample_id
    )

    # Capture match/drop stats before they scroll off screen
    raw_genera = aligned.shape[1]  # always 429 post-alignment, not useful here
    # re-derive counts directly for the return dict
    import pandas as pd
    raw_tsv = pd.read_csv(tsv_path, sep='\t', skiprows=1, index_col=0).T
    n_detected = raw_tsv.shape[1]
    n_matched = len(set(raw_tsv.columns) & set(training_columns))
    n_dropped = len(set(raw_tsv.columns) - set(training_columns))

    # Step 4: predict
    X = aligned.to_numpy(dtype=np.float32)
    proba = model.predict_proba(X)[0]
    pred_encoded = model.predict(X)[0]
    pred_class = le.inverse_transform([pred_encoded])[0]

    classes = list(le.classes_)
    class_probs = {cls: round(float(proba[classes.index(cls)]) * 100, 1) for cls in classes}
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
        print(f"\n{'='*60}")
        print(f"PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"  Sample ID        : {result['sample_id']}")
        print(f"  Predicted Class  : {result['predicted_class']}")
        print(f"  Confidence       : {result['confidence_pct']}%")
        print(f"  Class Probabilities:")
        for cls, pct in class_probs.items():
            print(f"    {cls:12s}: {pct}%")
        print(f"  Genera detected  : {result['n_genera_detected']}")
        print(f"  Genera matched   : {result['n_genera_matched']} (used in prediction)")
        print(f"  Genera dropped   : {result['n_genera_dropped']} (unseen in training)")
        print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    result = predict_new_sample(
        sample_id="TEST_SAMPLE",
        fwd_fastq="/home/siddharth/liver_cirrhosis/data/PRJNA471972/SRR7182199_1.fastq.gz",
        rev_fastq="/home/siddharth/liver_cirrhosis/data/PRJNA471972/SRR7182199_2.fastq.gz",
        work_dir="/home/siddharth/liver_cirrhosis/ml/new_sample_test2",
        classifier_path="/home/siddharth/liver_cirrhosis/ml/reference/silva-138-99-nb-classifier.qza",
        model_path="/home/siddharth/liver_cirrhosis/ml/pipeline_model.pkl"
    )
    print("Returned dict:")
    print(result)
