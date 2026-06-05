import pandas as pd

# drift/delayed_label_reconcile.py — run nightly
def reconcile_delayed_labels(predictions_log_path, ground_truth_path):
    preds = pd.read_parquet(predictions_log_path)
    truth = pd.read_parquet(ground_truth_path)
    merged = preds.merge(truth, on="transaction_id", how="inner")
    from sklearn.metrics import average_precision_score
    auprc = average_precision_score(merged["is_fraud"], merged["fraud_probability"])
    print(f"Retrospective AUPRC: {auprc:.4f}")
    return auprc