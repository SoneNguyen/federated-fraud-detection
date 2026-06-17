"""
LightGBM baseline for fraud detection.

Run this to see what AUPRC/AUROC is achievable with a simple tabular model
on the same feature set and data split. This helps validate whether the
feature engineering and data are adequate (target: ~0.70 AUPRC).

Usage:
    uv run python -m scripts.lightgbm_baseline

Output:
    - Prints AUPRC, AUROC, F1 on hold-out test set
    - Saves best model to outputs/lightgbm_baseline.pkl
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_data(client_id: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Load parquet data for a client."""
    path = Path(f"data/processed/client_{client_id}/transactions_normalized.parquet")
    df = pd.read_parquet(path)
    
    # Separate features and labels
    y = df["is_fraud"].values
    X = df.drop(columns=["is_fraud"])
    
    return X, y


def main():
    logger.info("Loading data from all 3 federated clients...")
    
    X_list, y_list = [], []
    for cid in range(3):
        X, y = load_data(cid)
        logger.info(f"  Client {cid}: {len(X):,} rows, fraud={y.mean()*100:.2f}%")
        X_list.append(X)
        y_list.append(y)
    
    # Combine clients and do train/val/test split
    X_all = pd.concat(X_list, ignore_index=True)
    y_all = np.concatenate(y_list)
    
    logger.info(f"\nTotal: {len(X_all):,} rows, fraud={y_all.mean()*100:.2f}%")
    logger.info(f"Features: {X_all.shape[1]}")
    
    # Split: train 60%, val 20%, test 20%
    n = len(X_all)
    train_idx = np.arange(0, int(0.6 * n))
    val_idx = np.arange(int(0.6 * n), int(0.8 * n))
    test_idx = np.arange(int(0.8 * n), n)
    
    X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
    X_val, y_val = X_all.iloc[val_idx], y_all[val_idx]
    X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]
    
    logger.info(f"\nTrain: {len(X_train):,} (fraud={y_train.mean()*100:.2f}%)")
    logger.info(f"Val:   {len(X_val):,} (fraud={y_val.mean()*100:.2f}%)")
    logger.info(f"Test:  {len(X_test):,} (fraud={y_test.mean()*100:.2f}%)")
    
    # Train LightGBM with hyperparameters tuned for fraud detection
    logger.info("\nTraining LightGBM...")
    
    # Calculate class weight for imbalanced data
    n_neg = sum(y_train == 0)
    n_pos = sum(y_train == 1)
    scale_pos = n_neg / max(n_pos, 1)
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    logger.info(f"Best iteration: {model.best_iteration_}")
    
    # Evaluate on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auprc = average_precision_score(y_test, y_pred_proba)
    auroc = roc_auc_score(y_test, y_pred_proba)
    
    # Find best F1 threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"LightGBM Baseline Results (Test Set)")
    logger.info(f"{'='*60}")
    logger.info(f"AUPRC:             {auprc:.4f}")
    logger.info(f"AUROC:             {auroc:.4f}")
    logger.info(f"Best F1:           {best_f1:.4f} (threshold={best_threshold:.3f})")
    logger.info(f"{'='*60}")
    
    # Save model
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    model.booster_.save_model(str(output_dir / "lightgbm_baseline.txt"))
    logger.info(f"Model saved to outputs/lightgbm_baseline.txt")
    
    # Save results
    results = {
        "model": "LightGBM",
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "auprc": float(auprc),
        "auroc": float(auroc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
    }
    
    with open(output_dir / "lightgbm_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to outputs/lightgbm_baseline_results.json")


if __name__ == "__main__":
    main()
