"""Print compact target progress from the latest training metrics."""

from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    results_dir = Path("results")
    latest = _load(results_dir / "latest_metrics.json")
    best = _load(results_dir / "best_round.json")
    history = _load(results_dir / "evaluation_history.json")

    if latest is None:
        print("No metrics yet. Start the server and wait for the first evaluation round.")
        return

    print("Latest round")
    for key in (
        "round",
        "learning_state",
        "learning_score",
        "high_band_score",
        "val_auprc",
        "min_client_auprc",
        "val_auroc",
        "min_client_auroc",
        "val_f1",
        "min_client_f1",
        "val_loss",
        "max_client_loss",
        "val_focal_loss",
        "val_bce_loss",
        "val_hybrid_loss",
        "val_threshold",
        "fit_train_loss",
        "fit_train_loss_delta",
        "fit_grad_norm_mean",
        "loss_slope_5",
        "f1_slope_5",
        "auprc_slope_5",
        "best_val_loss",
        "rounds_since_best_loss",
        "target_score",
        "target_met",
        "high_target_met",
        "client_floor_met",
        "gap_auprc",
        "gap_auroc",
        "gap_f1",
        "gap_high_auprc",
        "gap_high_auroc",
        "gap_high_f1",
        "gap_client_auprc",
        "gap_client_auroc",
        "gap_client_f1",
    ):
        if key in latest:
            print(f"  {key:14s}: {_fmt(latest[key])}")

    if best:
        print("\nBest target round")
        for key in (
            "round",
            "checkpoint",
            "learning_score",
            "high_band_score",
            "val_auprc",
            "min_client_auprc",
            "val_auroc",
            "min_client_auroc",
            "val_f1",
            "min_client_f1",
            "val_loss",
            "target_score",
            "target_met",
            "high_target_met",
            "client_floor_met",
        ):
            if key in best:
                print(f"  {key:14s}: {_fmt(best[key])}")

    if isinstance(history, list):
        print(f"\nEvaluated rounds: {len(history)}")
        recent = history[-5:]
        if recent:
            print("\nRecent pattern")
            print("  rnd  state       loss     d_loss   auprc/min  f1/min    train_d")
            prev_loss = None
            for row in recent:
                loss = row.get("val_loss")
                auprc = row.get("val_auprc", 0.0)
                min_auprc = row.get("min_client_auprc", auprc)
                f1 = row.get("val_f1", 0.0)
                min_f1 = row.get("min_client_f1", f1)
                d_loss = (
                    float(loss) - float(prev_loss)
                    if isinstance(loss, (int, float))
                    and isinstance(prev_loss, (int, float))
                    else 0.0
                )
                prev_loss = loss
                print(
                    "  "
                    f"{int(row.get('round', 0)):>3d}  "
                    f"{str(row.get('learning_state', 'n/a')):<10s} "
                    f"{_fmt(row.get('val_loss', 0.0)):>7s} "
                    f"{d_loss:>8.4f} "
                    f"{_fmt(auprc):>5s}/{_fmt(min_auprc):>5s} "
                    f"{_fmt(f1):>5s}/{_fmt(min_f1):>5s} "
                    f"{_fmt(row.get('fit_train_loss_delta', 0.0)):>8s}"
                )


if __name__ == "__main__":
    main()
