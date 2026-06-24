"""Create a compact comparison table for 3/10/100-client experiments."""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path
from typing import Any


def _load_history(run: str, results_root: Path) -> list[dict[str, Any]]:
    path = results_root / run / "evaluation_history.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value) if isinstance(value, (int, float, bool)) else default


def _summarize(run: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {"run": run, "rounds": 0}
    ranked = sorted(
        history,
        key=lambda row: (
            bool(row.get("target_met", False)),
            _float(row, "learning_score"),
            _float(row, "val_auprc"),
            _float(row, "val_f1"),
        ),
        reverse=True,
    )
    latest = history[-1]
    best = ranked[0]
    recent = history[-10:] if len(history) >= 10 else history

    def recent_mean(key: str) -> float:
        values = [_float(row, key) for row in recent]
        return sum(values) / max(len(values), 1)

    def recent_std(key: str) -> float:
        values = [_float(row, key) for row in recent]
        return stats.pstdev(values) if len(values) > 1 else 0.0

    regressing = sum(1 for row in history if row.get("learning_state") == "regressing")
    target_rounds = sum(1 for row in history if bool(row.get("target_met", False)))
    return {
        "run": run,
        "rounds": len(history),
        "latest_round": int(_float(latest, "round")),
        "latest_auprc": _float(latest, "val_auprc"),
        "latest_auroc": _float(latest, "val_auroc"),
        "latest_f1": _float(latest, "val_f1"),
        "latest_loss": _float(latest, "val_loss"),
        "best_round": int(_float(best, "round")),
        "best_auprc": _float(best, "val_auprc"),
        "best_auroc": _float(best, "val_auroc"),
        "best_f1": _float(best, "val_f1"),
        "best_loss": _float(best, "val_loss"),
        "target_rounds": target_rounds,
        "regressing_rounds": regressing,
        "recent_auprc_mean": recent_mean("val_auprc"),
        "recent_auprc_std": recent_std("val_auprc"),
        "recent_f1_mean": recent_mean("val_f1"),
        "recent_f1_std": recent_std("val_f1"),
        "recent_loss_mean": recent_mean("val_loss"),
        "recent_loss_std": recent_std("val_loss"),
        "recent_min_client_auprc": recent_mean("min_client_auprc"),
        "recent_min_client_f1": recent_mean("min_client_f1"),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Scalability Comparison",
        "",
        "| Run | Rounds | Best round | Best AUPRC | Best AUROC | Best F1 | Latest AUPRC | Latest F1 | Target rounds | Regressing rounds | Last-10 AUPRC std | Last-10 F1 std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if int(row.get("rounds", 0)) == 0:
            lines.append(f"| {row['run']} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | n/a | n/a |")
            continue
        lines.append(
            "| {run} | {rounds} | {best_round} | {best_auprc:.4f} | {best_auroc:.4f} | "
            "{best_f1:.4f} | {latest_auprc:.4f} | {latest_f1:.4f} | {target_rounds} | "
            "{regressing_rounds} | {recent_auprc_std:.4f} | {recent_f1_std:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Read this table as both performance and stability. A scalable run should keep strong best metrics while reducing late-round variance and regressions.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 3/10/100-client training histories.")
    parser.add_argument("--runs", nargs="+", default=["3_clients", "10_clients", "100_clients"])
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-json", default="results/scalability_comparison.json")
    parser.add_argument("--output-md", default="results/scalability_comparison.md")
    args = parser.parse_args()

    root = Path(args.results_root)
    rows = [_summarize(run, _load_history(run, root)) for run in args.runs]
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(rows), encoding="utf-8")
    print(_markdown(rows))


if __name__ == "__main__":
    main()
