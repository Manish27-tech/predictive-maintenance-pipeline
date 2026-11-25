"""Command-line interface for training and artifact checks."""

from __future__ import annotations

import argparse
import json

from .config import ARTIFACTS, MANIFEST_PATH, MODEL_PATH, REPORTS_DIR
from .pipeline import run_pipeline


def train() -> int:
    result = run_pipeline()
    print(json.dumps(result, indent=2))
    return 0


def check() -> int:
    required = [
        MODEL_PATH,
        MANIFEST_PATH,
        ARTIFACTS / "metrics.json",
        ARTIFACTS / "failure_modes.csv",
        REPORTS_DIR / "REPORT.md",
        REPORTS_DIR / "MODEL_CARD.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("missing artifacts:")
        for path in missing:
            print(f" - {path}")
        return 1
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    test = metrics["test_metrics"]
    print(
        f"ok model={metrics['model']} roc_auc={test['roc_auc']} "
        f"pr_auc={test['pr_auc']} recall={test['recall']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="maintenance", description="Predictive-maintenance CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="run the full pipeline")
    sub.add_parser("check", help="verify required artifacts")
    args = parser.parse_args(argv)
    return train() if args.command == "train" else check()


if __name__ == "__main__":
    raise SystemExit(main())
