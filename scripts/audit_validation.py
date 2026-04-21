#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _ensure_proba_2d(pred: np.ndarray, n_classes: int) -> np.ndarray:
    arr = np.asarray(pred)
    if arr.ndim == 2:
        if arr.shape[1] == n_classes:
            return arr.astype(np.float32)
        if n_classes == 2 and arr.shape[1] == 1:
            p1 = arr.reshape(-1).astype(np.float32)
            return np.column_stack([1.0 - p1, p1]).astype(np.float32)
    if arr.ndim == 1:
        if n_classes == 2:
            p1 = arr.astype(np.float32)
            return np.column_stack([1.0 - p1, p1]).astype(np.float32)
        out = np.zeros((arr.shape[0], n_classes), dtype=np.float32)
        idx = np.clip(np.rint(arr).astype(np.int32), 0, max(0, n_classes - 1))
        out[np.arange(arr.shape[0]), idx] = 1.0
        return out
    raise ValueError(f"Unsupported prediction shape: {arr.shape}")


def _score(y_true: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y_true).astype(np.int32).reshape(-1)
    n_classes = int(np.max(y)) + 1 if y.size else 0
    if n_classes <= 2:
        arr = np.asarray(pred)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(roc_auc_score(y, arr[:, 1]))
        return float(roc_auc_score(y, arr.reshape(-1)))

    proba = _ensure_proba_2d(pred, n_classes=n_classes)
    y_hat = np.argmax(proba, axis=1).astype(np.int32)
    return float(balanced_accuracy_score(y, y_hat))


def _metric_name(y_true: np.ndarray) -> str:
    n_classes = int(np.max(np.asarray(y_true).astype(np.int32))) + 1
    return "balanced_accuracy" if n_classes > 2 else "roc_auc"


def _load_valid_targets(root: Path) -> tuple[np.ndarray, str]:
    valid_path = root / "data" / "processed" / "valid.parquet"
    mapping_path = root / "data" / "processed" / "target_mapping.json"
    if not valid_path.exists():
        raise FileNotFoundError(f"Missing: {valid_path}")

    valid_df = pd.read_parquet(valid_path)
    target_col = "Irrigation_Need" if "Irrigation_Need" in valid_df.columns else valid_df.columns[-1]
    if target_col not in valid_df.columns:
        raise ValueError("Could not infer target column from valid split.")

    if mapping_path.exists():
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        class_labels = [str(x) for x in payload.get("class_labels", [])]
        if class_labels:
            class_to_index = {c: i for i, c in enumerate(class_labels)}
            y = valid_df[target_col].astype("string").fillna("__NA__").map(class_to_index).fillna(0).astype(np.int32)
            return y.to_numpy(), target_col

    y = pd.factorize(valid_df[target_col].astype("string").fillna("__NA__"), sort=True)[0].astype(np.int32)
    return y, target_col


def _stage_rows(root: Path) -> list[dict[str, Any]]:
    base = root / "outputs" / "level2_results"
    level_dir = base / "tree_level2_dataset_irrigation_digit_decimal_v1"
    return [
        {
            "stage": "level3",
            "pred": level_dir / "level3_valid_pred.npy",
            "metrics": level_dir / "level3_final_metrics.json",
            "metric_key": "valid_auc",
        },
        {
            "stage": "level4",
            "pred": level_dir / "level4_valid_pred.npy",
            "metrics": level_dir / "level4_final_metrics.json",
            "metric_key": "valid_auc",
        },
        {
            "stage": "hill_climb",
            "pred": base / "hill_climb" / "hill_climb_valid_pred.npy",
            "metrics": base / "hill_climb" / "hill_climb_selection.json",
            "metric_key": "final_valid_auc",
        },
        {
            "stage": "stacking",
            "pred": base / "stacking" / "stacked_valid_pred.npy",
            "metrics": base / "stacking" / "stacking_metrics.json",
            "metric_key": "valid_auc",
        },
        {
            "stage": "final_blend",
            "pred": base / "final_blend" / "final_blend_valid_pred.npy",
            "metrics": base / "final_blend" / "final_blend_summary.json",
            "metric_key": "best_valid_auc",
        },
    ]


def _bootstrap_ci(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    random_state: int = 2026,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = y_true.shape[0]
    stats = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        stats[i] = _score(y_true[idx], np.asarray(pred)[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local validation metrics against saved predictions.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text table.")
    args = parser.parse_args()

    root = args.root.resolve()
    y_valid, target_col = _load_valid_targets(root)
    metric_name = _metric_name(y_valid)
    rows: list[dict[str, Any]] = []
    final_warnings: list[str] = []

    for stage_row in _stage_rows(root):
        pred_path = stage_row["pred"]
        metrics_path = stage_row["metrics"]
        metric_key = stage_row["metric_key"]
        out: dict[str, Any] = {
            "stage": stage_row["stage"],
            "pred_path": str(pred_path),
            "metrics_path": str(metrics_path),
            "exists": bool(pred_path.exists() and metrics_path.exists()),
        }
        if not out["exists"]:
            rows.append(out)
            continue

        pred = np.load(pred_path)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        reported = payload.get(metric_key)
        calculated = _score(y_valid, pred)
        out.update(
            {
                "metric_key": metric_key,
                "reported": float(reported),
                "calculated": float(calculated),
                "abs_diff": float(abs(float(reported) - float(calculated))),
            }
        )
        rows.append(out)

        if stage_row["stage"] == "final_blend":
            selected = payload.get("selected_blend", [])
            selected_names = [str(item.get("name", "")) for item in selected if isinstance(item, dict)]
            if any(name == "previous_final_blend" for name in selected_names):
                final_warnings.append(
                    "final_blend used 'previous_final_blend' as an input candidate; this can inflate local score on reruns."
                )
            if any(name.startswith("history::") for name in selected_names):
                final_warnings.append(
                    "final_blend used history::* candidates; verify they match the same validation split."
                )
            lo, hi = _bootstrap_ci(y_valid, pred, n_bootstrap=max(200, int(args.bootstrap)))
            out["bootstrap_95ci"] = [float(lo), float(hi)]

    payload = {
        "root": str(root),
        "target_col": target_col,
        "metric_used": metric_name,
        "rows": rows,
        "warnings": final_warnings,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"root={payload['root']}")
    print(f"target_col={target_col}")
    print(f"metric_used={metric_name}")
    print("stage | reported | calculated | abs_diff")
    for row in rows:
        if not row.get("exists"):
            print(f"{row['stage']} | missing | missing | missing")
            continue
        print(
            f"{row['stage']} | {row['reported']:.12f} | "
            f"{row['calculated']:.12f} | {row['abs_diff']:.12f}"
        )
        if row["stage"] == "final_blend" and "bootstrap_95ci" in row:
            lo, hi = row["bootstrap_95ci"]
            print(f"final_blend 95% bootstrap CI: [{lo:.6f}, {hi:.6f}]")

    if final_warnings:
        print("warnings:")
        for w in final_warnings:
            print(f"- {w}")
    else:
        print("warnings: none")


if __name__ == "__main__":
    main()
