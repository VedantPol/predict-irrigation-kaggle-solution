#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExportPaths:
    root: Path
    tree_metrics: Path
    level3_metrics: Path
    level4_metrics: Path
    hill_metrics: Path
    stacking_metrics: Path
    pseudo_metrics: Path
    extra_metrics: Path
    out_json: Path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _round(value: Any, digits: int = 6) -> float | None:
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _build_default_paths(root: Path) -> ExportPaths:
    result_dir = root / "outputs" / "level2_results" / "tree_level2_dataset_irrigation_digit_decimal_v1"
    return ExportPaths(
        root=root,
        tree_metrics=result_dir / "tree_suite_metrics.json",
        level3_metrics=result_dir / "level3_final_metrics.json",
        level4_metrics=result_dir / "level4_final_metrics.json",
        hill_metrics=root / "outputs" / "level2_results" / "hill_climb" / "hill_climb_selection.json",
        stacking_metrics=root / "outputs" / "level2_results" / "stacking" / "stacking_metrics.json",
        pseudo_metrics=root / "outputs" / "level2_results" / "pseudo_labeling" / "pseudo_label_summary.json",
        extra_metrics=root / "outputs" / "level2_results" / "extra_training" / "extra_training_metrics.json",
        out_json=root / "docs" / "data" / "latest_run.json",
    )


def _group_type(library: str) -> str:
    if library.startswith("dl_"):
        return "deep"
    return "tree"


def export_dashboard_data(paths: ExportPaths) -> dict[str, Any]:
    tree = _load_json(paths.tree_metrics)
    level3 = _load_json(paths.level3_metrics)
    level4 = _load_json(paths.level4_metrics)
    hill = _load_json(paths.hill_metrics)
    stacking = _load_json(paths.stacking_metrics)
    pseudo = _load_json(paths.pseudo_metrics)
    extra = _load_json(paths.extra_metrics)

    model_metrics = list(tree.get("model_metrics", []))
    model_metrics_sorted = sorted(
        model_metrics,
        key=lambda row: float(row.get("valid_auc", 0.0)),
        reverse=True,
    )

    library_summary = []
    for name, stats in tree.get("library_summary", {}).items():
        library_summary.append(
            {
                "name": name,
                "kind": _group_type(name),
                "models": int(stats.get("models", 0)),
                "mean_valid_auc": _round(stats.get("mean_valid_auc")),
                "best_valid_auc": _round(stats.get("best_valid_auc")),
            }
        )
    library_summary.sort(key=lambda row: (row["best_valid_auc"] or 0.0), reverse=True)

    timeline = [
        {"stage": "level3", "valid_auc": _round(level3.get("valid_auc"))},
        {"stage": "level4", "valid_auc": _round(level4.get("valid_auc"))},
        {"stage": "hill_climb", "valid_auc": _round(hill.get("final_valid_auc"))},
        {"stage": "stacking", "valid_auc": _round(stacking.get("valid_auc"))},
    ]

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feature_set_name": tree.get("feature_set_name", "irrigation_digit_decimal_v1"),
            "result_folder": str(paths.tree_metrics.parent.relative_to(paths.root)),
        },
        "headline": {
            "models_trained": int(tree.get("n_models_trained", 0)),
            "level3_valid_auc": _round(level3.get("valid_auc")),
            "level4_valid_auc": _round(level4.get("valid_auc")),
            "stacking_valid_auc": _round(stacking.get("valid_auc")),
            "hill_climb_valid_auc": _round(hill.get("final_valid_auc")),
            "stacking_models": int(stacking.get("n_models", 0)),
            "pseudo_rows": int(pseudo.get("selected_rows", 0)),
            "pseudo_ratio": _round(pseudo.get("selection_ratio"), digits=4),
            "extra_train_rows": int(extra.get("n_train_rows", 0)),
            "pseudo_rows_added": int(extra.get("pseudo_rows_added", 0)),
        },
        "timeline": timeline,
        "library_summary": library_summary,
        "top_models": [
            {
                "model_name": row.get("model_name"),
                "library": row.get("library"),
                "backend": row.get("backend"),
                "valid_auc": _round(row.get("valid_auc")),
            }
            for row in model_metrics_sorted[:25]
        ],
    }
    return payload


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = _build_default_paths(root)
    payload = export_dashboard_data(paths)
    paths.out_json.parent.mkdir(parents=True, exist_ok=True)
    paths.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[DASHBOARD_EXPORT] Wrote: {paths.out_json}")


if __name__ == "__main__":
    main()
