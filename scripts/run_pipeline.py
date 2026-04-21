#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_early_warning.pipeline import run_stage_with_config


# Edit these values directly, then right-click and run this file in your IDE.
# Set PIPELINE_STAGE to any stage in AUTO_STAGE_ORDER to execute all prior stages too.
# Use "final_blend" to execute a robust notebook-style stacked blend and save submission_final_blend.csv.
PIPELINE_STAGE = "final_blend"
AUTO_STAGE_ORDER = (
    "prepare_data",
    "feature_engineering",
    "baseline",
    "eda",
    "hill_climb",
    "stacking",
    "final_blend",
    "pseudo_labeling",
    "extra_training",
)
# Resume behavior:
# - If True, unchanged heavy stages are skipped when their cached artifacts still match config+inputs.
# - Final target stage still runs by default so you always get a refreshed output file.
RESUME_FROM_ARTIFACTS = True
ALWAYS_RUN_TARGET_STAGE = True
# Add stage names here when you intentionally want to recompute them.
FORCE_RERUN_STAGES: set[str] = set()
CACHE_DIR = ROOT / "outputs" / "cache" / "stage_resume"

PREPARE_DATA_CONFIG = {
    "target_col": "Irrigation_Need",
    "random_state": 42,
    "valid_size": 0.20,
    "n_folds": 5,
    "source_train_csv": "data/train.csv",
    "source_test_csv": "data/test.csv",
    "id_col": "id",
}

# Level 1: digit and decimal extraction.
FEATURE_ENGINEERING_CONFIG = {
    "target_col": "Irrigation_Need",
    "base_numeric_columns": [
        "Soil_pH",
        "Soil_Moisture",
        "Organic_Carbon",
        "Electrical_Conductivity",
        "Temperature_C",
        "Humidity",
        "Rainfall_mm",
        "Sunlight_Hours",
        "Wind_Speed_kmh",
        "Field_Area_hectare",
        "Previous_Irrigation_mm",
    ],
    "snap_columns": [],
    "denominators": [2, 4, 5, 10],
    "round_threshold": 0.005,
    "include_digit_pair_strings": True,
    "keep_original_columns": True,
    "feature_set_name": "irrigation_digit_decimal_v1",
    "enable_advanced_features": True,
    "enabled_feature_families": [
        "target_encoding",
        "arithmetic",
        "multi_scale_binning",
        "cross_features",
        "frequency_count",
        "service_aggregations",
        "original_lookup",
        "radix_interactions",
        "artifact_detection",
        "projection_manifold",
    ],
    "nested_te_folds": 5,
    "fold_col": "cv_fold",
    "te_stats": [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "q05",
        "q10",
        "q45",
        "q55",
        "q90",
        "q95",
    ],
    "categorical_columns": [
        "Soil_Type",
        "Crop_Type",
        "Crop_Growth_Stage",
        "Season",
        "Irrigation_Type",
        "Water_Source",
        "Mulching_Used",
        "Region",
    ],
    "service_columns": [
        "Irrigation_Type",
        "Water_Source",
        "Mulching_Used",
        "Region",
    ],
    "numeric_columns_for_bins": [
        "Soil_pH",
        "Soil_Moisture",
        "Organic_Carbon",
        "Electrical_Conductivity",
        "Temperature_C",
        "Humidity",
        "Rainfall_mm",
        "Sunlight_Hours",
        "Wind_Speed_kmh",
        "Field_Area_hectare",
        "Previous_Irrigation_mm",
    ],
    "quantile_bins": [50, 200, 1000, 5000],
    "fixed_width_bins": [20, 50, 100],
    "log_bins": [20, 50],
    "monthly_charge_col": "Soil_Moisture",
    "total_charge_col": "Rainfall_mm",
    "tenure_col": "Previous_Irrigation_mm",
    "phone_service_col": "Irrigation_Type",
    "internet_service_col": "Water_Source",
    "original_reference_csv": "data/irrigation_prediction.csv",
    "original_target_col": "Irrigation_Need",
    "tfidf_max_features": 24,
    "projection_components": 12,
}

# Tree stack config:
# Level 1: original + engineered features
# Level 2: Level-1 features + outputs of tree and deep model suites
# Level 3: XGBoost on original Level-1 + all Level-2 outputs
# Level 4: Logistic stacker on Level-2 OOF + Level-3 OOF
BASELINE_CONFIG = {
    "target_col": "Irrigation_Need",
    "feature_set_name": "irrigation_digit_decimal_v1",
    "model_name": "irrigation_tree_suite_v1",
    "random_state": 42,
    "run_tree_level_stack": True,
    "models_per_library": 10,
    "use_tree_optuna": True,
    "tree_optuna_trials_per_library": 16,
    "tree_optuna_max_tuned_models_per_library": 5,
    "tree_optuna_train_max_rows": 220_000,
    "tree_optuna_timeout_sec": 900,
    "tree_libraries": ("xgboost", "xgboost_dart", "catboost", "catboost_native", "cuml_rf", "cuml_et", "lightgbm"),
    "strict_gpu_only": True,
    "min_tree_valid_auc_keep": 0.33,
    "run_deep_level2_stack": False,
    "deep_models_per_family": 5,
    "min_deep_valid_auc_keep": 0.40,
    "deep_model_families": (
        "embedding_mlp",
        "feature_interaction_mlp",
        "enhanced_feature_mlp",
        "realmlp",
        "graphsage_gnn",
        "ft_transformer",
        "tabtransformer",
        "tabm",
        "tabicl",
        "gandalf",
        "snn_selu",
        "tabular_resnet",
        "rff_kernel_network",
        "dae",
        "ffm",
        "deep_resnet",
        "fm",
        "deepfm",
        "liquid_nn",
        "vsn",
        "tabnet",
        "trompt",
        "danet",
        "tabpfn",
        "dae_transfer",
    ),
    "deep_epochs": 10,
    "deep_batch_size": 2048,
    "deep_learning_rate": 7e-4,
    "deep_weight_decay": 2e-5,
    "deep_eval_batch_size": 16384,
    "deep_early_stopping_patience": 4,
    "deep_use_amp": True,
    "deep_max_pos_weight": 50.0,
    "deep_feature_clip_value": 12.0,
    "deep_use_balanced_sampler": True,
    "deep_rescue_on_low_auc": True,
    "deep_rescue_auc_floor": 0.56,
    "deep_rescue_epochs": 8,
    "run_level4_stack": True,
    "level4_cv_folds": 5,
    "level4_regularization_c": 4.0,
    "min_oof_auc_for_selection": 0.36,
    "stacking_min_models": 24,
    "cleanup_oof_pred_before_baseline": True,
    "cleanup_level2_outputs_before_baseline": True,
    "level2_dataset_name": "tree_level2_dataset",
    "level3_model_name": "demo_xgb_level3",
    "level3_xgb_n_estimators": 260,
    "level3_xgb_learning_rate": 0.05,
    "level3_xgb_max_depth": 6,
    "level3_xgb_subsample": 0.90,
    "level3_xgb_colsample_bytree": 0.90,
    "level3_cv_enabled": True,
    "level3_cv_folds": 5,
    "level2_keep_top_models_for_meta": 24,
    "level2_keep_min_models_for_meta": 12,
    "level2_keep_model_auc_gap": 0.0025,
    "use_balanced_sample_weight": True,
    "min_sample_weight_ratio": 0.25,
    "max_sample_weight_ratio": 12.0,
    "enable_class_weight_calibration": True,
    "class_weight_calibration_grid": (0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.40, 1.60, 1.80, 2.00),
    "class_weight_calibration_random_trials": 120,
}


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_hash(v) for v in value]
    return value


def _hash_payload(payload: dict[str, Any]) -> str:
    normalized = _normalize_for_hash(payload)
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolve_path(raw_path: Any) -> Path | None:
    if not raw_path:
        return None
    p = Path(str(raw_path))
    return p if p.is_absolute() else (ROOT / p)


def _file_state(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _stage_marker(stage: str) -> Path:
    feature_set_name = str(FEATURE_ENGINEERING_CONFIG.get("feature_set_name", "irrigation_digit_decimal_v1"))
    level2_name = str(BASELINE_CONFIG.get("level2_dataset_name", "tree_level2_dataset"))
    marker_map: dict[str, Path] = {
        "prepare_data": ROOT / "data" / "processed" / "split_summary.json",
        "feature_engineering": ROOT / "data" / "processed" / "level1_features" / feature_set_name / "manifest.json",
        "baseline": ROOT / "outputs" / "level2_results" / f"{level2_name}_{feature_set_name}" / "tree_suite_metrics.json",
        "eda": ROOT / "outputs" / "level2_results" / "eda" / "eda_summary.json",
        "hill_climb": ROOT / "outputs" / "level2_results" / "hill_climb" / "hill_climb_selection.json",
        "stacking": ROOT / "outputs" / "level2_results" / "stacking" / "stacking_metrics.json",
        "final_blend": ROOT / "outputs" / "level2_results" / "final_blend" / "final_blend_summary.json",
        "pseudo_labeling": ROOT / "outputs" / "level2_results" / "pseudo_labeling" / "pseudo_label_summary.json",
        "extra_training": ROOT / "outputs" / "level2_results" / "extra_training" / "extra_training_metrics.json",
    }
    return marker_map.get(stage, ROOT / "outputs" / "level2_results" / stage / "done.marker")


def _stage_signature(stage: str) -> dict[str, Any]:
    if stage == "prepare_data":
        train_csv = _resolve_path(PREPARE_DATA_CONFIG.get("source_train_csv", "data/train.csv"))
        test_csv = _resolve_path(PREPARE_DATA_CONFIG.get("source_test_csv", "data/test.csv"))
        return {
            "stage": stage,
            "prepare_data_config": PREPARE_DATA_CONFIG,
            "train_csv": _file_state(train_csv),
            "test_csv": _file_state(test_csv),
            "pipeline_code": _file_state(ROOT / "src" / "fraud_risk_early_warning" / "pipeline.py"),
        }
    if stage == "feature_engineering":
        return {
            "stage": stage,
            "feature_engineering_config": FEATURE_ENGINEERING_CONFIG,
            "split_summary": _file_state(ROOT / "data" / "processed" / "split_summary.json"),
            "pipeline_code": _file_state(ROOT / "src" / "fraud_risk_early_warning" / "pipeline.py"),
            "advanced_features_code": _file_state(ROOT / "src" / "fraud_risk_early_warning" / "advanced_features.py"),
        }
    if stage == "baseline":
        feature_set_name = str(FEATURE_ENGINEERING_CONFIG.get("feature_set_name", "irrigation_digit_decimal_v1"))
        return {
            "stage": stage,
            "baseline_config": BASELINE_CONFIG,
            "feature_manifest": _file_state(
                ROOT / "data" / "processed" / "level1_features" / feature_set_name / "manifest.json"
            ),
            "pipeline_code": _file_state(ROOT / "src" / "fraud_risk_early_warning" / "pipeline.py"),
        }
    return {"stage": stage, "marker": _file_state(_stage_marker(stage))}


def _cache_file(stage: str) -> Path:
    return CACHE_DIR / f"{stage}.json"


def _read_stage_cache(stage: str) -> dict[str, Any] | None:
    path = _cache_file(stage)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_stage_cache(stage: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    marker = _stage_marker(stage)
    payload: dict[str, Any] = {
        "stage": stage,
        "marker": _file_state(marker),
    }
    if stage in {"prepare_data", "feature_engineering", "baseline"}:
        signature = _stage_signature(stage)
        payload["signature"] = signature
        payload["signature_hash"] = _hash_payload(signature)
    _cache_file(stage).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _should_skip_stage(stage: str, *, upstream_reran: bool, is_target_stage: bool) -> bool:
    if not RESUME_FROM_ARTIFACTS:
        return False
    if stage in FORCE_RERUN_STAGES:
        return False
    if upstream_reran:
        return False
    if is_target_stage and ALWAYS_RUN_TARGET_STAGE:
        return False

    marker = _stage_marker(stage)
    if not marker.exists():
        return False

    if stage not in {"prepare_data", "feature_engineering", "baseline"}:
        return True

    cache_payload = _read_stage_cache(stage)
    if not cache_payload:
        return False
    cached_hash = str(cache_payload.get("signature_hash", ""))
    current_hash = _hash_payload(_stage_signature(stage))
    return bool(cached_hash and cached_hash == current_hash)


def _run_pipeline_stage(stage: str) -> None:
    run_stage_with_config(
        stage=stage,
        prepare_data_config=PREPARE_DATA_CONFIG,
        feature_engineering_config=FEATURE_ENGINEERING_CONFIG,
        baseline_config=BASELINE_CONFIG,
    )


def main() -> None:
    if PIPELINE_STAGE in AUTO_STAGE_ORDER:
        target_idx = AUTO_STAGE_ORDER.index(PIPELINE_STAGE) + 1
        stages = list(AUTO_STAGE_ORDER[:target_idx])
    else:
        stages = [PIPELINE_STAGE]

    upstream_reran = False
    final_stage = stages[-1]
    for stage in stages:
        is_target_stage = stage == final_stage
        if _should_skip_stage(stage, upstream_reran=upstream_reran, is_target_stage=is_target_stage):
            print(f"[PIPELINE] Skipping stage (cached): {stage}")
            continue

        print(f"[PIPELINE] Running stage: {stage}")
        _run_pipeline_stage(stage)
        _write_stage_cache(stage)
        upstream_reran = True


if __name__ == "__main__":
    main()
