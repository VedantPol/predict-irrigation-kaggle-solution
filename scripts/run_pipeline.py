#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_early_warning.pipeline import run_stage_with_config


# Edit these values directly, then right-click and run this file in your IDE.
# Set PIPELINE_STAGE to any stage in AUTO_STAGE_ORDER to execute all prior stages too.
# Use "extra_training" to execute the full KGMON-style sequence in AUTO_STAGE_ORDER.
PIPELINE_STAGE = "extra_training"
AUTO_STAGE_ORDER = (
    "prepare_data",
    "feature_engineering",
    "baseline",
    "eda",
    "hill_climb",
    "stacking",
    "pseudo_labeling",
    "extra_training",
)

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
    "models_per_library": 5,
    "tree_libraries": ("xgboost", "catboost"),
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
    "level4_regularization_c": 1.0,
    "min_oof_auc_for_selection": 0.36,
    "stacking_min_models": 16,
    "cleanup_oof_pred_before_baseline": True,
    "cleanup_level2_outputs_before_baseline": True,
    "level2_dataset_name": "tree_level2_dataset",
    "level3_model_name": "demo_xgb_level3",
    "level3_xgb_n_estimators": 180,
    "level3_xgb_learning_rate": 0.05,
    "level3_xgb_max_depth": 5,
    "level3_xgb_subsample": 0.85,
    "level3_xgb_colsample_bytree": 0.85,
}


def main() -> None:
    if PIPELINE_STAGE in AUTO_STAGE_ORDER:
        target_idx = AUTO_STAGE_ORDER.index(PIPELINE_STAGE) + 1
        for stage in AUTO_STAGE_ORDER[:target_idx]:
            print(f"[PIPELINE] Running stage: {stage}")
            run_stage_with_config(
                stage=stage,
                prepare_data_config=PREPARE_DATA_CONFIG,
                feature_engineering_config=FEATURE_ENGINEERING_CONFIG,
                baseline_config=BASELINE_CONFIG,
            )
        return

    run_stage_with_config(
        stage=PIPELINE_STAGE,
        prepare_data_config=PREPARE_DATA_CONFIG,
        feature_engineering_config=FEATURE_ENGINEERING_CONFIG,
        baseline_config=BASELINE_CONFIG,
    )


if __name__ == "__main__":
    main()
