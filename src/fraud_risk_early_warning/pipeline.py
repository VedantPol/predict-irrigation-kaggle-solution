from __future__ import annotations

import copy
import inspect
import itertools
import json
import shutil
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .advanced_features import (
    DEFAULT_FAMILY_PREFIXES,
    DEFAULT_TELCO_CATEGORICAL_COLUMNS,
    DEFAULT_TELCO_SERVICE_COLUMNS,
    DEFAULT_TE_STATS,
    build_advanced_feature_set,
)
from .config import get_paths

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional runtime dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional runtime dependency
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional runtime dependency
    CatBoostClassifier = None

try:
    import optuna
    from optuna.samplers import TPESampler
except Exception:  # pragma: no cover - optional runtime dependency
    optuna = None
    TPESampler = None

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional runtime dependency
    cp = None

try:
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
except Exception:  # pragma: no cover - optional runtime dependency
    CuMLRandomForestClassifier = None

try:
    from cuml.ensemble import ExtraTreesClassifier as CuMLExtraTreesClassifier
except Exception:  # pragma: no cover - optional runtime dependency
    CuMLExtraTreesClassifier = None

try:
    from cuml.linear_model import LogisticRegression as CuMLLogisticRegression
except Exception:  # pragma: no cover - optional runtime dependency
    CuMLLogisticRegression = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None


_TORCH_MODULE_BASE = nn.Module if nn is not None else object


StageFn = Callable[[], None]


@dataclass(frozen=True)
class PrepareDataConfig:
    target_col: str = "Irrigation_Need"
    random_state: int = 42
    test_size: float = 0.20
    valid_size: float = 0.20
    n_folds: int = 5
    source_csv: str | None = None
    source_train_csv: str | None = "data/train.csv"
    source_test_csv: str | None = "data/test.csv"
    id_col: str = "id"


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    target_col: str = "Irrigation_Need"
    base_numeric_columns: list[str] = field(
        default_factory=lambda: [
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
        ]
    )
    snap_columns: list[str] = field(default_factory=list)
    denominators: list[int] = field(default_factory=lambda: [2, 4, 5, 10])
    round_threshold: float = 0.005
    include_digit_pair_strings: bool = True
    keep_original_columns: bool = True
    feature_set_name: str = "irrigation_digit_decimal_v1"
    enable_advanced_features: bool = True
    enabled_feature_families: list[str] = field(default_factory=lambda: list(DEFAULT_FAMILY_PREFIXES))
    nested_te_folds: int = 5
    fold_col: str = "cv_fold"
    te_stats: list[str] = field(default_factory=lambda: list(DEFAULT_TE_STATS))
    categorical_columns: list[str] = field(default_factory=lambda: list(DEFAULT_TELCO_CATEGORICAL_COLUMNS))
    service_columns: list[str] = field(default_factory=lambda: list(DEFAULT_TELCO_SERVICE_COLUMNS))
    numeric_columns_for_bins: list[str] = field(
        default_factory=lambda: [
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
        ]
    )
    quantile_bins: list[int] = field(default_factory=lambda: [50, 200, 1000, 5000])
    fixed_width_bins: list[int] = field(default_factory=lambda: [20, 50, 100])
    log_bins: list[int] = field(default_factory=lambda: [20, 50])
    monthly_charge_col: str = "Soil_Moisture"
    total_charge_col: str = "Rainfall_mm"
    tenure_col: str = "Previous_Irrigation_mm"
    phone_service_col: str = "Irrigation_Type"
    internet_service_col: str = "Water_Source"
    original_reference_csv: str | None = "data/irrigation_prediction.csv"
    original_target_col: str = "Irrigation_Need"
    tfidf_max_features: int = 24
    projection_components: int = 12


@dataclass(frozen=True)
class BaselineConfig:
    target_col: str = "Irrigation_Need"
    feature_set_name: str = "irrigation_digit_decimal_v1"
    model_name: str = "l2_hgb_fast_v1"
    learning_rate: float = 0.08
    max_depth: int = 6
    max_iter: int = 120
    min_samples_leaf: int = 100
    random_state: int = 42
    run_feature_family_suite: bool = False
    models_per_family: int = 5
    run_tree_level_stack: bool = True
    models_per_library: int = 5
    use_tree_optuna: bool = False
    tree_optuna_trials_per_library: int = 0
    tree_optuna_max_tuned_models_per_library: int = 2
    tree_optuna_train_max_rows: int = 180_000
    tree_optuna_timeout_sec: int = 0
    tree_libraries: tuple[str, ...] = ("xgboost", "catboost", "lightgbm", "cuml_rf", "cuml_et")
    strict_gpu_only: bool = True
    min_tree_valid_auc_keep: float = 0.0
    run_deep_level2_stack: bool = False
    deep_models_per_family: int = 5
    min_deep_valid_auc_keep: float = 0.60
    deep_model_families: tuple[str, ...] = (
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
    )
    deep_epochs: int = 2
    deep_batch_size: int = 8192
    deep_learning_rate: float = 8e-4
    deep_weight_decay: float = 1e-5
    deep_eval_batch_size: int = 16384
    deep_early_stopping_patience: int = 3
    deep_use_amp: bool = True
    deep_max_pos_weight: float = 50.0
    deep_feature_clip_value: float = 12.0
    deep_use_balanced_sampler: bool = True
    deep_rescue_on_low_auc: bool = True
    deep_rescue_auc_floor: float = 0.56
    deep_rescue_epochs: int = 8
    stacking_min_models: int = 8
    run_level4_stack: bool = True
    level4_cv_folds: int = 5
    level4_regularization_c: float = 1.0
    min_oof_auc_for_selection: float = 0.40
    cleanup_oof_pred_before_baseline: bool = True
    cleanup_level2_outputs_before_baseline: bool = True
    level2_dataset_name: str = "tree_level2_dataset"
    level3_model_name: str = "demo_xgb_level3"
    level3_xgb_n_estimators: int = 180
    level3_xgb_learning_rate: float = 0.05
    level3_xgb_max_depth: int = 5
    level3_xgb_subsample: float = 0.85
    level3_xgb_colsample_bytree: float = 0.85
    level3_cv_enabled: bool = True
    level3_cv_folds: int = 5
    level2_keep_top_models_for_meta: int = 36
    level2_keep_min_models_for_meta: int = 16
    level2_keep_model_auc_gap: float = 0.0025
    use_balanced_sample_weight: bool = True
    min_sample_weight_ratio: float = 0.25
    max_sample_weight_ratio: float = 12.0
    enable_class_weight_calibration: bool = True
    class_weight_calibration_grid: tuple[float, ...] = (
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
        1.10,
        1.20,
        1.40,
        1.60,
        1.80,
        2.00,
    )
    class_weight_calibration_random_trials: int = 120
    family_prefixes: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_FAMILY_PREFIXES.items()}
    )


def _ensure_base_dirs() -> None:
    paths = get_paths()
    for folder in (
        paths.raw_data,
        paths.processed_data,
        paths.level1_features_dir,
        paths.outputs_root,
        paths.level2_results_dir,
        paths.oof_dir,
        paths.pred_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet engine not found. Install pyarrow with: pip install pyarrow"
        ) from exc


def _as_binary_target(series: pd.Series, class_to_index: dict[str, int] | None = None) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.int32)

    s = series.astype("string").fillna("__NA__")
    if class_to_index is not None:
        mapped = s.map(class_to_index).fillna(0).astype(np.int32)
        return mapped.to_numpy()
    return pd.factorize(s, sort=True)[0].astype(np.int32)


def _target_mapping_from_series(series: pd.Series) -> tuple[list[str], dict[str, int]]:
    s = series.astype("string").fillna("__NA__")
    class_labels = sorted([str(v) for v in s.unique().tolist()])
    class_to_index = {c: i for i, c in enumerate(class_labels)}
    return class_labels, class_to_index


def _load_target_mapping(paths) -> tuple[list[str], dict[str, int]]:
    path = paths.processed_data / "target_mapping.json"
    if not path.exists():
        return [], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    class_labels = [str(x) for x in payload.get("class_labels", [])]
    class_to_index = {c: i for i, c in enumerate(class_labels)}
    return class_labels, class_to_index


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


def _score_predictions(y_true: np.ndarray, pred: np.ndarray) -> float:
    y_true_i = np.asarray(y_true).astype(np.int32).reshape(-1)
    if y_true_i.size == 0:
        return 0.0
    n_classes = int(np.max(y_true_i)) + 1
    pred_arr = np.asarray(pred)
    if n_classes <= 2:
        if pred_arr.ndim == 2 and pred_arr.shape[1] >= 2:
            return float(roc_auc_score(y_true_i, pred_arr[:, 1]))
        return float(roc_auc_score(y_true_i, pred_arr.reshape(-1)))
    proba = _ensure_proba_2d(pred_arr, n_classes=n_classes)
    y_hat = np.argmax(proba, axis=1).astype(np.int32)
    class_counts = np.bincount(y_true_i, minlength=n_classes).astype(np.float64)
    correct = np.bincount(y_true_i[y_hat == y_true_i], minlength=n_classes).astype(np.float64)
    valid = class_counts > 0
    if not np.any(valid):
        return 0.0
    recall = np.zeros(n_classes, dtype=np.float64)
    recall[valid] = correct[valid] / np.clip(class_counts[valid], 1.0, None)
    return float(np.mean(recall[valid]))


def _compute_balanced_sample_weight(
    y: np.ndarray,
    *,
    min_ratio: float,
    max_ratio: float,
) -> tuple[np.ndarray, dict[int, float]]:
    y_arr = np.asarray(y).astype(np.int32).reshape(-1)
    if y_arr.size == 0:
        return np.zeros(0, dtype=np.float32), {}

    counts = np.bincount(y_arr)
    present = np.where(counts > 0)[0]
    if present.size <= 1:
        return np.ones(y_arr.shape[0], dtype=np.float32), {int(present[0]) if present.size else 0: 1.0}

    avg_count = float(y_arr.shape[0]) / float(present.size)
    floor_ratio = float(max(1e-4, min_ratio))
    ceil_ratio = float(max(floor_ratio, max_ratio))
    class_weight_map: dict[int, float] = {}
    for cls in present:
        ratio = avg_count / float(max(1, counts[cls]))
        ratio = min(ceil_ratio, max(floor_ratio, ratio))
        class_weight_map[int(cls)] = float(ratio)

    sample_weight = np.array([class_weight_map.get(int(v), 1.0) for v in y_arr], dtype=np.float32)
    return sample_weight, class_weight_map


def _apply_class_probability_weights(pred: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    proba = _ensure_proba_2d(np.asarray(pred), n_classes=int(len(class_weights))).astype(np.float32)
    weights = np.asarray(class_weights, dtype=np.float32).reshape(1, -1)
    weighted = proba * weights
    row_sum = np.clip(weighted.sum(axis=1, keepdims=True), 1e-9, None)
    return (weighted / row_sum).astype(np.float32)


def _optimize_class_probability_weights(
    *,
    y_true: np.ndarray,
    pred: np.ndarray,
    grid_values: tuple[float, ...],
    random_trials: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    y_arr = np.asarray(y_true).astype(np.int32).reshape(-1)
    n_classes = int(np.max(y_arr)) + 1 if y_arr.size else 0
    if n_classes <= 1:
        proba = _ensure_proba_2d(np.asarray(pred), n_classes=max(1, n_classes)).astype(np.float32)
        return proba, np.ones(max(1, n_classes), dtype=np.float32), _score_predictions(y_arr, proba), 0

    base_proba = _ensure_proba_2d(np.asarray(pred), n_classes=n_classes).astype(np.float32)
    best_pred = base_proba
    best_weights = np.ones(n_classes, dtype=np.float32)
    best_score = float(_score_predictions(y_arr, best_pred))
    evaluated = 0

    grid = tuple(sorted(set(float(v) for v in grid_values if float(v) > 0.0)))
    if not grid:
        grid = (1.0,)

    if n_classes <= 4 and len(grid) <= 11:
        for weights in itertools.product(grid, repeat=n_classes):
            w = np.asarray(weights, dtype=np.float32)
            calibrated = _apply_class_probability_weights(base_proba, w)
            score = float(_score_predictions(y_arr, calibrated))
            evaluated += 1
            if score > best_score + 1e-10:
                best_score = score
                best_pred = calibrated
                best_weights = w
    else:
        rng = np.random.default_rng(random_state)
        for _ in range(max(0, int(random_trials))):
            log_w = rng.normal(0.0, 0.35, size=n_classes)
            w = np.exp(log_w).astype(np.float32)
            calibrated = _apply_class_probability_weights(base_proba, w)
            score = float(_score_predictions(y_arr, calibrated))
            evaluated += 1
            if score > best_score + 1e-10:
                best_score = score
                best_pred = calibrated
                best_weights = w

    return best_pred.astype(np.float32), best_weights.astype(np.float32), float(best_score), int(evaluated)


def _apply_logit_bias_temperature(
    pred: np.ndarray,
    *,
    bias: np.ndarray,
    temperature: float,
) -> np.ndarray:
    proba = _ensure_proba_2d(np.asarray(pred), n_classes=int(len(bias))).astype(np.float32)
    bias_arr = np.asarray(bias, dtype=np.float32).reshape(1, -1)
    t = float(max(1e-4, temperature))
    logits = np.log(np.clip(proba, 1e-9, 1.0)) / t
    logits = logits + bias_arr
    logits = logits - np.max(logits, axis=1, keepdims=True)
    out = np.exp(logits).astype(np.float32)
    out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
    return out.astype(np.float32)


def _optimize_logit_bias_temperature(
    *,
    y_true: np.ndarray,
    pred: np.ndarray,
    random_trials: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    y_arr = np.asarray(y_true).astype(np.int32).reshape(-1)
    n_classes = int(np.max(y_arr)) + 1 if y_arr.size else 0
    if n_classes <= 1:
        proba = _ensure_proba_2d(np.asarray(pred), n_classes=max(1, n_classes)).astype(np.float32)
        return proba, np.zeros(max(1, n_classes), dtype=np.float32), 1.0, _score_predictions(y_arr, proba), 0

    base_proba = _ensure_proba_2d(np.asarray(pred), n_classes=n_classes).astype(np.float32)
    best_pred = base_proba
    best_bias = np.zeros(n_classes, dtype=np.float32)
    best_temp = 1.0
    best_score = float(_score_predictions(y_arr, best_pred))
    evaluated = 0

    bias_grid = (-0.35, -0.25, -0.15, -0.08, 0.0, 0.08, 0.15, 0.25, 0.35)
    temp_grid = (0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.40)
    if n_classes <= 4:
        for temp in temp_grid:
            if n_classes == 2:
                for b1 in bias_grid:
                    b = np.array([0.0, float(b1)], dtype=np.float32)
                    calibrated = _apply_logit_bias_temperature(base_proba, bias=b, temperature=float(temp))
                    score = float(_score_predictions(y_arr, calibrated))
                    evaluated += 1
                    if score > best_score + 1e-10:
                        best_score = score
                        best_pred = calibrated
                        best_bias = b
                        best_temp = float(temp)
            elif n_classes == 3:
                for b1, b2 in itertools.product(bias_grid, repeat=2):
                    b = np.array([0.0, float(b1), float(b2)], dtype=np.float32)
                    calibrated = _apply_logit_bias_temperature(base_proba, bias=b, temperature=float(temp))
                    score = float(_score_predictions(y_arr, calibrated))
                    evaluated += 1
                    if score > best_score + 1e-10:
                        best_score = score
                        best_pred = calibrated
                        best_bias = b
                        best_temp = float(temp)
            else:
                for b1, b2, b3 in itertools.product(bias_grid, repeat=3):
                    b = np.array([0.0, float(b1), float(b2), float(b3)], dtype=np.float32)
                    calibrated = _apply_logit_bias_temperature(base_proba, bias=b, temperature=float(temp))
                    score = float(_score_predictions(y_arr, calibrated))
                    evaluated += 1
                    if score > best_score + 1e-10:
                        best_score = score
                        best_pred = calibrated
                        best_bias = b
                        best_temp = float(temp)

    rng = np.random.default_rng(random_state)
    ref_scale = 0.28
    for _ in range(max(0, int(random_trials))):
        bias = np.zeros(n_classes, dtype=np.float32)
        bias[1:] = rng.normal(0.0, ref_scale, size=n_classes - 1).astype(np.float32)
        temp = float(np.exp(rng.normal(0.0, 0.22)))
        calibrated = _apply_logit_bias_temperature(base_proba, bias=bias, temperature=temp)
        score = float(_score_predictions(y_arr, calibrated))
        evaluated += 1
        if score > best_score + 1e-10:
            best_score = score
            best_pred = calibrated
            best_bias = bias
            best_temp = float(temp)
            ref_scale = max(0.12, ref_scale * 0.95)

    return (
        best_pred.astype(np.float32),
        best_bias.astype(np.float32),
        float(best_temp),
        float(best_score),
        int(evaluated),
    )


def _proba_to_labels(pred: np.ndarray, class_labels: list[str]) -> np.ndarray:
    arr = np.asarray(pred)
    inferred_classes = arr.shape[1] if arr.ndim == 2 else max(2, len(class_labels))
    proba = _ensure_proba_2d(arr, n_classes=max(1, len(class_labels) or inferred_classes))
    idx = np.argmax(proba, axis=1).astype(np.int32)
    if not class_labels:
        return idx.astype(str)
    return np.array([class_labels[i] for i in idx], dtype=object)


def _model_variants(base_cfg: BaselineConfig, n_models: int) -> list[dict[str, float | int]]:
    presets = [
        {"learning_rate": 0.03, "max_depth": 4, "max_iter": 250, "min_samples_leaf": 60},
        {"learning_rate": 0.05, "max_depth": 6, "max_iter": 220, "min_samples_leaf": 80},
        {"learning_rate": 0.08, "max_depth": 8, "max_iter": 180, "min_samples_leaf": 120},
        {"learning_rate": 0.12, "max_depth": 6, "max_iter": 140, "min_samples_leaf": 160},
        {"learning_rate": 0.18, "max_depth": 4, "max_iter": 110, "min_samples_leaf": 220},
    ]
    out = []
    for idx in range(max(1, n_models)):
        base = presets[idx % len(presets)].copy()
        base["random_state"] = base_cfg.random_state + idx
        out.append(base)
    return out


def _fit_predict_hgb(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
    params: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray]:
    model = HistGradientBoostingClassifier(
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        max_iter=int(params["max_iter"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=int(params["random_state"]),
    )
    model.fit(x_train, y_train)
    valid_pred = model.predict_proba(x_valid)
    test_pred = model.predict_proba(x_test)
    return valid_pred, test_pred


def _save_model_outputs(
    *,
    paths,
    cfg: BaselineConfig,
    model_name: str,
    selected_cols: list[str],
    valid_pred: np.ndarray,
    test_pred: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, float | int] | None = None,
    family_name: str | None = None,
) -> dict[str, object]:
    valid_auc = _score_predictions(y_valid, valid_pred)
    pred_tag = f"{model_name}_{_pred_run_tag(cfg)}"
    np.save(paths.oof_dir / f"oof_{pred_tag}.npy", valid_pred.astype(np.float32))
    np.save(paths.pred_dir / f"pred_{pred_tag}.npy", test_pred.astype(np.float32))

    level2_dir = paths.level2_results_dir / pred_tag
    level2_dir.mkdir(parents=True, exist_ok=True)
    valid_df = pd.DataFrame({"y_true": y_valid.astype(np.int32)})
    if np.asarray(valid_pred).ndim == 1:
        valid_df["y_pred"] = np.asarray(valid_pred).astype(np.float32)
    else:
        vp = np.asarray(valid_pred)
        for c in range(vp.shape[1]):
            valid_df[f"y_pred_c{c}"] = vp[:, c].astype(np.float32)
    valid_df.to_parquet(level2_dir / "valid_predictions.parquet", index=False)

    metrics = {
        "stage": "level2_baseline",
        "model": "HistGradientBoostingClassifier",
        "model_name": model_name,
        "feature_set_name": cfg.feature_set_name,
        "valid_auc": float(valid_auc),
        "n_features_numeric_used": int(len(selected_cols)),
        "n_valid": int(len(y_valid)),
        "feature_family": family_name,
        "config": asdict(cfg),
        "model_params": params or {},
    }
    (level2_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _short_exc(exc: Exception, max_len: int = 240) -> str:
    msg = str(exc).strip()
    if not msg:
        return exc.__class__.__name__
    first_line = msg.splitlines()[0].strip()
    if len(first_line) > max_len:
        return first_line[:max_len] + "..."
    return first_line


def _pred_run_tag(cfg: BaselineConfig) -> str:
    return f"{cfg.feature_set_name}_rs{int(cfg.random_state)}"


def _cleanup_stale_artifacts(paths, cfg: BaselineConfig) -> None:
    removed_oof = 0
    removed_pred = 0
    removed_dirs = 0

    if cfg.cleanup_oof_pred_before_baseline:
        for p in paths.oof_dir.glob("oof_*.npy"):
            try:
                p.unlink()
                removed_oof += 1
            except Exception:
                continue
        for p in paths.pred_dir.glob("pred_*.npy"):
            try:
                p.unlink()
                removed_pred += 1
            except Exception:
                continue

    if cfg.cleanup_level2_outputs_before_baseline:
        target_dirs = [
            paths.level2_results_dir / f"{cfg.level2_dataset_name}_{cfg.feature_set_name}",
            paths.level2_results_dir / "hill_climb",
            paths.level2_results_dir / "stacking",
            paths.level2_results_dir / "pseudo_labeling",
            paths.level2_results_dir / "extra_training",
        ]
        for d in target_dirs:
            if not d.exists():
                continue
            try:
                shutil.rmtree(d)
                removed_dirs += 1
            except Exception:
                continue

    print(
        "[BASELINE] Cleaned stale artifacts: "
        f"oof={removed_oof}, pred={removed_pred}, dirs={removed_dirs}"
    )


_GPU_PREFLIGHT_CACHE: dict[str, tuple[bool, str]] = {}


def _gpu_preflight_library(library_name: str) -> tuple[bool, str]:
    cached = _GPU_PREFLIGHT_CACHE.get(library_name)
    if cached is not None:
        return cached

    if cp is None:
        result = (False, "cupy not installed")
        _GPU_PREFLIGHT_CACHE[library_name] = result
        return result

    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            result = (False, "no CUDA device visible")
            _GPU_PREFLIGHT_CACHE[library_name] = result
            return result
    except Exception as exc:  # pragma: no cover - runtime dependent
        result = (False, f"CUDA runtime check failed: {_short_exc(exc)}")
        _GPU_PREFLIGHT_CACHE[library_name] = result
        return result

    x_np = np.random.RandomState(42).randn(128, 8).astype(np.float32)
    y_np = (np.random.RandomState(7).rand(128) > 0.5).astype(np.float32)

    try:
        if library_name in {"xgboost", "xgboost_dart"}:
            if XGBClassifier is None:
                result = (False, "xgboost not installed")
            else:
                params = dict(
                    objective="binary:logistic",
                    eval_metric="auc",
                    device="cuda",
                    tree_method="hist",
                    n_estimators=8,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    n_jobs=1,
                    random_state=42,
                )
                if library_name == "xgboost_dart":
                    params.update({"booster": "dart", "rate_drop": 0.1, "skip_drop": 0.4})
                model = XGBClassifier(**params)
                model.fit(cp.asarray(x_np), y_np)
                _ = model.predict_proba(cp.asarray(x_np))
                result = (True, "ok")

        elif library_name == "lightgbm":
            if LGBMClassifier is None:
                result = (False, "lightgbm not installed")
            else:
                model = LGBMClassifier(
                    objective="binary",
                    device_type="gpu",
                    n_estimators=12,
                    max_depth=4,
                    learning_rate=0.1,
                    n_jobs=1,
                    verbose=-1,
                )
                model.fit(x_np, y_np.astype(np.int32))
                _ = model.predict_proba(x_np)
                result = (True, "ok")

        elif library_name in {"catboost", "catboost_native"}:
            if CatBoostClassifier is None:
                result = (False, "catboost not installed")
            else:
                model = CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="AUC",
                    task_type="GPU",
                    devices="0",
                    allow_writing_files=False,
                    verbose=False,
                    iterations=10,
                    depth=4,
                    learning_rate=0.1,
                )
                model.fit(x_np, y_np.astype(np.int32))
                _ = model.predict_proba(x_np)
                result = (True, "ok")

        elif library_name == "cuml_rf":
            if CuMLRandomForestClassifier is None:
                result = (False, "cuml RandomForestClassifier unavailable")
            else:
                model = CuMLRandomForestClassifier(n_estimators=8, max_depth=6, random_state=42)
                model.fit(cp.asarray(x_np), cp.asarray(y_np))
                _ = model.predict_proba(cp.asarray(x_np))
                result = (True, "ok")

        elif library_name == "cuml_et":
            if CuMLExtraTreesClassifier is None:
                result = (False, "cuml ExtraTreesClassifier unavailable")
            else:
                model = CuMLExtraTreesClassifier(n_estimators=8, max_depth=6, random_state=42)
                model.fit(cp.asarray(x_np), cp.asarray(y_np))
                _ = model.predict_proba(cp.asarray(x_np))
                result = (True, "ok")

        elif library_name == "ydf":
            result = (False, "ydf backend is CPU-only in this project")
        else:
            result = (False, f"unsupported library '{library_name}'")
    except Exception as exc:
        result = (False, _short_exc(exc))

    _GPU_PREFLIGHT_CACHE[library_name] = result
    return result


def _to_numpy(x: object) -> np.ndarray:
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def _to_cupy_frame(x: pd.DataFrame) -> object:
    if cp is None:
        raise RuntimeError("GPU mode requires cupy, but cupy is not installed.")
    return cp.asarray(x.to_numpy(dtype=np.float32))


def _to_cupy_target(y: np.ndarray) -> object:
    if cp is None:
        raise RuntimeError("GPU mode requires cupy, but cupy is not installed.")
    return cp.asarray(y.astype(np.int32))


def _predict_score(model: object, x: pd.DataFrame | np.ndarray, *, n_classes: int = 2) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        pred = getattr(model, "predict_proba")(x)
        arr = _to_numpy(pred)
        return _ensure_proba_2d(arr, n_classes=n_classes)

    if hasattr(model, "decision_function"):
        pred = _to_numpy(getattr(model, "decision_function")(x)).astype(np.float32)
        if pred.ndim == 1:
            return _ensure_proba_2d(_sigmoid(pred), n_classes=n_classes)
        ex = np.exp(np.clip(pred, -20.0, 20.0))
        probs = ex / np.clip(np.sum(ex, axis=1, keepdims=True), 1e-9, None)
        return _ensure_proba_2d(probs, n_classes=n_classes)

    pred = _to_numpy(getattr(model, "predict")(x)).astype(np.float32).reshape(-1)
    return _ensure_proba_2d(pred, n_classes=n_classes)


def _encode_level_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    candidate_cols = [c for c in train_df.columns if c not in {target_col, "cv_fold"}]
    train_cols: dict[str, pd.Series] = {}
    valid_cols: dict[str, pd.Series] = {}
    test_cols: dict[str, pd.Series] = {}

    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            train_cols[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).astype(np.float32)
            valid_cols[col] = pd.to_numeric(valid_df[col], errors="coerce").fillna(0.0).astype(np.float32)
            test_cols[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(0.0).astype(np.float32)
            continue

        train_s = train_df[col].astype("string").fillna("__NA__")
        valid_s = valid_df[col].astype("string").fillna("__NA__")
        test_s = test_df[col].astype("string").fillna("__NA__")

        categories = pd.Index(train_s.unique())
        train_codes = pd.Categorical(train_s, categories=categories).codes.astype(np.int32)
        valid_codes = pd.Categorical(valid_s, categories=categories).codes.astype(np.int32)
        test_codes = pd.Categorical(test_s, categories=categories).codes.astype(np.int32)

        train_cols[col] = pd.Series(train_codes.astype(np.float32), index=train_df.index)
        valid_cols[col] = pd.Series(valid_codes.astype(np.float32), index=valid_df.index)
        test_cols[col] = pd.Series(test_codes.astype(np.float32), index=test_df.index)

    train_encoded = pd.DataFrame(train_cols, index=train_df.index)
    valid_encoded = pd.DataFrame(valid_cols, index=valid_df.index)
    test_encoded = pd.DataFrame(test_cols, index=test_df.index)
    return train_encoded, valid_encoded, test_encoded, candidate_cols


_DEEP_MODEL_FAMILY_SET: set[str] = {
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
}


class _TorchTabularMLP(_TORCH_MODULE_BASE):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        activation: str = "gelu",
        residual: bool = False,
        multiplicative: bool = False,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.multiplicative = multiplicative
        self.activation = activation

        self.input = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(1, depth))])
        self.gates = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(1, depth))])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity() for _ in range(max(1, depth))]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def _act(self, x):
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "selu":
            return F.selu(x)
        return F.gelu(x)

    def forward(self, x):
        h = self._act(self.input(x))
        for block, gate, norm in zip(self.blocks, self.gates, self.norms):
            z = self._act(norm(block(h)))
            if self.multiplicative:
                z = z * torch.sigmoid(gate(h))
            z = self.dropout(z)
            if self.residual and z.shape == h.shape:
                h = h + z
            else:
                h = z
        return self.out(h).squeeze(1)


class _TorchFMHead(_TORCH_MODULE_BASE):
    def __init__(self, in_features: int, factor_dim: int, with_deep: bool, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.factors = nn.Parameter(torch.randn(in_features, factor_dim) * 0.01)
        self.with_deep = with_deep
        if with_deep:
            self.deep = _TorchTabularMLP(
                in_features=in_features,
                hidden_dim=hidden_dim,
                depth=2,
                dropout=dropout,
                activation="gelu",
                residual=False,
                multiplicative=False,
                layer_norm=True,
            )
        else:
            self.deep = None

    def forward(self, x):
        linear_term = self.linear(x).squeeze(1)
        xv = x @ self.factors
        x2v2 = (x * x) @ (self.factors * self.factors)
        fm_term = 0.5 * torch.sum((xv * xv) - x2v2, dim=1)
        out = linear_term + fm_term
        if self.with_deep and self.deep is not None:
            out = out + self.deep(x)
        return out


def _resolve_available_deep_families(families: tuple[str, ...]) -> tuple[str, ...]:
    unknown = sorted([name for name in families if name not in _DEEP_MODEL_FAMILY_SET])
    if unknown:
        raise ValueError(f"Unsupported deep model families requested: {', '.join(unknown)}")
    if torch is None:
        raise RuntimeError("Deep Level-2 strict GPU mode requires PyTorch, but torch is not installed.")
    if not torch.cuda.is_available():
        raise RuntimeError("Deep Level-2 strict GPU mode requires CUDA-enabled PyTorch.")
    return tuple(families)


def _deep_family_profile(family: str) -> dict[str, object]:
    defaults = {
        "mode": "mlp",
        "activation": "gelu",
        "residual": False,
        "multiplicative": False,
        "layer_norm": True,
        "noise_std": 0.0,
        "label_smoothing": 0.0,
    }
    family_overrides = {
        "embedding_mlp": {"label_smoothing": 0.02},
        "feature_interaction_mlp": {"multiplicative": True},
        "enhanced_feature_mlp": {},
        "realmlp": {"activation": "silu"},
        "graphsage_gnn": {"residual": True},
        "ft_transformer": {"activation": "gelu", "residual": True},
        "tabtransformer": {"activation": "gelu", "residual": True},
        "tabm": {"activation": "silu", "multiplicative": True},
        "tabicl": {"activation": "gelu", "residual": True},
        "gandalf": {"activation": "silu", "multiplicative": True},
        "snn_selu": {"activation": "selu", "layer_norm": False},
        "tabular_resnet": {"activation": "relu", "residual": True},
        "rff_kernel_network": {"activation": "gelu"},
        "dae": {"activation": "silu", "noise_std": 0.04},
        "ffm": {"mode": "ffm"},
        "deep_resnet": {"activation": "gelu", "residual": True},
        "fm": {"mode": "fm"},
        "deepfm": {"mode": "deepfm"},
        "liquid_nn": {"activation": "silu", "residual": True},
        "vsn": {"activation": "silu", "multiplicative": True},
        "tabnet": {"activation": "relu", "multiplicative": True},
        "trompt": {"activation": "gelu", "residual": True},
        "danet": {"activation": "relu", "multiplicative": True},
        "tabpfn": {"activation": "gelu", "residual": True},
        "dae_transfer": {"activation": "silu", "noise_std": 0.03},
    }
    profile = defaults.copy()
    profile.update(family_overrides[family])
    return profile


def _deep_variant_grid(seed: int, n_models: int, cfg: BaselineConfig) -> list[dict[str, object]]:
    presets = [
        {"hidden_dim": 512, "depth": 3, "dropout": 0.15, "lr_scale": 1.00, "wd_scale": 1.00, "factor_dim": 32},
        {"hidden_dim": 384, "depth": 4, "dropout": 0.10, "lr_scale": 0.80, "wd_scale": 1.00, "factor_dim": 24},
        {"hidden_dim": 640, "depth": 3, "dropout": 0.18, "lr_scale": 0.90, "wd_scale": 0.80, "factor_dim": 40},
        {"hidden_dim": 448, "depth": 5, "dropout": 0.08, "lr_scale": 0.65, "wd_scale": 1.20, "factor_dim": 28},
        {"hidden_dim": 768, "depth": 3, "dropout": 0.20, "lr_scale": 0.55, "wd_scale": 0.90, "factor_dim": 48},
    ]
    out: list[dict[str, object]] = []
    for idx in range(max(1, n_models)):
        preset = presets[idx % len(presets)].copy()
        preset["seed"] = seed + idx
        preset["epochs"] = max(1, cfg.deep_epochs)
        preset["batch_size"] = int(cfg.deep_batch_size)
        preset["learning_rate"] = float(cfg.deep_learning_rate * preset.pop("lr_scale"))
        preset["weight_decay"] = float(cfg.deep_weight_decay * preset.pop("wd_scale"))
        out.append(preset)
    return out


def _rank_transform_like(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr, axis=0)
    ranks = np.empty_like(order, dtype=np.float32)
    n = max(1, arr.shape[0] - 1)
    for col in range(arr.shape[1]):
        ranks[order[:, col], col] = np.arange(arr.shape[0], dtype=np.float32) / float(n)
    return ranks


def _frequency_encode_like(train: np.ndarray, other: np.ndarray) -> np.ndarray:
    out = np.zeros_like(other, dtype=np.float32)
    for col in range(train.shape[1]):
        tr = np.round(train[:, col], 3)
        ot = np.round(other[:, col], 3)
        vals, counts = np.unique(tr, return_counts=True)
        mapper = dict(zip(vals.tolist(), (counts / max(1, train.shape[0])).astype(np.float32).tolist()))
        out[:, col] = np.array([mapper.get(v, 0.0) for v in ot], dtype=np.float32)
    return out


def _augment_deep_family_features(
    family: str,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enhanced_like_families = {
        "enhanced_feature_mlp",
        "ft_transformer",
        "tabtransformer",
        "tabicl",
        "tabpfn",
        "trompt",
        "graphsage_gnn",
    }
    if family in enhanced_like_families:
        n_cols = min(32, x_train.shape[1])
        x_train_sub = x_train[:, :n_cols]
        x_valid_sub = x_valid[:, :n_cols]
        x_test_sub = x_test[:, :n_cols]
        train_abs = np.abs(x_train_sub)
        valid_abs = np.abs(x_valid_sub)
        test_abs = np.abs(x_test_sub)

        train_rank = _rank_transform_like(x_train_sub)
        valid_rank = _rank_transform_like(x_valid_sub)
        test_rank = _rank_transform_like(x_test_sub)

        train_freq = _frequency_encode_like(x_train_sub, x_train_sub)
        valid_freq = _frequency_encode_like(x_train_sub, x_valid_sub)
        test_freq = _frequency_encode_like(x_train_sub, x_test_sub)

        train_aug = np.concatenate(
            [
                x_train,
                train_freq,
                train_rank,
                np.log1p(train_abs),
                np.sqrt(train_abs + 1e-8),
                1.0 / (1.0 + train_abs),
            ],
            axis=1,
        )
        valid_aug = np.concatenate(
            [
                x_valid,
                valid_freq,
                valid_rank,
                np.log1p(valid_abs),
                np.sqrt(valid_abs + 1e-8),
                1.0 / (1.0 + valid_abs),
            ],
            axis=1,
        )
        test_aug = np.concatenate(
            [
                x_test,
                test_freq,
                test_rank,
                np.log1p(test_abs),
                np.sqrt(test_abs + 1e-8),
                1.0 / (1.0 + test_abs),
            ],
            axis=1,
        )
        return train_aug.astype(np.float32), valid_aug.astype(np.float32), test_aug.astype(np.float32)

    if family == "rff_kernel_network":
        rng = np.random.default_rng(seed)
        in_dim = x_train.shape[1]
        out_dim = min(512, max(128, in_dim * 2))
        sigma = float(np.std(x_train) + 1e-6)
        w = rng.normal(0.0, 1.0 / sigma, size=(in_dim, out_dim)).astype(np.float32)
        b = rng.uniform(0.0, 2.0 * np.pi, size=(out_dim,)).astype(np.float32)
        scale = np.sqrt(2.0 / float(out_dim))
        train_rff = scale * np.cos((x_train @ w) + b)
        valid_rff = scale * np.cos((x_valid @ w) + b)
        test_rff = scale * np.cos((x_test @ w) + b)
        return (
            np.concatenate([x_train, train_rff], axis=1).astype(np.float32),
            np.concatenate([x_valid, valid_rff], axis=1).astype(np.float32),
            np.concatenate([x_test, test_rff], axis=1).astype(np.float32),
        )

    return x_train.astype(np.float32), x_valid.astype(np.float32), x_test.astype(np.float32)


def _robust_scale_deep_inputs(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    clip_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(x_train, dtype=np.float32)
    valid = np.asarray(x_valid, dtype=np.float32)
    test = np.asarray(x_test, dtype=np.float32)

    med = np.median(train, axis=0).astype(np.float32)
    q75 = np.percentile(train, 75, axis=0).astype(np.float32)
    q25 = np.percentile(train, 25, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)
    iqr = np.where(np.abs(iqr) < 1e-6, 1.0, iqr).astype(np.float32)

    train_s = (train - med) / iqr
    valid_s = (valid - med) / iqr
    test_s = (test - med) / iqr

    clip_val = float(max(3.0, clip_value))
    train_s = np.clip(train_s, -clip_val, clip_val)
    valid_s = np.clip(valid_s, -clip_val, clip_val)
    test_s = np.clip(test_s, -clip_val, clip_val)

    train_s = np.nan_to_num(train_s, nan=0.0, posinf=clip_val, neginf=-clip_val).astype(np.float32)
    valid_s = np.nan_to_num(valid_s, nan=0.0, posinf=clip_val, neginf=-clip_val).astype(np.float32)
    test_s = np.nan_to_num(test_s, nan=0.0, posinf=clip_val, neginf=-clip_val).astype(np.float32)
    return train_s, valid_s, test_s


def _build_deep_family_model(family: str, in_features: int, params: dict[str, object]):
    profile = _deep_family_profile(family)
    mode = str(profile["mode"])
    if mode in {"fm", "ffm"}:
        factor_dim = int(params.get("factor_dim", 32))
        return _TorchFMHead(
            in_features=in_features,
            factor_dim=factor_dim,
            with_deep=False,
            hidden_dim=int(params["hidden_dim"]),
            dropout=float(params["dropout"]),
        )
    if mode == "deepfm":
        factor_dim = int(params.get("factor_dim", 32))
        return _TorchFMHead(
            in_features=in_features,
            factor_dim=factor_dim,
            with_deep=True,
            hidden_dim=int(params["hidden_dim"]),
            dropout=float(params["dropout"]),
        )
    return _TorchTabularMLP(
        in_features=in_features,
        hidden_dim=int(params["hidden_dim"]),
        depth=int(params["depth"]),
        dropout=float(params["dropout"]),
        activation=str(profile["activation"]),
        residual=bool(profile["residual"]),
        multiplicative=bool(profile["multiplicative"]),
        layer_norm=bool(profile["layer_norm"]),
    )


def _predict_with_torch_model(model, x_np: np.ndarray, batch_size: int, device) -> np.ndarray:
    was_training = bool(model.training)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_np.shape[0], batch_size):
            xb = torch.from_numpy(x_np[start : start + batch_size]).to(device=device, dtype=torch.float32)
            logits = model(xb)
            prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            preds.append(prob.reshape(-1))
    if was_training:
        model.train()
    return np.concatenate(preds, axis=0).astype(np.float32)


def _fit_deep_family_model(
    cfg: BaselineConfig,
    family: str,
    params: dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if torch is None:
        raise RuntimeError("Deep Level-2 stack requires torch.")
    device = torch.device("cuda")
    torch.manual_seed(int(params["seed"]))
    torch.cuda.manual_seed_all(int(params["seed"]))

    x_train_s, x_valid_s, x_test_s = _robust_scale_deep_inputs(
        x_train=x_train,
        x_valid=x_valid,
        x_test=x_test,
        clip_value=cfg.deep_feature_clip_value,
    )
    profile = _deep_family_profile(family)
    noise_std = float(profile.get("noise_std", 0.0))
    label_smoothing = float(profile.get("label_smoothing", 0.0))

    y_train_float = y_train.astype(np.float32)
    pos_count = float(np.sum(y_train_float > 0.5))
    neg_count = float(max(0, y_train_float.shape[0] - pos_count))
    def _train_once(local_params: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        model = _build_deep_family_model(family, in_features=x_train_s.shape[1], params=local_params).to(device)
        batch_size = max(1024, int(local_params["batch_size"]))

        train_ds = TensorDataset(
            torch.from_numpy(x_train_s),
            torch.from_numpy(y_train_float),
        )
        imbalance_ratio = (neg_count / max(1.0, pos_count)) if pos_count > 0 else 1.0
        use_balanced_sampler = bool(
            cfg.deep_use_balanced_sampler
            and WeightedRandomSampler is not None
            and pos_count > 0
            and imbalance_ratio >= 4.0
        )
        if use_balanced_sampler:
            sample_w = np.where(y_train_float > 0.5, float(min(cfg.deep_max_pos_weight, imbalance_ratio)), 1.0).astype(
                np.float32
            )
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_w),
                num_samples=int(sample_w.shape[0]),
                replacement=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=False,
                pin_memory=True,
                num_workers=0,
            )
            criterion = nn.BCEWithLogitsLoss()
        else:
            if pos_count > 0:
                pos_weight_value = min(float(cfg.deep_max_pos_weight), imbalance_ratio)
            else:
                pos_weight_value = 1.0
            pos_weight_t = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                pin_memory=True,
                num_workers=0,
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(local_params["learning_rate"]),
            weight_decay=float(local_params["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(local_params["epochs"])),
        )
        use_amp = bool(cfg.deep_use_amp and torch.cuda.is_available())
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        patience = max(1, int(cfg.deep_early_stopping_patience))
        best_auc_local = -1.0
        best_state_local = copy.deepcopy(model.state_dict())
        no_improve_epochs = 0

        for _ in range(int(local_params["epochs"])):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device=device, non_blocking=True)
                yb = yb.to(device=device, non_blocking=True)
                if noise_std > 0.0:
                    xb = xb + (noise_std * torch.randn_like(xb))
                if label_smoothing > 0.0:
                    yb = yb * (1.0 - label_smoothing) + (0.5 * label_smoothing)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            valid_pred_epoch = _predict_with_torch_model(
                model,
                x_valid_s,
                max(4096, int(cfg.deep_eval_batch_size)),
                device,
            )
            valid_auc_epoch = float(roc_auc_score(y_valid, valid_pred_epoch))
            if valid_auc_epoch > best_auc_local + 1e-5:
                best_auc_local = valid_auc_epoch
                best_state_local = copy.deepcopy(model.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    break

        model.load_state_dict(best_state_local)
        infer_bs = max(4096, int(cfg.deep_eval_batch_size))
        train_pred_local = _predict_with_torch_model(model, x_train_s, infer_bs, device)
        valid_pred_local = _predict_with_torch_model(model, x_valid_s, infer_bs, device)
        test_pred_local = _predict_with_torch_model(model, x_test_s, infer_bs, device)
        valid_auc_local = float(roc_auc_score(y_valid, valid_pred_local))
        return train_pred_local, valid_pred_local, test_pred_local, valid_auc_local

    train_pred, valid_pred, test_pred, valid_auc = _train_once(dict(params))

    # If the model learned an inverted ranking, flip it so AUC stays >= 0.5.
    if valid_auc < 0.5:
        inv_auc = float(roc_auc_score(y_valid, 1.0 - valid_pred))
        if inv_auc > valid_auc:
            train_pred = 1.0 - train_pred
            valid_pred = 1.0 - valid_pred
            test_pred = 1.0 - test_pred
            valid_auc = inv_auc

    # Rescue pass for collapsed/underfit runs.
    if cfg.deep_rescue_on_low_auc and valid_auc < float(cfg.deep_rescue_auc_floor):
        rescue_params = dict(params)
        rescue_params["seed"] = int(params["seed"]) + 7013
        rescue_params["epochs"] = max(int(params["epochs"]), int(cfg.deep_rescue_epochs))
        rescue_params["batch_size"] = max(1024, int(params["batch_size"]) // 2)
        rescue_params["hidden_dim"] = max(512, int(params["hidden_dim"]))
        rescue_params["depth"] = max(2, min(4, int(params["depth"])))
        rescue_params["dropout"] = min(0.10, float(params["dropout"]))
        rescue_params["learning_rate"] = max(2.5e-4, float(params["learning_rate"]) * 0.80)
        rescue_params["weight_decay"] = max(1e-6, float(params["weight_decay"]) * 0.80)
        rs_train, rs_valid, rs_test, rs_auc = _train_once(rescue_params)
        if rs_auc > valid_auc + 1e-4:
            train_pred, valid_pred, test_pred, valid_auc = rs_train, rs_valid, rs_test, rs_auc

    backend = "torch_cuda"
    return train_pred, valid_pred, test_pred, backend


def _run_deep_level2_stack(
    cfg: BaselineConfig,
    x_train_encoded: pd.DataFrame,
    x_valid_encoded: pd.DataFrame,
    x_test_encoded: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    paths,
    suite_metrics: list[dict[str, object]],
    model_counter: int,
    level2_train_cols: dict[str, np.ndarray],
    level2_valid_cols: dict[str, np.ndarray],
    level2_test_cols: dict[str, np.ndarray],
) -> tuple[int, tuple[str, ...]]:
    families = _resolve_available_deep_families(cfg.deep_model_families)
    x_train_base = x_train_encoded.to_numpy(dtype=np.float32)
    x_valid_base = x_valid_encoded.to_numpy(dtype=np.float32)
    x_test_base = x_test_encoded.to_numpy(dtype=np.float32)
    family_inputs_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for family_idx, family in enumerate(families):
        if family not in family_inputs_cache:
            family_inputs_cache[family] = _augment_deep_family_features(
                family=family,
                x_train=x_train_base,
                x_valid=x_valid_base,
                x_test=x_test_base,
                seed=cfg.random_state + (1000 * family_idx),
            )
        x_train_family, x_valid_family, x_test_family = family_inputs_cache[family]
        variants = _deep_variant_grid(
            seed=cfg.random_state + (100 * family_idx),
            n_models=cfg.deep_models_per_family,
            cfg=cfg,
        )
        for model_idx, params in enumerate(variants, start=1):
            model_name = f"dl_{family}_m{model_idx}"
            pred_col = f"l1_pred__dl_{family}__m{model_idx}"
            try:
                train_pred, valid_pred, test_pred, backend = _fit_deep_family_model(
                    cfg=cfg,
                    family=family,
                    params=params,
                    x_train=x_train_family,
                    y_train=y_train,
                    x_valid=x_valid_family,
                    y_valid=y_valid,
                    x_test=x_test_family,
                )
            except Exception as exc:
                print(f"[DEEP_SUITE][WARN] {model_name} failed and was skipped: {exc}")
                continue

            valid_auc = roc_auc_score(y_valid, valid_pred)
            if valid_auc < float(cfg.min_deep_valid_auc_keep):
                print(
                    f"[DEEP_SUITE] {model_name} dropped | valid_auc={valid_auc:.6f} "
                    f"< min_deep_valid_auc_keep={cfg.min_deep_valid_auc_keep:.3f}"
                )
                continue

            level2_train_cols[pred_col] = train_pred.astype(np.float32)
            level2_valid_cols[pred_col] = valid_pred.astype(np.float32)
            level2_test_cols[pred_col] = test_pred.astype(np.float32)

            pred_tag = f"{model_name}_{_pred_run_tag(cfg)}"
            np.save(paths.oof_dir / f"oof_{pred_tag}.npy", valid_pred.astype(np.float32))
            np.save(paths.pred_dir / f"pred_{pred_tag}.npy", test_pred.astype(np.float32))

            run_metrics = {
                "model_name": model_name,
                "library": f"dl_{family}",
                "backend": backend,
                "feature_set_name": cfg.feature_set_name,
                "valid_auc": float(valid_auc),
                "params": params,
            }
            suite_metrics.append(run_metrics)
            model_counter += 1
            print(f"[DEEP_SUITE] {model_name} | backend={backend} | valid_auc={valid_auc:.6f}")

    return model_counter, families


def _xgboost_variant_grid(seed: int, n_models: int) -> list[dict[str, object]]:
    presets = [
        {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 140, "learning_rate": 0.04, "max_depth": 5, "subsample": 0.90, "colsample_bytree": 0.80},
        {"n_estimators": 110, "learning_rate": 0.07, "max_depth": 5, "subsample": 0.80, "colsample_bytree": 0.90},
        {"n_estimators": 160, "learning_rate": 0.035, "max_depth": 4, "subsample": 0.75, "colsample_bytree": 0.95},
        {"n_estimators": 130, "learning_rate": 0.06, "max_depth": 3, "subsample": 0.95, "colsample_bytree": 0.75},
    ]
    out: list[dict[str, object]] = []
    for idx in range(max(1, n_models)):
        base = presets[idx % len(presets)].copy()
        base["random_state"] = seed + idx
        out.append(base)
    return out


def _xgboost_dart_variant_grid(seed: int, n_models: int) -> list[dict[str, object]]:
    presets = [
        {
            "n_estimators": 220,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "booster": "dart",
            "rate_drop": 0.10,
            "skip_drop": 0.40,
        },
        {
            "n_estimators": 260,
            "learning_rate": 0.04,
            "max_depth": 5,
            "subsample": 0.80,
            "colsample_bytree": 0.90,
            "booster": "dart",
            "rate_drop": 0.08,
            "skip_drop": 0.30,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.06,
            "max_depth": 4,
            "subsample": 0.90,
            "colsample_bytree": 0.80,
            "booster": "dart",
            "rate_drop": 0.12,
            "skip_drop": 0.50,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.035,
            "max_depth": 5,
            "subsample": 0.78,
            "colsample_bytree": 0.92,
            "booster": "dart",
            "rate_drop": 0.06,
            "skip_drop": 0.25,
        },
        {
            "n_estimators": 240,
            "learning_rate": 0.045,
            "max_depth": 3,
            "subsample": 0.95,
            "colsample_bytree": 0.75,
            "booster": "dart",
            "rate_drop": 0.10,
            "skip_drop": 0.35,
        },
    ]
    out: list[dict[str, object]] = []
    for idx in range(max(1, n_models)):
        base = presets[idx % len(presets)].copy()
        base["random_state"] = seed + idx
        out.append(base)
    return out


def _lightgbm_variant_grid(seed: int, n_models: int) -> list[dict[str, object]]:
    presets = [
        {"n_estimators": 120, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 40, "subsample": 0.85, "colsample_bytree": 0.80},
        {"n_estimators": 150, "learning_rate": 0.04, "num_leaves": 77, "min_child_samples": 56, "subsample": 0.80, "colsample_bytree": 0.90},
        {"n_estimators": 110, "learning_rate": 0.07, "num_leaves": 45, "min_child_samples": 72, "subsample": 0.90, "colsample_bytree": 0.80},
        {"n_estimators": 170, "learning_rate": 0.03, "num_leaves": 95, "min_child_samples": 90, "subsample": 0.75, "colsample_bytree": 0.75},
        {"n_estimators": 130, "learning_rate": 0.06, "num_leaves": 55, "min_child_samples": 48, "subsample": 0.95, "colsample_bytree": 0.95},
    ]
    out: list[dict[str, object]] = []
    for idx in range(max(1, n_models)):
        base = presets[idx % len(presets)].copy()
        base["random_state"] = seed + idx
        out.append(base)
    return out


def _catboost_variant_grid(seed: int, n_models: int) -> list[dict[str, object]]:
    presets = [
        {"iterations": 160, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 4.0, "random_strength": 1.0, "bagging_temperature": 0.2},
        {"iterations": 220, "learning_rate": 0.04, "depth": 5, "l2_leaf_reg": 6.0, "random_strength": 1.8, "bagging_temperature": 0.4},
        {"iterations": 140, "learning_rate": 0.07, "depth": 6, "l2_leaf_reg": 8.0, "random_strength": 2.4, "bagging_temperature": 0.0},
        {"iterations": 240, "learning_rate": 0.035, "depth": 4, "l2_leaf_reg": 10.0, "random_strength": 2.9, "bagging_temperature": 0.3},
        {"iterations": 180, "learning_rate": 0.06, "depth": 6, "l2_leaf_reg": 5.0, "random_strength": 1.4, "bagging_temperature": 0.1},
    ]
    out: list[dict[str, object]] = []
    for idx in range(max(1, n_models)):
        base = presets[idx % len(presets)].copy()
        base["random_seed"] = seed + idx
        out.append(base)
    return out


def _ydf_style_variant_grid(seed: int, n_models: int) -> list[dict[str, object]]:
    presets = [
        {"n_estimators": 110, "learning_rate": 0.06, "max_depth": 2, "subsample": 0.90, "max_features": 0.8},
        {"n_estimators": 140, "learning_rate": 0.05, "max_depth": 2, "subsample": 0.85, "max_features": 0.9},
        {"n_estimators": 100, "learning_rate": 0.07, "max_depth": 2, "subsample": 0.95, "max_features": 0.7},
        {"n_estimators": 150, "learning_rate": 0.04, "max_depth": 2, "subsample": 0.80, "max_features": 1.0},
        {"n_estimators": 120, "learning_rate": 0.055, "max_depth": 2, "subsample": 0.88, "max_features": 0.85},
    ]
    out: list[dict[str, object]] = []
    for idx in range(max(1, n_models)):
        base = presets[idx % len(presets)].copy()
        base["random_state"] = seed + idx
        out.append(base)
    return out


def _cuml_rf_variant_grid(seed: int, n_models: int) -> list[dict[str, object]]:
    presets = [
        {"n_estimators": 150, "max_depth": 14, "max_features": "sqrt", "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": 18, "max_features": 0.6, "min_samples_leaf": 2},
        {"n_estimators": 130, "max_depth": 12, "max_features": 0.8, "min_samples_leaf": 1},
        {"n_estimators": 240, "max_depth": 16, "max_features": "log2", "min_samples_leaf": 3},
        {"n_estimators": 170, "max_depth": 13, "max_features": 0.7, "min_samples_leaf": 2},
    ]
    out: list[dict[str, object]] = []
    for idx in range(max(1, n_models)):
        base = presets[idx % len(presets)].copy()
        base["random_state"] = seed + idx
        out.append(base)
    return out


def _cuml_et_variant_grid(seed: int, n_models: int) -> list[dict[str, object]]:
    presets = [
        {"n_estimators": 150, "max_depth": 14, "max_features": "sqrt", "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": 18, "max_features": 0.6, "min_samples_leaf": 2},
        {"n_estimators": 130, "max_depth": 12, "max_features": 0.8, "min_samples_leaf": 1},
        {"n_estimators": 240, "max_depth": 16, "max_features": "log2", "min_samples_leaf": 3},
        {"n_estimators": 170, "max_depth": 13, "max_features": 0.7, "min_samples_leaf": 2},
    ]
    out: list[dict[str, object]] = []
    for idx in range(max(1, n_models)):
        base = presets[idx % len(presets)].copy()
        base["random_state"] = seed + idx
        out.append(base)
    return out


def _tree_variant_grid(library_name: str, seed: int, n_models: int) -> list[dict[str, object]]:
    if library_name == "xgboost":
        return _xgboost_variant_grid(seed=seed, n_models=n_models)
    if library_name == "xgboost_dart":
        return _xgboost_dart_variant_grid(seed=seed, n_models=n_models)
    if library_name == "lightgbm":
        return _lightgbm_variant_grid(seed=seed, n_models=n_models)
    if library_name == "catboost":
        return _catboost_variant_grid(seed=seed, n_models=n_models)
    if library_name == "catboost_native":
        return _catboost_variant_grid(seed=seed, n_models=n_models)
    if library_name == "ydf":
        return _ydf_style_variant_grid(seed=seed, n_models=n_models)
    if library_name == "cuml_rf":
        return _cuml_rf_variant_grid(seed=seed, n_models=n_models)
    if library_name == "cuml_et":
        return _cuml_et_variant_grid(seed=seed, n_models=n_models)
    raise ValueError(f"Unsupported tree library '{library_name}'.")


def _subsample_rows_for_tuning(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if max_rows <= 0 or len(y_train) <= max_rows:
        return x_train, y_train
    idx_all = np.arange(len(y_train))
    y_arr = np.asarray(y_train).astype(np.int32)
    try:
        idx_keep, _ = train_test_split(
            idx_all,
            train_size=max_rows,
            random_state=random_state,
            stratify=y_arr,
        )
    except Exception:
        rng = np.random.default_rng(random_state)
        idx_keep = rng.choice(idx_all, size=max_rows, replace=False)
    idx_keep = np.asarray(idx_keep, dtype=np.int64)
    return x_train.iloc[idx_keep].copy(), y_arr[idx_keep]


def _optuna_suggest_tree_params(
    library_name: str,
    trial,
    base_seed: int,
) -> dict[str, object]:
    if library_name in {"xgboost", "xgboost_dart"}:
        params = {
            "n_estimators": int(trial.suggest_int("n_estimators", 110, 360)),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.02, 0.12, log=True)),
            "max_depth": int(trial.suggest_int("max_depth", 3, 8)),
            "subsample": float(trial.suggest_float("subsample", 0.70, 1.00)),
            "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.70, 1.00)),
            "min_child_weight": float(trial.suggest_float("min_child_weight", 0.5, 12.0, log=True)),
            "gamma": float(trial.suggest_float("gamma", 0.0, 3.0)),
            "reg_lambda": float(trial.suggest_float("reg_lambda", 0.5, 15.0, log=True)),
            "random_state": int(base_seed + trial.number),
        }
        if library_name == "xgboost_dart":
            params.update(
                {
                    "booster": "dart",
                    "rate_drop": float(trial.suggest_float("rate_drop", 0.03, 0.25)),
                    "skip_drop": float(trial.suggest_float("skip_drop", 0.10, 0.60)),
                }
            )
        return params
    if library_name == "catboost":
        return {
            "iterations": int(trial.suggest_int("iterations", 120, 420)),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.02, 0.12, log=True)),
            "depth": int(trial.suggest_int("depth", 4, 9)),
            "l2_leaf_reg": float(trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True)),
            "random_strength": float(trial.suggest_float("random_strength", 0.1, 5.0)),
            "bagging_temperature": float(trial.suggest_float("bagging_temperature", 0.0, 1.0)),
            "random_seed": int(base_seed + trial.number),
        }
    if library_name == "lightgbm":
        return {
            "n_estimators": int(trial.suggest_int("n_estimators", 120, 420)),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.02, 0.12, log=True)),
            "num_leaves": int(trial.suggest_int("num_leaves", 31, 191)),
            "min_child_samples": int(trial.suggest_int("min_child_samples", 16, 140)),
            "subsample": float(trial.suggest_float("subsample", 0.70, 1.00)),
            "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.70, 1.00)),
            "reg_lambda": float(trial.suggest_float("reg_lambda", 0.2, 12.0, log=True)),
            "random_state": int(base_seed + trial.number),
        }
    return {}


def _optuna_tuned_tree_variants(
    *,
    cfg: BaselineConfig,
    library_name: str,
    seed: int,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> list[dict[str, object]]:
    if not cfg.use_tree_optuna or cfg.tree_optuna_trials_per_library <= 0:
        return []
    if optuna is None or TPESampler is None:
        print(f"[TREE_SUITE][WARN] Optuna unavailable; skipping tuning for '{library_name}'.")
        return []
    if library_name not in {"xgboost", "xgboost_dart", "catboost", "lightgbm"}:
        return []
    ok, reason = _gpu_preflight_library(library_name)
    if not ok:
        print(f"[TREE_SUITE][WARN] Optuna skipped for '{library_name}': {reason}")
        return []

    x_train_tune, y_train_tune = _subsample_rows_for_tuning(
        x_train=x_train,
        y_train=y_train,
        max_rows=int(cfg.tree_optuna_train_max_rows),
        random_state=seed + 17,
    )
    tune_sample_weight = None
    if cfg.use_balanced_sample_weight:
        tune_sample_weight, _ = _compute_balanced_sample_weight(
            y_train_tune,
            min_ratio=cfg.min_sample_weight_ratio,
            max_ratio=cfg.max_sample_weight_ratio,
        )
    x_test_dummy = x_valid
    error_counts: dict[str, int] = {}
    max_error_logs = 3

    def _objective(trial) -> float:
        params = _optuna_suggest_tree_params(library_name=library_name, trial=trial, base_seed=seed + 1000)
        if not params:
            return 0.0
        try:
            _, valid_pred, _, _ = _fit_tree_library_model(
                library_name=library_name,
                params=params,
                x_train=x_train_tune,
                y_train=y_train_tune,
                x_valid=x_valid,
                x_test=x_test_dummy,
                random_state=seed + trial.number,
                sample_weight=tune_sample_weight,
            )
            score = float(_score_predictions(y_valid, valid_pred))
            trial.set_user_attr("params", params)
            return score
        except Exception as exc:
            err = _short_exc(exc)
            trial.set_user_attr("error", err)
            error_counts[err] = error_counts.get(err, 0) + 1
            if sum(error_counts.values()) <= max_error_logs:
                print(f"[TREE_SUITE][WARN] Optuna trial failed for '{library_name}': {err}")
            return 0.0

    timeout = None
    if int(cfg.tree_optuna_timeout_sec) > 0:
        timeout = int(cfg.tree_optuna_timeout_sec)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed + 701),
    )
    study.optimize(
        _objective,
        n_trials=int(cfg.tree_optuna_trials_per_library),
        timeout=timeout,
        show_progress_bar=False,
    )
    failed_trials = [t for t in study.trials if "error" in t.user_attrs]
    if failed_trials:
        top_errors = sorted(error_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        err_msg = "; ".join(f"{msg} (x{count})" for msg, count in top_errors)
        print(
            f"[TREE_SUITE][WARN] Optuna failures for '{library_name}': "
            f"{len(failed_trials)}/{len(study.trials)} trials. Top errors: {err_msg}"
        )
        if len(failed_trials) == len(study.trials):
            print(f"[TREE_SUITE][WARN] All Optuna trials failed for '{library_name}', skipping tuned variants.")
            return []

    tuned: list[dict[str, object]] = []
    seen: set[str] = set()
    max_models = max(0, int(cfg.tree_optuna_max_tuned_models_per_library))
    for tr in sorted(study.trials, key=lambda t: float(t.value or 0.0), reverse=True):
        if len(tuned) >= max_models:
            break
        params = tr.user_attrs.get("params")
        if not isinstance(params, dict):
            continue
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        tuned.append(dict(params))

    if tuned:
        best = float(study.best_value) if study.best_value is not None else float("nan")
        print(
            f"[TREE_SUITE] Optuna tuned '{library_name}' | trials={cfg.tree_optuna_trials_per_library} "
            f"| keep={len(tuned)} | best_valid_auc={best:.6f}"
        )
    return tuned


def _resolve_available_gpu_libraries(libraries: tuple[str, ...], *, strict_gpu_only: bool = True) -> tuple[str, ...]:
    known = {"xgboost", "xgboost_dart", "lightgbm", "catboost", "catboost_native", "ydf", "cuml_rf", "cuml_et"}
    unknown = sorted([name for name in libraries if name not in known])
    if unknown:
        raise ValueError(f"Unsupported tree libraries requested: {', '.join(unknown)}")

    if cp is None:
        raise RuntimeError("GPU-only mode requires cupy, but cupy is not installed.")
    try:
        gpu_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(f"GPU-only mode requires a working CUDA runtime: {_short_exc(exc)}") from exc
    if gpu_count < 1:
        raise RuntimeError("GPU-only mode requires at least one visible CUDA device.")

    available: list[str] = []
    skipped: list[str] = []
    for name in libraries:
        if name == "ydf":
            skipped.append("ydf (CPU-only backend in this project)")
            if strict_gpu_only:
                continue
            available.append(name)
            continue

        ok, reason = _gpu_preflight_library(name)
        if ok:
            available.append(name)
        else:
            skipped.append(f"{name} ({reason})")

    if skipped:
        print(f"[TREE_SUITE][WARN] Skipping non-GPU-ready libraries: {', '.join(skipped)}")
    if not available:
        raise RuntimeError(
            "None of the requested tree libraries are GPU-ready in this environment. "
            "Install/fix at least one GPU backend (xgboost/catboost/lightgbm/cuml)."
        )
    return tuple(available)


def _prepare_catboost_native_inputs(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], list[str]]:
    candidate_cols = [c for c in train_df.columns if c not in {target_col, "cv_fold"}]
    x_train_raw = train_df[candidate_cols].copy()
    x_valid_raw = valid_df[candidate_cols].copy()
    x_test_raw = test_df[candidate_cols].copy()
    cat_feature_idx: list[int] = []

    for idx, col in enumerate(candidate_cols):
        if pd.api.types.is_numeric_dtype(x_train_raw[col]):
            x_train_raw[col] = pd.to_numeric(x_train_raw[col], errors="coerce").fillna(0.0).astype(np.float32)
            x_valid_raw[col] = pd.to_numeric(x_valid_raw[col], errors="coerce").fillna(0.0).astype(np.float32)
            x_test_raw[col] = pd.to_numeric(x_test_raw[col], errors="coerce").fillna(0.0).astype(np.float32)
            continue
        cat_feature_idx.append(idx)
        x_train_raw[col] = x_train_raw[col].astype("string").fillna("__NA__").astype(str)
        x_valid_raw[col] = x_valid_raw[col].astype("string").fillna("__NA__").astype(str)
        x_test_raw[col] = x_test_raw[col].astype("string").fillna("__NA__").astype(str)

    return x_train_raw, x_valid_raw, x_test_raw, cat_feature_idx, candidate_cols


def _fit_tree_library_model(
    library_name: str,
    params: dict[str, object],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
    random_state: int,
    sample_weight: np.ndarray | None = None,
    raw_train_df: pd.DataFrame | None = None,
    raw_valid_df: pd.DataFrame | None = None,
    raw_test_df: pd.DataFrame | None = None,
    target_col: str = "Irrigation_Need",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    y_train_np = y_train.astype(np.int32)
    sample_weight_np = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    n_classes = int(np.max(y_train_np)) + 1

    if library_name == "catboost_native":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed.")
        if raw_train_df is None or raw_valid_df is None or raw_test_df is None:
            raise RuntimeError("catboost_native requires raw split DataFrames.")

        x_train_raw, x_valid_raw, x_test_raw, cat_feature_idx, _ = _prepare_catboost_native_inputs(
            train_df=raw_train_df,
            valid_df=raw_valid_df,
            test_df=raw_test_df,
            target_col=target_col,
        )
        cat_params = dict(
            task_type="GPU",
            devices="0",
            allow_writing_files=False,
            verbose=False,
            one_hot_max_size=64,
            **params,
        )
        if n_classes > 2:
            cat_params.update({"loss_function": "MultiClass", "eval_metric": "MultiClass"})
        else:
            cat_params.update({"loss_function": "Logloss", "eval_metric": "AUC"})
        model = CatBoostClassifier(**cat_params)
        model.fit(
            x_train_raw,
            y_train_np,
            cat_features=cat_feature_idx,
            sample_weight=sample_weight_np,
        )
        return (
            _predict_score(model, x_train_raw, n_classes=n_classes),
            _predict_score(model, x_valid_raw, n_classes=n_classes),
            _predict_score(model, x_test_raw, n_classes=n_classes),
            "catboost_native_gpu",
        )

    if cp is None:
        raise RuntimeError("GPU-only mode requires cupy.")

    x_train_gpu = _to_cupy_frame(x_train)
    x_valid_gpu = _to_cupy_frame(x_valid)
    x_test_gpu = _to_cupy_frame(x_test)
    y_train_gpu = _to_cupy_target(y_train)

    if library_name in {"xgboost", "xgboost_dart"}:
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed.")
        xgb_params: dict[str, object] = {
            "device": "cuda",
            "tree_method": "hist",
            "n_jobs": -1,
            "reg_lambda": 1.0,
            "min_child_weight": 1.0,
            "gamma": 0.0,
        }
        xgb_params.update(params)
        xgb_params.setdefault("random_state", int(random_state))
        if n_classes > 2:
            xgb_params.update(
                {
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "num_class": int(n_classes),
                }
            )
        else:
            xgb_params.update({"objective": "binary:logistic", "eval_metric": "auc"})
        model = XGBClassifier(**xgb_params)
        try:
            model.fit(x_train_gpu, y_train_np, sample_weight=sample_weight_np)
            return (
                _predict_score(model, x_train_gpu, n_classes=n_classes),
                _predict_score(model, x_valid_gpu, n_classes=n_classes),
                _predict_score(model, x_test_gpu, n_classes=n_classes),
                "xgboost_dart_gpu_cuda" if library_name == "xgboost_dart" else "xgboost_gpu_cuda",
            )
        except Exception as gpu_exc:
            # Some environments fail with CuPy matrix input despite working GPU training.
            x_train_np = cp.asnumpy(x_train_gpu)
            x_valid_np = cp.asnumpy(x_valid_gpu)
            x_test_np = cp.asnumpy(x_test_gpu)
            model = XGBClassifier(**xgb_params)
            try:
                model.fit(x_train_np, y_train_np, sample_weight=sample_weight_np)
                return (
                    _predict_score(model, x_train_np, n_classes=n_classes),
                    _predict_score(model, x_valid_np, n_classes=n_classes),
                    _predict_score(model, x_test_np, n_classes=n_classes),
                    (
                        "xgboost_dart_gpu_cuda_numpy_input_fallback"
                        if library_name == "xgboost_dart"
                        else "xgboost_gpu_cuda_numpy_input_fallback"
                    ),
                )
            except Exception as np_exc:
                raise RuntimeError(
                    f"XGBoost GPU fit failed (cupy input: {_short_exc(gpu_exc)}; numpy input: {_short_exc(np_exc)})"
                ) from np_exc

    if library_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed.")
        lgb_params = dict(
            device_type="gpu",
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        if n_classes > 2:
            lgb_params.update({"objective": "multiclass", "num_class": int(n_classes)})
        else:
            lgb_params.update({"objective": "binary"})
        model = LGBMClassifier(
            **lgb_params,
        )
        # LightGBM GPU expects CPU numpy inputs.
        x_train_np = cp.asnumpy(x_train_gpu)
        x_valid_np = cp.asnumpy(x_valid_gpu)
        x_test_np = cp.asnumpy(x_test_gpu)
        model.fit(x_train_np, y_train_np, sample_weight=sample_weight_np)
        return (
            _predict_score(model, x_train_np, n_classes=n_classes),
            _predict_score(model, x_valid_np, n_classes=n_classes),
            _predict_score(model, x_test_np, n_classes=n_classes),
            "lightgbm_gpu",
        )

    if library_name == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed.")
        cat_params = dict(
            task_type="GPU",
            devices="0",
            allow_writing_files=False,
            verbose=False,
            **params,
        )
        if n_classes > 2:
            cat_params.update({"loss_function": "MultiClass", "eval_metric": "TotalF1"})
        else:
            cat_params.update({"loss_function": "Logloss", "eval_metric": "AUC"})
        model = CatBoostClassifier(
            **cat_params,
        )
        x_train_np = cp.asnumpy(x_train_gpu)
        x_valid_np = cp.asnumpy(x_valid_gpu)
        x_test_np = cp.asnumpy(x_test_gpu)
        model.fit(x_train_np, y_train_np, sample_weight=sample_weight_np)
        return (
            _predict_score(model, x_train_np, n_classes=n_classes),
            _predict_score(model, x_valid_np, n_classes=n_classes),
            _predict_score(model, x_test_np, n_classes=n_classes),
            "catboost_gpu",
        )

    if library_name == "cuml_rf":
        if CuMLRandomForestClassifier is None:
            raise RuntimeError("cuml RandomForestClassifier is unavailable.")
        model = CuMLRandomForestClassifier(**params)
        model.fit(x_train_gpu, y_train_gpu)
        return (
            _predict_score(model, x_train_gpu, n_classes=n_classes),
            _predict_score(model, x_valid_gpu, n_classes=n_classes),
            _predict_score(model, x_test_gpu, n_classes=n_classes),
            "cuml_random_forest_gpu",
        )

    if library_name == "cuml_et":
        if CuMLExtraTreesClassifier is None:
            raise RuntimeError("cuml ExtraTreesClassifier is unavailable.")
        model = CuMLExtraTreesClassifier(**params)
        model.fit(x_train_gpu, y_train_gpu)
        return (
            _predict_score(model, x_train_gpu, n_classes=n_classes),
            _predict_score(model, x_valid_gpu, n_classes=n_classes),
            _predict_score(model, x_test_gpu, n_classes=n_classes),
            "cuml_extra_trees_gpu",
        )

    raise ValueError(f"Unsupported or CPU-only tree library '{library_name}' in strict GPU mode.")


def _save_tree_suite_artifacts(
    out_dir: Path,
    cfg: BaselineConfig,
    y_valid: np.ndarray,
    level2_train: pd.DataFrame,
    level2_valid: pd.DataFrame,
    level2_test: pd.DataFrame,
    suite_metrics: list[dict[str, object]],
    libraries_requested: list[str],
    sample_weight_by_class: dict[int, float] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_parquet(level2_train, out_dir / "level2_train.parquet")
    _save_parquet(level2_valid, out_dir / "level2_valid.parquet")
    _save_parquet(level2_test, out_dir / "level2_test.parquet")

    if suite_metrics:
        by_library: dict[str, list[float]] = {}
        for row in suite_metrics:
            by_library.setdefault(str(row["library"]), []).append(float(row["valid_auc"]))
        library_summary = {
            lib: {
                "models": int(len(scores)),
                "mean_valid_auc": float(np.mean(scores)),
                "best_valid_auc": float(np.max(scores)),
            }
            for lib, scores in by_library.items()
        }
    else:
        library_summary = {}

    summary = {
        "stage": "tree_level_stack",
        "feature_set_name": cfg.feature_set_name,
        "level1_definition": "original_data_plus_engineered_features",
        "level2_definition": "level1_plus_tree_and_deep_model_outputs",
        "level3_definition": "xgboost_on_level1_plus_level2_outputs",
        "level4_definition": "l2_logistic_meta_model_on_level2_plus_level3_outputs",
        "models_per_library": cfg.models_per_library,
        "use_tree_optuna": bool(cfg.use_tree_optuna),
        "tree_optuna_trials_per_library": int(cfg.tree_optuna_trials_per_library),
        "tree_optuna_max_tuned_models_per_library": int(cfg.tree_optuna_max_tuned_models_per_library),
        "min_tree_valid_auc_keep": float(cfg.min_tree_valid_auc_keep),
        "min_deep_valid_auc_keep": float(cfg.min_deep_valid_auc_keep),
        "level2_keep_top_models_for_meta": int(cfg.level2_keep_top_models_for_meta),
        "level2_keep_min_models_for_meta": int(cfg.level2_keep_min_models_for_meta),
        "level2_keep_model_auc_gap": float(cfg.level2_keep_model_auc_gap),
        "min_oof_auc_for_selection": float(cfg.min_oof_auc_for_selection),
        "stacking_min_models": int(cfg.stacking_min_models),
        "use_balanced_sample_weight": bool(cfg.use_balanced_sample_weight),
        "min_sample_weight_ratio": float(cfg.min_sample_weight_ratio),
        "max_sample_weight_ratio": float(cfg.max_sample_weight_ratio),
        "sample_weight_by_class": {
            str(k): float(v) for k, v in (sample_weight_by_class or {}).items()
        },
        "enable_class_weight_calibration": bool(cfg.enable_class_weight_calibration),
        "class_weight_calibration_grid": [float(v) for v in cfg.class_weight_calibration_grid],
        "class_weight_calibration_random_trials": int(cfg.class_weight_calibration_random_trials),
        "libraries_requested": libraries_requested,
        "libraries_completed": sorted(list({str(m["library"]) for m in suite_metrics})),
        "library_summary": library_summary,
        "n_models_trained": int(len(suite_metrics)),
        "n_valid": int(len(y_valid)),
        "model_metrics": suite_metrics,
    }
    (out_dir / "tree_suite_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _model_name_to_l1_prefix(model_name: str) -> str | None:
    name = str(model_name).strip()
    if "_m" not in name:
        return None
    library_part, model_idx = name.rsplit("_m", 1)
    if not model_idx.isdigit():
        return None
    return f"l1_pred__{library_part}__m{int(model_idx)}"


def _select_level2_prediction_columns_for_meta(
    cfg: BaselineConfig,
    suite_metrics: list[dict[str, object]],
    level2_columns: list[str],
) -> tuple[list[str], dict[str, object]]:
    all_pred_cols = sorted([c for c in level2_columns if c.startswith("l1_pred__")])
    info: dict[str, object] = {
        "total_pred_columns": int(len(all_pred_cols)),
        "selected_pred_columns": int(len(all_pred_cols)),
        "total_candidate_models": 0,
        "selected_models": 0,
        "best_model_auc": None,
        "auc_gap": float(cfg.level2_keep_model_auc_gap),
        "selected_model_names": [],
    }
    if not all_pred_cols or not suite_metrics:
        return all_pred_cols, info

    model_rows: list[dict[str, object]] = []
    for row in suite_metrics:
        prefix = _model_name_to_l1_prefix(str(row.get("model_name", "")))
        if not prefix:
            continue
        try:
            valid_auc = float(row.get("valid_auc"))
        except Exception:
            continue
        model_pred_cols = [c for c in all_pred_cols if c == prefix or c.startswith(prefix + "__")]
        if not model_pred_cols:
            continue
        model_rows.append(
            {
                "model_name": str(row.get("model_name", "")),
                "valid_auc": valid_auc,
                "cols": model_pred_cols,
            }
        )

    if not model_rows:
        return all_pred_cols, info

    model_rows.sort(key=lambda x: float(x["valid_auc"]), reverse=True)
    best_auc = float(model_rows[0]["valid_auc"])
    auc_gap = max(0.0, float(cfg.level2_keep_model_auc_gap))
    top_cap = max(0, int(cfg.level2_keep_top_models_for_meta))
    min_keep = max(1, int(cfg.level2_keep_min_models_for_meta))

    kept_rows = [row for row in model_rows if float(row["valid_auc"]) >= (best_auc - auc_gap - 1e-12)]
    if top_cap > 0 and len(kept_rows) > top_cap:
        kept_rows = kept_rows[:top_cap]

    target_min = min(min_keep, len(model_rows))
    if len(kept_rows) < target_min:
        kept_names = {str(row["model_name"]) for row in kept_rows}
        for row in model_rows:
            if str(row["model_name"]) in kept_names:
                continue
            kept_rows.append(row)
            kept_names.add(str(row["model_name"]))
            if len(kept_rows) >= target_min:
                break

    selected_cols = sorted({col for row in kept_rows for col in row["cols"]})
    if not selected_cols:
        selected_cols = all_pred_cols

    info.update(
        {
            "selected_pred_columns": int(len(selected_cols)),
            "total_candidate_models": int(len(model_rows)),
            "selected_models": int(len(kept_rows)),
            "best_model_auc": float(best_auc),
            "selected_model_names": [str(row["model_name"]) for row in kept_rows],
        }
    )
    return selected_cols, info


def _train_level3_xgb(
    cfg: BaselineConfig,
    level2_train: pd.DataFrame,
    level2_valid: pd.DataFrame,
    level2_test: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    out_dir: Path,
    sample_weight: np.ndarray | None = None,
    sample_weight_by_class: dict[int, float] | None = None,
) -> dict[str, object]:
    x_train_l3, x_valid_l3, x_test_l3, used_cols = _encode_level_features(
        train_df=level2_train,
        valid_df=level2_valid,
        test_df=level2_test,
        target_col=cfg.target_col,
    )

    if XGBClassifier is None:
        raise RuntimeError("Level-3 strict GPU mode requires xgboost.")
    if cp is None:
        raise RuntimeError("Level-3 strict GPU mode requires cupy.")

    n_classes = int(np.max(y_train)) + 1
    xgb_params: dict[str, object] = {
        "device": "cuda",
        "tree_method": "hist",
        "n_jobs": -1,
        "n_estimators": cfg.level3_xgb_n_estimators,
        "learning_rate": cfg.level3_xgb_learning_rate,
        "max_depth": cfg.level3_xgb_max_depth,
        "subsample": cfg.level3_xgb_subsample,
        "colsample_bytree": cfg.level3_xgb_colsample_bytree,
        "random_state": cfg.random_state,
    }
    if n_classes > 2:
        xgb_params.update({"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": int(n_classes)})
    else:
        xgb_params.update({"objective": "binary:logistic", "eval_metric": "auc"})
    level3_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    x_valid_gpu = _to_cupy_frame(x_valid_l3)
    x_test_gpu = _to_cupy_frame(x_test_l3)

    use_cv = bool(cfg.level3_cv_enabled and int(cfg.level3_cv_folds) > 1)
    cv_fold_scores: list[float] = []
    if use_cv:
        cv = StratifiedKFold(
            n_splits=max(2, int(cfg.level3_cv_folds)),
            shuffle=True,
            random_state=cfg.random_state,
        )
        valid_fold_preds: list[np.ndarray] = []
        test_fold_preds: list[np.ndarray] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_train_l3, y_train)):
            fold_params = dict(xgb_params)
            fold_params["random_state"] = int(cfg.random_state + fold_idx)
            model = XGBClassifier(**fold_params)

            x_tr_gpu = _to_cupy_frame(x_train_l3.iloc[tr_idx])
            x_va_gpu = _to_cupy_frame(x_train_l3.iloc[va_idx])
            y_tr = y_train[tr_idx].astype(np.int32)
            sw_tr = None if level3_weight is None else level3_weight[tr_idx]
            model.fit(x_tr_gpu, y_tr, sample_weight=sw_tr)

            fold_train_oof = _predict_score(model, x_va_gpu, n_classes=n_classes)
            fold_train_score = _score_predictions(y_train[va_idx], fold_train_oof)
            cv_fold_scores.append(float(fold_train_score))

            valid_fold_preds.append(_predict_score(model, x_valid_gpu, n_classes=n_classes))
            test_fold_preds.append(_predict_score(model, x_test_gpu, n_classes=n_classes))

        valid_pred = np.mean(np.stack(valid_fold_preds, axis=0), axis=0).astype(np.float32)
        test_pred = np.mean(np.stack(test_fold_preds, axis=0), axis=0).astype(np.float32)
        backend = "xgboost_gpu_cuda_cv_ensemble"
    else:
        model = XGBClassifier(**xgb_params)
        x_train_gpu = _to_cupy_frame(x_train_l3)
        model.fit(x_train_gpu, y_train.astype(np.int32), sample_weight=level3_weight)
        valid_pred = _predict_score(model, x_valid_gpu, n_classes=n_classes)
        test_pred = _predict_score(model, x_test_gpu, n_classes=n_classes)
        backend = "xgboost_gpu_cuda_single_fit"

    valid_auc = _score_predictions(y_valid, valid_pred)

    np.save(out_dir / "level3_valid_pred.npy", valid_pred.astype(np.float32))
    np.save(out_dir / "level3_test_pred.npy", test_pred.astype(np.float32))
    valid_df = pd.DataFrame({"y_true": y_valid.astype(np.int32)})
    for c in range(valid_pred.shape[1]):
        valid_df[f"y_pred_c{c}"] = valid_pred[:, c].astype(np.float32)
    _save_parquet(valid_df, out_dir / "level3_valid_predictions.parquet")

    final_metrics = {
        "stage": "level3_final",
        "model_name": cfg.level3_model_name,
        "backend": backend,
        "feature_set_name": cfg.feature_set_name,
        "valid_auc": float(valid_auc),
        "n_level2_features_used": int(len(used_cols)),
        "n_valid": int(len(y_valid)),
        "model_params": {
            "n_estimators": cfg.level3_xgb_n_estimators,
            "learning_rate": cfg.level3_xgb_learning_rate,
            "max_depth": cfg.level3_xgb_max_depth,
            "subsample": cfg.level3_xgb_subsample,
            "colsample_bytree": cfg.level3_xgb_colsample_bytree,
            "random_state": cfg.random_state,
            "cv_enabled": bool(use_cv),
            "cv_folds": int(max(1, cfg.level3_cv_folds)),
            "cv_fold_train_scores": [float(s) for s in cv_fold_scores],
        },
        "sample_weight_by_class": {
            str(k): float(v) for k, v in (sample_weight_by_class or {}).items()
        },
    }
    (out_dir / "level3_final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    return final_metrics


def _build_level4_meta_features(
    level2_valid: pd.DataFrame,
    level2_test: pd.DataFrame,
    level3_valid_pred: np.ndarray | None = None,
    level3_test_pred: np.ndarray | None = None,
    include_level3: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pred_cols = sorted([c for c in level2_valid.columns if c.startswith("l1_pred__")])
    if not pred_cols:
        raise RuntimeError("Level-4 stacking requires Level-2 prediction columns (prefix 'l1_pred__').")

    x_valid_l2 = level2_valid[pred_cols].fillna(0.0).astype(np.float32).to_numpy()
    x_test_l2 = level2_test[pred_cols].fillna(0.0).astype(np.float32).to_numpy()

    if include_level3:
        if level3_valid_pred is None or level3_test_pred is None:
            raise RuntimeError("include_level3=True requires level3 predictions.")
        l3_valid = np.asarray(level3_valid_pred)
        l3_test = np.asarray(level3_test_pred)
        if l3_valid.ndim == 1:
            l3_valid = l3_valid.reshape(-1, 1)
        if l3_test.ndim == 1:
            l3_test = l3_test.reshape(-1, 1)
        x_valid_meta = np.column_stack([x_valid_l2, l3_valid.astype(np.float32)])
        x_test_meta = np.column_stack([x_test_l2, l3_test.astype(np.float32)])
        meta_cols = pred_cols + [f"level3_xgb_pred_c{i}" for i in range(l3_valid.shape[1])]
    else:
        x_valid_meta = x_valid_l2
        x_test_meta = x_test_l2
        meta_cols = pred_cols
    return x_valid_meta, x_test_meta, meta_cols


def _fit_level4_torch_logistic_cv(
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if torch is None or nn is None:
        raise RuntimeError("GPU-only Level-4 fallback requires torch.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only Level-4 fallback requires CUDA-enabled torch.")

    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=random_state)
    oof = np.zeros(x_valid.shape[0], dtype=np.float32)
    device = torch.device("cuda")

    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    test_preds: list[np.ndarray] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_valid, y_valid)):
        torch.manual_seed(random_state + fold_idx)
        model = nn.Linear(x_valid.shape[1], 1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        x_tr = torch.tensor(x_valid[tr_idx], dtype=torch.float32, device=device)
        y_tr = torch.tensor(y_valid[tr_idx], dtype=torch.float32, device=device)
        x_va = torch.tensor(x_valid[va_idx], dtype=torch.float32, device=device)

        model.train()
        for _ in range(180):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tr).squeeze(1)
            loss = criterion(logits, y_tr)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va_prob = torch.sigmoid(model(x_va).squeeze(1)).detach().cpu().numpy().astype(np.float32)
            te_prob = torch.sigmoid(model(x_test_t).squeeze(1)).detach().cpu().numpy().astype(np.float32)
        oof[va_idx] = va_prob
        test_preds.append(te_prob)

        del x_tr, y_tr, x_va, model
        torch.cuda.empty_cache()

    test_pred = np.mean(np.vstack(test_preds), axis=0).astype(np.float32)
    return oof, test_pred, "torch_cuda_logistic_fallback"


def _fit_level4_xgb_meta_cv(
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if XGBClassifier is None:
        raise RuntimeError("xgboost not installed.")
    if cp is None:
        raise RuntimeError("cupy not installed.")

    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=random_state)
    y_arr = np.asarray(y_valid).astype(np.int32).reshape(-1)
    n_classes = int(np.max(y_arr)) + 1
    if n_classes > 2:
        oof = np.zeros((x_valid.shape[0], n_classes), dtype=np.float32)
    else:
        oof = np.zeros(x_valid.shape[0], dtype=np.float32)
    test_preds: list[np.ndarray] = []

    pos = float(np.sum(y_arr > 0.5))
    neg = float(max(0, y_arr.shape[0] - pos))
    scale_pos_weight = float(min(100.0, max(1.0, neg / max(1.0, pos))))

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_valid, y_arr)):
        model_params: dict[str, object] = {
            "device": "cuda",
            "tree_method": "hist",
            "n_estimators": 260,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 2.0,
            "min_child_weight": 2.0,
            "gamma": 0.0,
            "random_state": random_state + fold_idx,
        }
        if n_classes > 2:
            model_params.update(
                {
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "num_class": int(n_classes),
                }
            )
        else:
            model_params.update(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "scale_pos_weight": scale_pos_weight,
                }
            )
        model = XGBClassifier(**model_params)
        x_tr = cp.asarray(x_valid[tr_idx], dtype=cp.float32)
        y_tr = y_arr[tr_idx].astype(np.int32)
        x_va = cp.asarray(x_valid[va_idx], dtype=cp.float32)
        x_te = cp.asarray(x_test, dtype=cp.float32)

        model.fit(x_tr, y_tr)
        va_pred = _predict_score(model, x_va, n_classes=n_classes).astype(np.float32)
        te_pred = _predict_score(model, x_te, n_classes=n_classes).astype(np.float32)
        if n_classes > 2:
            oof[va_idx] = _ensure_proba_2d(va_pred, n_classes=n_classes)
            test_preds.append(_ensure_proba_2d(te_pred, n_classes=n_classes))
        else:
            oof[va_idx] = va_pred.reshape(-1).astype(np.float32)
            test_preds.append(te_pred.reshape(-1).astype(np.float32))

    if n_classes > 2:
        test_pred = np.mean(np.stack(test_preds, axis=0), axis=0).astype(np.float32)
    else:
        test_pred = np.mean(np.vstack(test_preds), axis=0).astype(np.float32)
    return oof, test_pred, "xgboost_gpu_meta_cv"


def _fit_level4_catboost_meta_cv(
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if CatBoostClassifier is None:
        raise RuntimeError("catboost not installed.")

    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=random_state)
    y_arr = np.asarray(y_valid).astype(np.int32).reshape(-1)
    n_classes = int(np.max(y_arr)) + 1
    if n_classes > 2:
        oof = np.zeros((x_valid.shape[0], n_classes), dtype=np.float32)
    else:
        oof = np.zeros(x_valid.shape[0], dtype=np.float32)
    test_preds: list[np.ndarray] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_valid, y_arr)):
        params: dict[str, object] = {
            "task_type": "GPU",
            "devices": "0",
            "allow_writing_files": False,
            "verbose": False,
            "iterations": 360,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 4.0,
            "random_strength": 1.0,
            "bagging_temperature": 0.2,
            "random_seed": random_state + fold_idx,
        }
        if n_classes > 2:
            params.update({"loss_function": "MultiClass", "eval_metric": "MultiClass"})
        else:
            params.update({"loss_function": "Logloss", "eval_metric": "AUC"})

        model = CatBoostClassifier(**params)
        model.fit(
            x_valid[tr_idx].astype(np.float32),
            y_arr[tr_idx],
        )
        va_pred = _predict_score(model, x_valid[va_idx].astype(np.float32), n_classes=n_classes).astype(np.float32)
        te_pred = _predict_score(model, x_test.astype(np.float32), n_classes=n_classes).astype(np.float32)
        if n_classes > 2:
            oof[va_idx] = _ensure_proba_2d(va_pred, n_classes=n_classes)
            test_preds.append(_ensure_proba_2d(te_pred, n_classes=n_classes))
        else:
            oof[va_idx] = va_pred.reshape(-1).astype(np.float32)
            test_preds.append(te_pred.reshape(-1).astype(np.float32))

    if n_classes > 2:
        test_pred = np.mean(np.stack(test_preds, axis=0), axis=0).astype(np.float32)
    else:
        test_pred = np.mean(np.vstack(test_preds), axis=0).astype(np.float32)
    return oof, test_pred, "catboost_gpu_meta_cv"


def _fit_level4_logistic_cv(
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    n_splits: int,
    random_state: int,
    c_value: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=random_state)
    y_arr = np.asarray(y_valid).astype(np.int32).reshape(-1)
    n_classes = int(np.max(y_arr)) + 1
    oof = np.zeros((x_valid.shape[0], n_classes), dtype=np.float32)
    test_preds: list[np.ndarray] = []
    supports_multi_class = "multi_class" in inspect.signature(LogisticRegression).parameters

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_valid, y_arr)):
        model_kwargs = {
            "C": float(c_value),
            "max_iter": 1600,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": random_state + fold_idx,
        }
        if supports_multi_class:
            model_kwargs["multi_class"] = "auto"
        model = LogisticRegression(**model_kwargs)
        model.fit(x_valid[tr_idx], y_arr[tr_idx])
        va_pred = model.predict_proba(x_valid[va_idx]).astype(np.float32)
        te_pred = model.predict_proba(x_test).astype(np.float32)
        oof[va_idx] = _ensure_proba_2d(va_pred, n_classes=n_classes)
        test_preds.append(_ensure_proba_2d(te_pred, n_classes=n_classes))

    test_pred = np.mean(np.stack(test_preds, axis=0), axis=0).astype(np.float32)
    return oof, test_pred, "sklearn_logistic_cv"


def _train_level4_logit_stack(
    cfg: BaselineConfig,
    level2_valid: pd.DataFrame,
    level2_test: pd.DataFrame,
    y_valid: np.ndarray,
    out_dir: Path,
    paths,
) -> dict[str, object]:
    level3_valid_path = out_dir / "level3_valid_pred.npy"
    level3_test_path = out_dir / "level3_test_pred.npy"
    level3_available = level3_valid_path.exists() and level3_test_path.exists()
    level3_valid_pred = np.load(level3_valid_path).astype(np.float32) if level3_available else None
    level3_test_pred = np.load(level3_test_path).astype(np.float32) if level3_available else None

    feature_variants: list[tuple[str, np.ndarray, np.ndarray, list[str]]] = []
    x_valid_l2_only, x_test_l2_only, meta_cols_l2_only = _build_level4_meta_features(
        level2_valid=level2_valid,
        level2_test=level2_test,
        include_level3=False,
    )
    feature_variants.append(("l2_only", x_valid_l2_only, x_test_l2_only, meta_cols_l2_only))
    if level3_available:
        x_valid_l3, x_test_l3, meta_cols_l3 = _build_level4_meta_features(
            level2_valid=level2_valid,
            level2_test=level2_test,
            level3_valid_pred=level3_valid_pred,
            level3_test_pred=level3_test_pred,
            include_level3=True,
        )
        feature_variants.append(("l2_plus_l3", x_valid_l3, x_test_l3, meta_cols_l3))

    candidates: list[dict[str, object]] = []
    c_grid = sorted(set([float(cfg.level4_regularization_c), 0.5, 1.0, 2.0, 4.0]))
    for variant_name, x_valid_meta, x_test_meta, meta_cols in feature_variants:
        for c_value in c_grid:
            try:
                oof_pred, test_pred, backend = _fit_level4_logistic_cv(
                    x_valid=x_valid_meta,
                    y_valid=y_valid,
                    x_test=x_test_meta,
                    n_splits=cfg.level4_cv_folds,
                    random_state=cfg.random_state,
                    c_value=float(c_value),
                )
                valid_auc = _score_predictions(y_valid, oof_pred)
                candidates.append(
                    {
                        "variant": variant_name,
                        "backend": backend,
                        "regularization_c": float(c_value),
                        "valid_auc": float(valid_auc),
                        "oof_pred": oof_pred,
                        "test_pred": test_pred,
                        "meta_cols": meta_cols,
                        "n_meta_features": int(x_valid_meta.shape[1]),
                    }
                )
            except Exception as exc:
                print(
                    f"[LEVEL4][WARN] Logistic variant failed ({variant_name}, C={c_value:.3g}): {_short_exc(exc)}"
                )

        if XGBClassifier is not None and cp is not None:
            try:
                oof_pred, test_pred, backend = _fit_level4_xgb_meta_cv(
                    x_valid=x_valid_meta,
                    y_valid=y_valid,
                    x_test=x_test_meta,
                    n_splits=cfg.level4_cv_folds,
                    random_state=cfg.random_state + 17,
                )
                valid_auc = _score_predictions(y_valid, oof_pred)
                candidates.append(
                    {
                        "variant": variant_name,
                        "backend": backend,
                        "regularization_c": None,
                        "valid_auc": float(valid_auc),
                        "oof_pred": oof_pred,
                        "test_pred": test_pred,
                        "meta_cols": meta_cols,
                        "n_meta_features": int(x_valid_meta.shape[1]),
                    }
                )
            except Exception as exc:
                print(f"[LEVEL4][WARN] XGBoost meta-CV failed ({variant_name}): {_short_exc(exc)}")

        if CatBoostClassifier is not None:
            try:
                oof_pred, test_pred, backend = _fit_level4_catboost_meta_cv(
                    x_valid=x_valid_meta,
                    y_valid=y_valid,
                    x_test=x_test_meta,
                    n_splits=cfg.level4_cv_folds,
                    random_state=cfg.random_state + 29,
                )
                valid_auc = _score_predictions(y_valid, oof_pred)
                candidates.append(
                    {
                        "variant": variant_name,
                        "backend": backend,
                        "regularization_c": None,
                        "valid_auc": float(valid_auc),
                        "oof_pred": oof_pred,
                        "test_pred": test_pred,
                        "meta_cols": meta_cols,
                        "n_meta_features": int(x_valid_meta.shape[1]),
                    }
                )
            except Exception as exc:
                print(f"[LEVEL4][WARN] CatBoost meta-CV failed ({variant_name}): {_short_exc(exc)}")

    if not candidates:
        raise RuntimeError("Level-4 stack failed: no valid meta-model candidate.")

    best = max(candidates, key=lambda d: float(d["valid_auc"]))
    oof_pred = np.asarray(best["oof_pred"], dtype=np.float32)
    test_pred = np.asarray(best["test_pred"], dtype=np.float32)
    backend = str(best["backend"])
    valid_auc = float(best["valid_auc"])
    meta_cols = list(best["meta_cols"])
    n_meta_features = int(best["n_meta_features"])

    np.save(out_dir / "level4_valid_pred.npy", oof_pred.astype(np.float32))
    np.save(out_dir / "level4_test_pred.npy", test_pred.astype(np.float32))
    valid_df = pd.DataFrame({"y_true": y_valid.astype(np.int32)})
    if oof_pred.ndim == 1:
        valid_df["y_pred"] = oof_pred.astype(np.float32)
    else:
        for c in range(oof_pred.shape[1]):
            valid_df[f"y_pred_c{c}"] = oof_pred[:, c].astype(np.float32)
    _save_parquet(valid_df, out_dir / "level4_valid_predictions.parquet")
    np.save(paths.oof_dir / f"oof_level4_logit_{_pred_run_tag(cfg)}.npy", oof_pred.astype(np.float32))
    np.save(paths.pred_dir / f"pred_level4_logit_{_pred_run_tag(cfg)}.npy", test_pred.astype(np.float32))

    metrics = {
        "stage": "level4_final",
        "model_name": "kgmon_level4_l2_logit",
        "backend": backend,
        "feature_variant": str(best["variant"]),
        "feature_set_name": cfg.feature_set_name,
        "valid_auc": float(valid_auc),
        "n_meta_features": n_meta_features,
        "n_valid": int(len(y_valid)),
        "cv_folds": int(cfg.level4_cv_folds),
        "regularization_c": (
            None if best["regularization_c"] is None else float(best["regularization_c"])
        ),
        "meta_feature_columns": meta_cols,
        "candidate_scores": [
            {
                "variant": str(c["variant"]),
                "backend": str(c["backend"]),
                "regularization_c": (
                    None if c["regularization_c"] is None else float(c["regularization_c"])
                ),
                "valid_auc": float(c["valid_auc"]),
                "n_meta_features": int(c["n_meta_features"]),
            }
            for c in sorted(candidates, key=lambda d: float(d["valid_auc"]), reverse=True)
        ],
    }
    (out_dir / "level4_final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _run_tree_level_stack(
    cfg: BaselineConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    paths,
) -> dict[str, object]:
    if cfg.target_col not in train_df.columns or cfg.target_col not in valid_df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' missing from level1 feature splits.")
    active_libraries = _resolve_available_gpu_libraries(
        cfg.tree_libraries,
        strict_gpu_only=cfg.strict_gpu_only,
    )
    active_deep_families: tuple[str, ...] = tuple()
    if cfg.run_deep_level2_stack:
        active_deep_families = _resolve_available_deep_families(cfg.deep_model_families)

    class_labels, class_to_index = _load_target_mapping(paths)
    if not class_to_index:
        class_labels, class_to_index = _target_mapping_from_series(train_df[cfg.target_col])
    y_train = _as_binary_target(train_df[cfg.target_col], class_to_index=class_to_index)
    y_valid = _as_binary_target(valid_df[cfg.target_col], class_to_index=class_to_index)
    n_classes = len(class_labels) if class_labels else int(np.max(y_train)) + 1
    train_sample_weight = None
    sample_weight_by_class: dict[int, float] = {}
    if cfg.use_balanced_sample_weight:
        train_sample_weight, sample_weight_by_class = _compute_balanced_sample_weight(
            y_train,
            min_ratio=cfg.min_sample_weight_ratio,
            max_ratio=cfg.max_sample_weight_ratio,
        )
        if sample_weight_by_class:
            pretty = ", ".join(f"class_{k}={v:.3f}" for k, v in sorted(sample_weight_by_class.items()))
            print(f"[TREE_SUITE] Using balanced sample weights: {pretty}")

    x_train, x_valid, x_test, _ = _encode_level_features(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        target_col=cfg.target_col,
    )

    level2_train = train_df.copy()
    level2_valid = valid_df.copy()
    level2_test = test_df.copy()
    level2_train_cols: dict[str, np.ndarray] = {}
    level2_valid_cols: dict[str, np.ndarray] = {}
    level2_test_cols: dict[str, np.ndarray] = {}

    suite_metrics: list[dict[str, object]] = []
    model_counter = 0

    for lib_idx, library_name in enumerate(active_libraries):
        variants = _tree_variant_grid(
            library_name=library_name,
            seed=cfg.random_state + (100 * lib_idx),
            n_models=cfg.models_per_library,
        )
        tuned_variants = _optuna_tuned_tree_variants(
            cfg=cfg,
            library_name=library_name,
            seed=cfg.random_state + (100 * lib_idx),
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
        )
        if tuned_variants:
            variants = variants + tuned_variants

        library_success_count = 0
        for model_idx, params in enumerate(variants, start=1):
            model_name = f"{library_name}_m{model_idx}"
            try:
                train_pred, valid_pred, test_pred, backend = _fit_tree_library_model(
                    library_name=library_name,
                    params=params,
                    x_train=x_train,
                    y_train=y_train,
                    x_valid=x_valid,
                    x_test=x_test,
                    random_state=cfg.random_state + model_counter,
                    sample_weight=train_sample_weight,
                    raw_train_df=train_df,
                    raw_valid_df=valid_df,
                    raw_test_df=test_df,
                    target_col=cfg.target_col,
                )
            except Exception as exc:
                print(f"[TREE_SUITE][WARN] {model_name} failed and will be skipped: {_short_exc(exc)}")
                # If the first variant fails, remaining variants for this backend usually fail as well.
                if library_success_count == 0:
                    print(f"[TREE_SUITE][WARN] Disabling remaining variants for '{library_name}'.")
                    break
                continue

            train_pred = _ensure_proba_2d(train_pred, n_classes=n_classes)
            valid_pred = _ensure_proba_2d(valid_pred, n_classes=n_classes)
            test_pred = _ensure_proba_2d(test_pred, n_classes=n_classes)
            valid_auc = _score_predictions(y_valid, valid_pred)
            if valid_auc < float(cfg.min_tree_valid_auc_keep):
                print(
                    f"[TREE_SUITE] {model_name} dropped | valid_auc={valid_auc:.6f} "
                    f"< min_tree_valid_auc_keep={cfg.min_tree_valid_auc_keep:.3f}"
                )
                continue

            for c in range(train_pred.shape[1]):
                pred_col = f"l1_pred__{library_name}__m{model_idx}__c{c}"
                level2_train_cols[pred_col] = train_pred[:, c].astype(np.float32)
                level2_valid_cols[pred_col] = valid_pred[:, c].astype(np.float32)
                level2_test_cols[pred_col] = test_pred[:, c].astype(np.float32)
            model_counter += 1
            library_success_count += 1
            pred_tag = f"{model_name}_{_pred_run_tag(cfg)}"
            np.save(paths.oof_dir / f"oof_{pred_tag}.npy", valid_pred.astype(np.float32))
            np.save(paths.pred_dir / f"pred_{pred_tag}.npy", test_pred.astype(np.float32))

            run_metrics = {
                "model_name": model_name,
                "library": library_name,
                "backend": backend,
                "feature_set_name": cfg.feature_set_name,
                "valid_auc": float(valid_auc),
                "n_classes": int(train_pred.shape[1]),
                "params": params,
            }
            suite_metrics.append(run_metrics)
            print(
                f"[TREE_SUITE] {model_name} | backend={backend}"
                f" | valid_auc={valid_auc:.6f}"
            )

    if cfg.run_deep_level2_stack and active_deep_families:
        model_counter, _ = _run_deep_level2_stack(
            cfg=cfg,
            x_train_encoded=x_train,
            x_valid_encoded=x_valid,
            x_test_encoded=x_test,
            y_train=y_train,
            y_valid=y_valid,
            paths=paths,
            suite_metrics=suite_metrics,
            model_counter=model_counter,
            level2_train_cols=level2_train_cols,
            level2_valid_cols=level2_valid_cols,
            level2_test_cols=level2_test_cols,
        )

    if level2_train_cols:
        col_order = sorted(level2_train_cols)
        add_train = pd.DataFrame({c: level2_train_cols[c] for c in col_order}, index=level2_train.index)
        add_valid = pd.DataFrame({c: level2_valid_cols[c] for c in col_order}, index=level2_valid.index)
        add_test = pd.DataFrame({c: level2_test_cols[c] for c in col_order}, index=level2_test.index)
        level2_train = pd.concat([level2_train, add_train], axis=1)
        level2_valid = pd.concat([level2_valid, add_valid], axis=1)
        level2_test = pd.concat([level2_test, add_test], axis=1)

    if not suite_metrics:
        raise RuntimeError(
            "No Level-2 models completed successfully in strict GPU mode. "
            "Fix GPU backend availability for requested tree/deep libraries."
        )

    level2_train_meta = level2_train
    level2_valid_meta = level2_valid
    level2_test_meta = level2_test
    all_level2_pred_cols = sorted([c for c in level2_train.columns if c.startswith("l1_pred__")])
    selected_meta_pred_cols, meta_filter_info = _select_level2_prediction_columns_for_meta(
        cfg=cfg,
        suite_metrics=suite_metrics,
        level2_columns=list(level2_train.columns),
    )
    if all_level2_pred_cols and selected_meta_pred_cols and len(selected_meta_pred_cols) < len(all_level2_pred_cols):
        selected_meta_pred_cols = [
            c
            for c in selected_meta_pred_cols
            if c in level2_train.columns and c in level2_valid.columns and c in level2_test.columns
        ]
        base_cols = [c for c in level2_test.columns if not c.startswith("l1_pred__")]
        keep_cols = base_cols + selected_meta_pred_cols
        level2_train_meta = level2_train[keep_cols].copy()
        level2_valid_meta = level2_valid[keep_cols].copy()
        level2_test_meta = level2_test[keep_cols].copy()
        meta_filter_info["selected_pred_columns"] = int(len(selected_meta_pred_cols))
        print(
            "[LEVEL3] Meta feature filter: "
            f"kept {len(selected_meta_pred_cols)}/{len(all_level2_pred_cols)} Level-2 prediction columns "
            f"from {meta_filter_info.get('selected_models', 0)}/{meta_filter_info.get('total_candidate_models', 0)} models."
        )
    elif all_level2_pred_cols:
        print(f"[LEVEL3] Meta feature filter: using all {len(all_level2_pred_cols)} Level-2 prediction columns.")

    libraries_requested = list(active_libraries)
    libraries_requested.extend([f"dl_{family}" for family in active_deep_families])
    out_dir = paths.level2_results_dir / f"{cfg.level2_dataset_name}_{cfg.feature_set_name}"
    _save_tree_suite_artifacts(
        out_dir=out_dir,
        cfg=cfg,
        y_valid=y_valid,
        level2_train=level2_train,
        level2_valid=level2_valid,
        level2_test=level2_test,
        suite_metrics=suite_metrics,
        libraries_requested=libraries_requested,
        sample_weight_by_class=sample_weight_by_class,
    )
    (out_dir / "meta_feature_filter.json").write_text(json.dumps(meta_filter_info, indent=2), encoding="utf-8")

    level3_variants: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = [
        ("filtered", level2_train_meta, level2_valid_meta, level2_test_meta),
    ]
    has_filtered_variant = len(level2_train_meta.columns) < len(level2_train.columns)
    if has_filtered_variant:
        level3_variants.append(("all", level2_train, level2_valid, level2_test))

    level3_runs: list[dict[str, object]] = []
    for variant_name, l2_train_variant, l2_valid_variant, l2_test_variant in level3_variants:
        metrics = _train_level3_xgb(
            cfg=cfg,
            level2_train=l2_train_variant,
            level2_valid=l2_valid_variant,
            level2_test=l2_test_variant,
            y_train=y_train,
            y_valid=y_valid,
            out_dir=out_dir,
            sample_weight=train_sample_weight,
            sample_weight_by_class=sample_weight_by_class,
        )
        metrics = dict(metrics)
        metrics["level2_feature_variant"] = variant_name
        valid_pred = np.load(out_dir / "level3_valid_pred.npy").astype(np.float32)
        test_pred = np.load(out_dir / "level3_test_pred.npy").astype(np.float32)
        level3_runs.append(
            {
                "name": variant_name,
                "metrics": metrics,
                "valid_pred": valid_pred,
                "test_pred": test_pred,
                "level2_valid": l2_valid_variant,
                "level2_test": l2_test_variant,
            }
        )
        print(
            f"[LEVEL3] Variant '{variant_name}' | valid_auc={float(metrics.get('valid_auc', 0.0)):.6f}"
            f" | features={int(metrics.get('n_level2_features_used', 0))}"
        )

    best_level3_run = max(level3_runs, key=lambda r: float(r["metrics"]["valid_auc"]))
    final_metrics = dict(best_level3_run["metrics"])
    final_metrics["variant_search"] = [
        {
            "variant": str(r["name"]),
            "valid_auc": float(r["metrics"]["valid_auc"]),
            "n_level2_features_used": int(r["metrics"].get("n_level2_features_used", 0)),
        }
        for r in level3_runs
    ]
    final_metrics["selected_level2_feature_variant"] = str(best_level3_run["name"])
    np.save(out_dir / "level3_valid_pred.npy", best_level3_run["valid_pred"].astype(np.float32))
    np.save(out_dir / "level3_test_pred.npy", best_level3_run["test_pred"].astype(np.float32))
    valid_pred_arr = _ensure_proba_2d(
        np.asarray(best_level3_run["valid_pred"]),
        n_classes=int(np.max(y_valid)) + 1,
    )
    valid_df = pd.DataFrame({"y_true": y_valid.astype(np.int32)})
    for c in range(valid_pred_arr.shape[1]):
        valid_df[f"y_pred_c{c}"] = valid_pred_arr[:, c].astype(np.float32)
    _save_parquet(valid_df, out_dir / "level3_valid_predictions.parquet")
    (out_dir / "level3_final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    level2_valid_for_level4 = best_level3_run["level2_valid"]
    level2_test_for_level4 = best_level3_run["level2_test"]
    level4_metrics = None
    if cfg.run_level4_stack:
        level4_metrics = _train_level4_logit_stack(
            cfg=cfg,
            level2_valid=level2_valid_for_level4,
            level2_test=level2_test_for_level4,
            y_valid=y_valid,
            out_dir=out_dir,
            paths=paths,
        )

    print(f"[LEVEL3] Final validation score: {final_metrics['valid_auc']:.6f}")
    if level4_metrics is not None:
        print(f"[LEVEL4] Final validation score: {level4_metrics['valid_auc']:.6f}")
    print(f"[LEVEL3] Artifacts saved in: {out_dir}")
    return {
        "level3": final_metrics,
        "level4": level4_metrics,
    }


def _prepare_data(config: PrepareDataConfig | None = None) -> None:
    cfg = config or PrepareDataConfig()
    paths = get_paths()

    train_candidates: list[Path] = []
    if cfg.source_train_csv:
        train_candidates.append(Path(cfg.source_train_csv))
    if cfg.source_csv:
        train_candidates.append(Path(cfg.source_csv))
    train_candidates.extend(
        [
            paths.root / "data" / "train.csv",
            paths.root / "train.csv",
        ]
    )
    train_path = next((p for p in train_candidates if p.exists()), None)
    if train_path is None:
        raise FileNotFoundError("Could not find training CSV. Checked source_train_csv/source_csv/data/train.csv.")

    test_candidates: list[Path] = []
    if cfg.source_test_csv:
        test_candidates.append(Path(cfg.source_test_csv))
    test_candidates.extend(
        [
            paths.root / "data" / "test.csv",
            paths.root / "test.csv",
        ]
    )
    test_path = next((p for p in test_candidates if p.exists()), None)

    train_raw_df = pd.read_csv(train_path)
    if cfg.target_col not in train_raw_df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in training dataset.")

    if test_path is not None:
        test_df = pd.read_csv(test_path)
    else:
        train_raw_df, test_df = train_test_split(
            train_raw_df,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=train_raw_df[cfg.target_col],
        )

    _save_parquet(train_raw_df, paths.processed_data / "train_raw.parquet")
    _save_parquet(test_df, paths.processed_data / "test_raw.parquet")

    train_df, valid_df = train_test_split(
        train_raw_df,
        test_size=cfg.valid_size,
        random_state=cfg.random_state,
        stratify=train_raw_df[cfg.target_col],
    )

    train_folds_df = train_df.copy()
    train_folds_df["cv_fold"] = -1
    skf = StratifiedKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )
    fold_col = train_folds_df.columns.get_loc("cv_fold")
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[cfg.target_col])):
        train_folds_df.iloc[val_idx, fold_col] = fold

    _save_parquet(train_df, paths.processed_data / "train.parquet")
    _save_parquet(valid_df, paths.processed_data / "valid.parquet")
    _save_parquet(test_df, paths.processed_data / "test.parquet")
    _save_parquet(train_folds_df, paths.processed_data / "train_folds.parquet")

    class_labels, class_to_index = _target_mapping_from_series(train_raw_df[cfg.target_col])
    (paths.processed_data / "target_mapping.json").write_text(
        json.dumps({"target_col": cfg.target_col, "class_labels": class_labels}, indent=2),
        encoding="utf-8",
    )
    y_full = _as_binary_target(train_raw_df[cfg.target_col], class_to_index=class_to_index)
    y_train = _as_binary_target(train_df[cfg.target_col], class_to_index=class_to_index)
    y_valid = _as_binary_target(valid_df[cfg.target_col], class_to_index=class_to_index)

    summary = {
        "source_train_csv": str(train_path),
        "source_test_csv": str(test_path) if test_path is not None else None,
        "target_col": cfg.target_col,
        "id_col": cfg.id_col,
        "class_labels": class_labels,
        "seed": cfg.random_state,
        "n_folds": cfg.n_folds,
        "rows": {
            "full": int(len(train_raw_df)),
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "test": int(len(test_df)),
        },
        "target_distribution_index": {
            "full": {str(i): int(np.sum(y_full == i)) for i in range(len(class_labels))},
            "train": {str(i): int(np.sum(y_train == i)) for i in range(len(class_labels))},
            "valid": {str(i): int(np.sum(y_valid == i)) for i in range(len(class_labels))},
        },
        "split_params": {
            "valid_size_within_train_valid": cfg.valid_size,
            "test_source": "provided_csv" if test_path is not None else "split_from_train",
        },
    }
    (paths.processed_data / "split_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"[PREPARE_DATA] Saved split files in: {paths.processed_data}")


def _resolve_feature_columns(df: pd.DataFrame, cfg: FeatureEngineeringConfig) -> list[str]:
    requested = cfg.base_numeric_columns + cfg.snap_columns
    if requested:
        unique = list(dict.fromkeys(requested))
        return [c for c in unique if c in df.columns]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric_cols if c not in {cfg.target_col, "cv_fold"}]


def _build_digit_decimal_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    cfg: FeatureEngineeringConfig,
) -> pd.DataFrame:
    feats: dict[str, pd.Series] = {}

    for col in feature_columns:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        floor_x = np.floor(x)
        frac = x - floor_x

        d1 = np.floor(frac * 10.0).astype(np.int16)
        d2 = (np.floor(frac * 100.0).astype(np.int16) % 10).astype(np.int16)
        frac100 = np.round(frac * 100.0).astype(np.int16)
        mod10 = (floor_x.astype(np.int64) % 10).astype(np.int16)
        mod100 = (floor_x.astype(np.int64) % 100).astype(np.int16)

        feats[f"{col}__frac"] = pd.Series(frac.astype(np.float32), index=df.index)
        feats[f"{col}__d1"] = pd.Series(d1, index=df.index)
        feats[f"{col}__d2"] = pd.Series(d2, index=df.index)
        feats[f"{col}__frac100"] = pd.Series(frac100, index=df.index)
        feats[f"{col}__mod10"] = pd.Series(mod10, index=df.index)
        feats[f"{col}__mod100"] = pd.Series(mod100, index=df.index)

        is_round = (np.abs(frac) < cfg.round_threshold).astype(np.int8)
        feats[f"{col}__is_round"] = pd.Series(is_round, index=df.index)

        for denom in cfg.denominators:
            residual = np.abs((x * denom) - np.round(x * denom)).astype(np.float32)
            feats[f"{col}__residual_d{denom}"] = pd.Series(residual, index=df.index)

        if cfg.include_digit_pair_strings:
            digit_pair = pd.Series(
                d1.astype(str) + "_" + d2.astype(str),
                index=df.index,
                dtype="string",
            ).astype("category")
            feats[f"{col}__digit_pair"] = digit_pair

    return pd.DataFrame(feats, index=df.index)


def _combine_base_and_features(
    df: pd.DataFrame,
    engineered: pd.DataFrame,
    cfg: FeatureEngineeringConfig,
) -> pd.DataFrame:
    passthrough_drop = [c for c in [cfg.target_col, "cv_fold"] if c in df.columns]
    if cfg.keep_original_columns:
        out = pd.concat([df.drop(columns=passthrough_drop), engineered], axis=1)
    else:
        out = engineered.copy()

    if cfg.target_col in df.columns:
        out[cfg.target_col] = df[cfg.target_col].to_numpy()
    if "cv_fold" in df.columns:
        out["cv_fold"] = df["cv_fold"].to_numpy()

    return out


def _feature_engineering(config: FeatureEngineeringConfig | None = None) -> None:
    cfg = config or FeatureEngineeringConfig()
    paths = get_paths()

    required = {
        "train": paths.processed_data / "train.parquet",
        "valid": paths.processed_data / "valid.parquet",
        "test": paths.processed_data / "test.parquet",
        "train_folds": paths.processed_data / "train_folds.parquet",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split files: {missing}. Run prepare_data stage first."
        )

    split_frames = {name: pd.read_parquet(path) for name, path in required.items()}
    feature_columns = _resolve_feature_columns(split_frames["train"], cfg)
    if not feature_columns:
        raise ValueError("No valid feature columns found for digit/decimal extraction.")

    advanced_result = None
    if cfg.enable_advanced_features:
        advanced_result = build_advanced_feature_set(split_frames=split_frames, cfg=cfg)

    feature_root = paths.level1_features_dir / cfg.feature_set_name
    feature_root.mkdir(parents=True, exist_ok=True)

    saved_files: dict[str, str] = {}
    engineered_counts: dict[str, int] = {}
    for split_name, split_df in split_frames.items():
        digit_feats = _build_digit_decimal_features(split_df, feature_columns, cfg)
        if advanced_result is not None and split_name in advanced_result.features_by_split:
            feats = pd.concat([digit_feats, advanced_result.features_by_split[split_name]], axis=1)
        else:
            feats = digit_feats
        out = _combine_base_and_features(split_df, feats, cfg)
        out_path = feature_root / f"{split_name}.parquet"
        _save_parquet(out, out_path)
        saved_files[split_name] = str(out_path)
        engineered_counts[split_name] = int(feats.shape[1])

    digit_feature_count = int(_build_digit_decimal_features(split_frames["train"], feature_columns, cfg).shape[1])
    manifest = {
        "feature_set_name": cfg.feature_set_name,
        "feature_type": "digit_decimal_extraction_plus_advanced",
        "source_columns": feature_columns,
        "denominators": cfg.denominators,
        "round_threshold": cfg.round_threshold,
        "include_digit_pair_strings": cfg.include_digit_pair_strings,
        "keep_original_columns": cfg.keep_original_columns,
        "enable_advanced_features": cfg.enable_advanced_features,
        "enabled_feature_families": cfg.enabled_feature_families,
        "engineered_feature_count": engineered_counts["train"],
        "digit_decimal_feature_count": digit_feature_count,
        "advanced_feature_count": int(
            0
            if advanced_result is None
            else advanced_result.features_by_split["train"].shape[1]
        ),
        "advanced_family_feature_counts": (
            {} if advanced_result is None else advanced_result.family_feature_counts
        ),
        "advanced_feature_columns_by_family": (
            {} if advanced_result is None else advanced_result.family_columns
        ),
        "advanced_metadata": {} if advanced_result is None else advanced_result.metadata,
        "files": saved_files,
    }
    (feature_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[FEATURE_ENGINEERING] Saved Level-1 features: {feature_root}")
    advanced_count = 0 if advanced_result is None else int(advanced_result.features_by_split["train"].shape[1])
    print(
        "[FEATURE_ENGINEERING] Columns used: "
        + ", ".join(feature_columns)
        + f" | digit={digit_feature_count}"
        + f" | advanced={advanced_count}"
        + f" | engineered={engineered_counts['train']}"
    )


def _baseline(config: BaselineConfig | None = None) -> None:
    cfg = config or BaselineConfig()
    paths = get_paths()

    feature_root = paths.level1_features_dir / cfg.feature_set_name
    required = {
        "train": feature_root / "train.parquet",
        "valid": feature_root / "valid.parquet",
        "test": feature_root / "test.parquet",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Level-1 features: {missing}. Run feature_engineering stage first."
        )

    train_df = pd.read_parquet(required["train"])
    valid_df = pd.read_parquet(required["valid"])
    test_df = pd.read_parquet(required["test"])

    if cfg.run_tree_level_stack:
        _cleanup_stale_artifacts(paths, cfg)
        _run_tree_level_stack(
            cfg=cfg,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            paths=paths,
        )
        return

    for split_name, split_df in (("train", train_df), ("valid", valid_df)):
        if cfg.target_col not in split_df.columns:
            raise ValueError(f"Target column '{cfg.target_col}' missing in {split_name} features.")

    candidate_cols = [c for c in train_df.columns if c not in {cfg.target_col, "cv_fold"}]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns available for baseline model.")

    _, class_to_index = _load_target_mapping(paths)
    y_train = _as_binary_target(train_df[cfg.target_col], class_to_index=class_to_index or None)
    y_valid = _as_binary_target(valid_df[cfg.target_col], class_to_index=class_to_index or None)

    if cfg.run_feature_family_suite:
        family_summaries: list[dict[str, object]] = []
        prefix_to_family = cfg.family_prefixes
        all_family_prefixes = tuple(prefix for prefixes in prefix_to_family.values() for prefix in prefixes)
        base_numeric_cols = [c for c in numeric_cols if not c.startswith(all_family_prefixes)]

        variants = _model_variants(cfg, cfg.models_per_family)
        any_family_run = False

        for family_name, prefixes in prefix_to_family.items():
            family_cols = [c for c in numeric_cols if c.startswith(prefixes)]
            if not family_cols:
                continue

            any_family_run = True
            selected_cols = sorted(set(base_numeric_cols + family_cols))
            x_train = train_df[selected_cols].fillna(0.0).astype(np.float32)
            x_valid = valid_df[selected_cols].fillna(0.0).astype(np.float32)
            x_test = test_df[selected_cols].fillna(0.0).astype(np.float32)

            for idx, params in enumerate(variants, start=1):
                suite_model_name = f"{cfg.model_name}_{family_name}_m{idx}"
                valid_pred, test_pred = _fit_predict_hgb(
                    x_train=x_train,
                    y_train=y_train,
                    x_valid=x_valid,
                    x_test=x_test,
                    params=params,
                )
                metrics = _save_model_outputs(
                    paths=paths,
                    cfg=cfg,
                    model_name=suite_model_name,
                    selected_cols=selected_cols,
                    valid_pred=valid_pred,
                    test_pred=test_pred,
                    y_valid=y_valid,
                    params=params,
                    family_name=family_name,
                )
                family_summaries.append(metrics)
                print(
                    f"[BASELINE] {suite_model_name} | family={family_name}"
                    f" | features={len(selected_cols)} | valid_auc={metrics['valid_auc']:.6f}"
                )

        if not any_family_run:
            print("[BASELINE] No family-prefixed numeric features found; running single baseline model.")
            cfg = BaselineConfig(**{**asdict(cfg), "run_feature_family_suite": False})
        else:
            summary_dir = paths.level2_results_dir / f"suite_{cfg.model_name}_{cfg.feature_set_name}"
            summary_dir.mkdir(parents=True, exist_ok=True)
            (summary_dir / "suite_metrics.json").write_text(
                json.dumps(
                    {
                        "feature_set_name": cfg.feature_set_name,
                        "model_name_prefix": cfg.model_name,
                        "models_per_family": cfg.models_per_family,
                        "families_requested": list(prefix_to_family),
                        "runs_completed": len(family_summaries),
                        "metrics": family_summaries,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[BASELINE] Saved suite summary in: {summary_dir}")
            return

    x_train = train_df[numeric_cols].fillna(0.0).astype(np.float32)
    x_valid = valid_df[numeric_cols].fillna(0.0).astype(np.float32)
    x_test = test_df[numeric_cols].fillna(0.0).astype(np.float32)
    params = {
        "learning_rate": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "max_iter": cfg.max_iter,
        "min_samples_leaf": cfg.min_samples_leaf,
        "random_state": cfg.random_state,
    }
    valid_pred, test_pred = _fit_predict_hgb(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        x_test=x_test,
        params=params,
    )
    metrics = _save_model_outputs(
        paths=paths,
        cfg=cfg,
        model_name=cfg.model_name,
        selected_cols=numeric_cols,
        valid_pred=valid_pred,
        test_pred=test_pred,
        y_valid=y_valid,
        params=params,
        family_name="single_baseline",
    )

    pred_tag = f"{cfg.model_name}_{cfg.feature_set_name}"
    level2_dir = paths.level2_results_dir / pred_tag
    print(f"[BASELINE] Valid AUC: {metrics['valid_auc']:.6f}")
    print(f"[BASELINE] Saved metrics/preds in: {level2_dir}")


def _infer_target_column(df: pd.DataFrame) -> str:
    for col in ("Irrigation_Need", "Class", "Churn", "target", "label"):
        if col in df.columns:
            return col
    raise ValueError("Could not infer target column. Expected one of: Irrigation_Need, Class, Churn, target, label.")


def _resolve_selection_min_auc(paths, default: float = 0.60) -> float:
    tree_metric_files = sorted(paths.level2_results_dir.glob("*/tree_suite_metrics.json"), key=lambda p: p.stat().st_mtime)
    if not tree_metric_files:
        return float(default)
    latest = tree_metric_files[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return float(default)
    value = payload.get("min_oof_auc_for_selection")
    try:
        resolved = float(value)
        return min(0.80, max(0.30, resolved))
    except Exception:
        return min(0.80, max(0.30, float(default)))


def _resolve_stacking_min_models(paths, default: int = 8) -> int:
    tree_metric_files = sorted(paths.level2_results_dir.glob("*/tree_suite_metrics.json"), key=lambda p: p.stat().st_mtime)
    if not tree_metric_files:
        return int(default)
    latest = tree_metric_files[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return int(default)
    value = payload.get("stacking_min_models")
    try:
        return max(8, int(value))
    except Exception:
        return max(8, int(default))


def _resolve_class_weight_calibration_enabled(paths, default: bool = True) -> bool:
    tree_metric_files = sorted(paths.level2_results_dir.glob("*/tree_suite_metrics.json"), key=lambda p: p.stat().st_mtime)
    if not tree_metric_files:
        return bool(default)
    latest = tree_metric_files[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return bool(default)
    value = payload.get("enable_class_weight_calibration")
    if value is None:
        return bool(default)
    return bool(value)


def _resolve_class_weight_calibration_grid(
    paths,
    default: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.40, 1.60, 1.80, 2.00),
) -> tuple[float, ...]:
    tree_metric_files = sorted(paths.level2_results_dir.glob("*/tree_suite_metrics.json"), key=lambda p: p.stat().st_mtime)
    if not tree_metric_files:
        return tuple(float(v) for v in default)
    latest = tree_metric_files[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return tuple(float(v) for v in default)
    value = payload.get("class_weight_calibration_grid")
    if not isinstance(value, list) or not value:
        return tuple(float(v) for v in default)
    out = tuple(float(v) for v in value if float(v) > 0.0)
    return out if out else tuple(float(v) for v in default)


def _resolve_class_weight_calibration_trials(paths, default: int = 120) -> int:
    tree_metric_files = sorted(paths.level2_results_dir.glob("*/tree_suite_metrics.json"), key=lambda p: p.stat().st_mtime)
    if not tree_metric_files:
        return int(default)
    latest = tree_metric_files[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return int(default)
    value = payload.get("class_weight_calibration_random_trials")
    try:
        return max(0, int(value))
    except Exception:
        return max(0, int(default))


def _collect_oof_candidates(paths, y_valid: np.ndarray, min_auc: float = 0.0) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for oof_path in sorted(paths.oof_dir.glob("oof_*.npy")):
        try:
            oof = np.asarray(np.load(oof_path), dtype=np.float32)
        except Exception:
            continue
        if oof.shape[0] != y_valid.shape[0] or not np.all(np.isfinite(oof)):
            continue

        pred_path = paths.pred_dir / oof_path.name.replace("oof_", "pred_", 1)
        pred = None
        if pred_path.exists():
            try:
                pred_arr = np.asarray(np.load(pred_path), dtype=np.float32)
                if np.all(np.isfinite(pred_arr)):
                    pred = pred_arr
            except Exception:
                pred = None
        try:
            auc = float(_score_predictions(y_valid, oof))
        except Exception:
            continue
        if auc < float(min_auc):
            continue
        candidates.append(
            {
                "name": oof_path.stem.replace("oof_", "", 1),
                "oof_path": str(oof_path),
                "pred_path": str(pred_path) if pred_path.exists() else None,
                "oof": oof,
                "pred": pred,
                "individual_auc": auc,
            }
        )
    return candidates


def _eda() -> None:
    paths = get_paths()
    out_dir = paths.level2_results_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": paths.processed_data / "train.parquet",
        "valid": paths.processed_data / "valid.parquet",
        "test": paths.processed_data / "test.parquet",
    }
    missing = [k for k, p in split_paths.items() if not p.exists()]
    if missing:
        print(f"[EDA] Missing split files: {missing}. Run prepare_data first.")
        return

    frames = {name: pd.read_parquet(path) for name, path in split_paths.items()}
    target_col = _infer_target_column(frames["train"])
    summary: dict[str, object] = {"target_col": target_col, "splits": {}}

    for name, df in frames.items():
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        split_info: dict[str, object] = {
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "numeric_cols": int(len(numeric_cols)),
            "non_numeric_cols": int(len(non_numeric_cols)),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_ratio": float(df.isna().sum().sum() / max(1, df.shape[0] * df.shape[1])),
        }
        if target_col in df.columns:
            y = _as_binary_target(df[target_col])
            split_info["target_class_distribution"] = {
                str(i): int(np.sum(y == i)) for i in range(int(np.max(y)) + 1)
            }
        summary["splits"][name] = split_info

    summary["train_top_missing_ratio"] = (
        frames["train"].isna().mean().sort_values(ascending=False).head(25).astype(float).to_dict()
    )
    out_path = out_dir / "eda_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[EDA] Saved summary: {out_path}")


def _hill_climb() -> None:
    paths = get_paths()
    valid_path = paths.processed_data / "valid.parquet"
    if not valid_path.exists():
        print("[HILL_CLIMB] Missing valid split. Run prepare_data first.")
        return

    valid_df = pd.read_parquet(valid_path)
    target_col = _infer_target_column(valid_df)
    class_labels, class_to_index = _load_target_mapping(paths)
    y_valid = _as_binary_target(valid_df[target_col], class_to_index=class_to_index or None)
    min_auc = _resolve_selection_min_auc(paths, default=0.60)
    min_models = _resolve_stacking_min_models(paths, default=8)
    candidates = _collect_oof_candidates(paths, y_valid, min_auc=min_auc)
    if not candidates:
        print(f"[HILL_CLIMB] No compatible OOF predictions found at min_auc={min_auc:.3f}.")
        return

    candidates = [c for c in candidates if c.get("pred") is not None]
    if not candidates:
        print("[HILL_CLIMB] No candidates have both OOF and test predictions.")
        return

    candidates = sorted(candidates, key=lambda x: float(x["individual_auc"]), reverse=True)
    candidates = candidates[: min(300, len(candidates))]
    selected_names: list[str] = [str(candidates[0]["name"])]
    selected_weights: list[float] = [1.0]
    selected_set: set[str] = set()
    selected_set.add(selected_names[0])
    progression: list[dict[str, object]] = []
    ensemble = np.asarray(candidates[0]["oof"], dtype=np.float32).copy()
    ensemble_test = np.asarray(candidates[0]["pred"], dtype=np.float32).copy()
    current_auc = float(_score_predictions(y_valid, ensemble))
    progression.append(
        {
            "step": 1,
            "added_model": selected_names[0],
            "blend_alpha_prev": 0.0,
            "blend_alpha_new": 1.0,
            "ensemble_valid_auc": float(current_auc),
        }
    )
    alpha_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    for step in range(2, min(150, len(candidates)) + 1):
        best_name = None
        best_alpha_prev = None
        best_auc = current_auc
        best_pred = None
        best_test_pred = None

        for cand in candidates:
            name = str(cand["name"])
            if name in selected_set:
                continue

            cand_oof = np.asarray(cand["oof"], dtype=np.float32)
            cand_test = np.asarray(cand["pred"], dtype=np.float32)
            for alpha_prev in alpha_grid:
                alpha_new = 1.0 - float(alpha_prev)
                trial = (float(alpha_prev) * ensemble) + (alpha_new * cand_oof)
                auc = float(_score_predictions(y_valid, trial))
                if best_name is None or auc > best_auc + 1e-12:
                    best_name = name
                    best_alpha_prev = float(alpha_prev)
                    best_auc = auc
                    best_pred = trial.astype(np.float32)
                    best_test_pred = ((float(alpha_prev) * ensemble_test) + (alpha_new * cand_test)).astype(np.float32)

        if best_name is None:
            break
        if best_auc <= current_auc + 1e-6:
            break

        for idx in range(len(selected_weights)):
            selected_weights[idx] *= float(best_alpha_prev)
        selected_weights.append(1.0 - float(best_alpha_prev))
        selected_names.append(best_name)
        selected_set.add(best_name)
        ensemble = best_pred.astype(np.float32)
        ensemble_test = best_test_pred.astype(np.float32)
        current_auc = best_auc
        progression.append(
            {
                "step": int(step),
                "added_model": best_name,
                "blend_alpha_prev": float(best_alpha_prev),
                "blend_alpha_new": float(1.0 - float(best_alpha_prev)),
                "ensemble_valid_auc": float(current_auc),
            }
        )

    out_dir = paths.level2_results_dir / "hill_climb"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "hill_climb_valid_pred.npy", ensemble.astype(np.float32))
    np.save(out_dir / "hill_climb_test_pred.npy", ensemble_test.astype(np.float32))

    result = {
        "target_col": target_col,
        "candidates_considered": int(len(candidates)),
        "models_selected": selected_names,
        "model_weights": [float(w) for w in selected_weights],
        "stacking_seed_models": [str(c["name"]) for c in candidates[: max(min_models, 16)]],
        "final_valid_auc": float(current_auc),
        "progression": progression,
        "top_individual_models": [
            {"name": c["name"], "valid_auc": float(c["individual_auc"])} for c in candidates[:30]
        ],
    }
    (out_dir / "hill_climb_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[HILL_CLIMB] Selected {len(selected_names)} models | valid_auc={current_auc:.6f}")
    print(f"[HILL_CLIMB] Saved artifacts in: {out_dir}")


def _as_proba_matrix(arr: np.ndarray, n_classes: int) -> np.ndarray:
    out = _ensure_proba_2d(np.asarray(arr), n_classes=n_classes).astype(np.float32)
    return out


def _power_normalize_proba(pred: np.ndarray, power: float) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float32)
    if abs(float(power) - 1.0) <= 1e-8:
        return p.astype(np.float32)
    p = np.clip(p, 1e-9, 1.0)
    p = np.power(p, float(power)).astype(np.float32)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-9, None)
    return p.astype(np.float32)


def _safe_feature_name(name: str, idx: int) -> str:
    raw = "".join(ch if ch.isalnum() else "_" for ch in str(name))
    raw = "_".join([part for part in raw.split("_") if part])
    return f"m{idx}_{raw[:80]}"


def _build_stacking_meta_features(
    selected: list[dict[str, object]],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    valid_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    names: list[str] = []
    valid_stack: list[np.ndarray] = []
    test_stack: list[np.ndarray] = []

    for idx, cand in enumerate(selected):
        base = _safe_feature_name(str(cand["name"]), idx)
        oof = _as_proba_matrix(np.asarray(cand["oof"]), n_classes=n_classes)
        pred = _as_proba_matrix(np.asarray(cand["pred"]), n_classes=n_classes)
        valid_stack.append(oof)
        test_stack.append(pred)

        for c in range(n_classes):
            valid_parts.append(oof[:, c])
            test_parts.append(pred[:, c])
            names.append(f"{base}__proba_c{c}")

        oof_sorted = np.sort(oof, axis=1)
        pred_sorted = np.sort(pred, axis=1)
        oof_top = oof_sorted[:, -1]
        pred_top = pred_sorted[:, -1]
        if n_classes > 1:
            oof_margin = oof_sorted[:, -1] - oof_sorted[:, -2]
            pred_margin = pred_sorted[:, -1] - pred_sorted[:, -2]
        else:
            oof_margin = oof_top
            pred_margin = pred_top
        oof_entropy = -(oof * np.log(np.clip(oof, 1e-9, 1.0))).sum(axis=1)
        pred_entropy = -(pred * np.log(np.clip(pred, 1e-9, 1.0))).sum(axis=1)

        valid_parts.extend([oof_top, oof_margin.astype(np.float32), oof_entropy.astype(np.float32)])
        test_parts.extend([pred_top, pred_margin.astype(np.float32), pred_entropy.astype(np.float32)])
        names.extend([f"{base}__top1", f"{base}__margin", f"{base}__entropy"])

    if not valid_stack:
        raise ValueError("No models available for stacking meta features.")

    stack_valid = np.stack(valid_stack, axis=0).astype(np.float32)
    stack_test = np.stack(test_stack, axis=0).astype(np.float32)
    mean_valid = stack_valid.mean(axis=0)
    mean_test = stack_test.mean(axis=0)
    std_valid = stack_valid.std(axis=0)
    std_test = stack_test.std(axis=0)

    for c in range(n_classes):
        valid_parts.extend([mean_valid[:, c], std_valid[:, c]])
        test_parts.extend([mean_test[:, c], std_test[:, c]])
        names.extend([f"ens__mean_c{c}", f"ens__std_c{c}"])

    mean_sorted_valid = np.sort(mean_valid, axis=1)
    mean_sorted_test = np.sort(mean_test, axis=1)
    mean_top_valid = mean_sorted_valid[:, -1]
    mean_top_test = mean_sorted_test[:, -1]
    if n_classes > 1:
        mean_margin_valid = mean_sorted_valid[:, -1] - mean_sorted_valid[:, -2]
        mean_margin_test = mean_sorted_test[:, -1] - mean_sorted_test[:, -2]
    else:
        mean_margin_valid = mean_top_valid
        mean_margin_test = mean_top_test
    mean_entropy_valid = -(mean_valid * np.log(np.clip(mean_valid, 1e-9, 1.0))).sum(axis=1)
    mean_entropy_test = -(mean_test * np.log(np.clip(mean_test, 1e-9, 1.0))).sum(axis=1)
    std_sum_valid = std_valid.sum(axis=1)
    std_sum_test = std_test.sum(axis=1)

    valid_parts.extend(
        [
            mean_top_valid.astype(np.float32),
            mean_margin_valid.astype(np.float32),
            mean_entropy_valid.astype(np.float32),
            std_sum_valid.astype(np.float32),
        ]
    )
    test_parts.extend(
        [
            mean_top_test.astype(np.float32),
            mean_margin_test.astype(np.float32),
            mean_entropy_test.astype(np.float32),
            std_sum_test.astype(np.float32),
        ]
    )
    names.extend(["ens__top1", "ens__margin", "ens__entropy", "ens__stdsum"])

    x_valid = np.column_stack(valid_parts).astype(np.float32)
    x_test = np.column_stack(test_parts).astype(np.float32)
    return x_valid, x_test, names


def _resolve_id_col(paths) -> str:
    split_summary_path = paths.processed_data / "split_summary.json"
    if split_summary_path.exists():
        try:
            payload = json.loads(split_summary_path.read_text(encoding="utf-8"))
            return str(payload.get("id_col", "id"))
        except Exception:
            return "id"
    return "id"


def _write_submission_from_proba(
    *,
    paths,
    pred: np.ndarray,
    class_labels: list[str],
    target_col: str,
    filename: str,
) -> Path:
    test_path = paths.processed_data / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError("Missing processed test split for submission writing.")
    test_df = pd.read_parquet(test_path)
    id_col = _resolve_id_col(paths)
    pred_labels = _proba_to_labels(pred, class_labels)
    if id_col in test_df.columns:
        submission_id = test_df[id_col].to_numpy()
    else:
        submission_id = np.arange(len(test_df), dtype=np.int64)
    submission = pd.DataFrame({id_col: submission_id, target_col: pred_labels})
    submissions_dir = paths.outputs_root / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submissions_dir / filename
    submission.to_csv(submission_path, index=False)
    return submission_path


def _weight_grid(n_candidates: int, step: float) -> list[np.ndarray]:
    step_units = max(1, int(round(1.0 / max(1e-6, step))))
    grids: list[np.ndarray] = []
    if n_candidates == 1:
        return [np.array([1.0], dtype=np.float32)]
    if n_candidates == 2:
        for a in range(step_units + 1):
            b = step_units - a
            grids.append(np.array([a / step_units, b / step_units], dtype=np.float32))
        return grids
    if n_candidates == 3:
        for a in range(step_units + 1):
            for b in range(step_units - a + 1):
                c = step_units - a - b
                grids.append(np.array([a / step_units, b / step_units, c / step_units], dtype=np.float32))
        return grids
    raise ValueError("Weight grid supports 1 to 3 candidates.")


def _hard_vote_proba_from_stack(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected shape (n_models, n_rows, n_classes), got {arr.shape}")
    n_models, n_rows, n_classes = arr.shape
    if n_models <= 0 or n_rows <= 0 or n_classes <= 1:
        raise ValueError(f"Invalid stack shape: {arr.shape}")

    labels = np.argmax(arr, axis=2).astype(np.int32)  # (n_models, n_rows)
    counts = np.zeros((n_rows, n_classes), dtype=np.int32)
    for cls in range(n_classes):
        counts[:, cls] = np.sum(labels == cls, axis=0, dtype=np.int32)

    winners = np.argmax(counts, axis=1).astype(np.int32)
    max_counts = counts[np.arange(n_rows), winners]
    tie_mask = np.sum(counts == max_counts.reshape(-1, 1), axis=1) > 1
    if np.any(tie_mask):
        # Tie-break using mean probability to avoid arbitrary class ordering effects.
        mean_proba = np.mean(arr, axis=0)
        winners[tie_mask] = np.argmax(mean_proba[tie_mask], axis=1).astype(np.int32)

    out = np.zeros((n_rows, n_classes), dtype=np.float32)
    out[np.arange(n_rows), winners] = 1.0
    return out


def _collect_historical_final_blend_candidates(
    *,
    paths,
    y_valid: np.ndarray,
    n_classes: int,
    max_items: int = 8,
) -> list[dict[str, object]]:
    history_root = paths.level2_results_dir / "history"
    if not history_root.exists():
        return []

    candidates: list[dict[str, object]] = []
    run_dirs = sorted(
        [p for p in history_root.glob("final_blend_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        valid_path = run_dir / "final_blend_valid_pred.npy"
        test_path = run_dir / "final_blend_test_pred.npy"
        if not valid_path.exists() or not test_path.exists():
            continue
        try:
            oof = _as_proba_matrix(np.load(valid_path), n_classes=n_classes)
            pred = _as_proba_matrix(np.load(test_path), n_classes=n_classes)
        except Exception:
            continue
        if oof.shape[0] != y_valid.shape[0]:
            continue
        if not np.all(np.isfinite(oof)) or not np.all(np.isfinite(pred)):
            continue
        try:
            auc = float(_score_predictions(y_valid, oof))
        except Exception:
            continue
        candidates.append(
            {
                "name": f"history::{run_dir.name}",
                "oof": oof.astype(np.float32),
                "pred": pred.astype(np.float32),
                "individual_auc": auc,
            }
        )
        if len(candidates) >= int(max_items):
            break
    return candidates


def _archive_final_blend_artifacts(
    *,
    paths,
    valid_pred: np.ndarray,
    test_pred: np.ndarray,
    summary: dict[str, object],
    max_keep: int = 24,
) -> None:
    history_root = paths.level2_results_dir / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    score = float(summary.get("best_valid_auc", 0.0))
    score_tag = f"{score:.6f}".replace(".", "p")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = history_root / f"final_blend_{stamp}_{score_tag}"
    suffix = 1
    while run_dir.exists():
        run_dir = history_root / f"final_blend_{stamp}_{score_tag}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    np.save(run_dir / "final_blend_valid_pred.npy", np.asarray(valid_pred, dtype=np.float32))
    np.save(run_dir / "final_blend_test_pred.npy", np.asarray(test_pred, dtype=np.float32))
    (run_dir / "final_blend_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_dirs = sorted(
        [p for p in history_root.glob("final_blend_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in run_dirs[int(max_keep) :]:
        try:
            shutil.rmtree(stale, ignore_errors=True)
        except Exception:
            continue


def _stacking() -> None:
    paths = get_paths()
    valid_path = paths.processed_data / "valid.parquet"
    if not valid_path.exists():
        print("[STACKING] Missing valid split. Run prepare_data first.")
        return

    valid_df = pd.read_parquet(valid_path)
    target_col = _infer_target_column(valid_df)
    class_labels, class_to_index = _load_target_mapping(paths)
    y_valid = _as_binary_target(valid_df[target_col], class_to_index=class_to_index or None)
    min_auc = _resolve_selection_min_auc(paths, default=0.60)
    min_models = _resolve_stacking_min_models(paths, default=8)
    candidates = _collect_oof_candidates(paths, y_valid, min_auc=min_auc)
    if not candidates:
        print(f"[STACKING] No compatible OOF predictions found at min_auc={min_auc:.3f}.")
        return

    hill_file = paths.level2_results_dir / "hill_climb" / "hill_climb_selection.json"
    selected_names: list[str] = []
    if hill_file.exists():
        try:
            payload = json.loads(hill_file.read_text(encoding="utf-8"))
            selected_names = [str(x) for x in payload.get("models_selected", [])]
            if len(selected_names) < min_models:
                seed_names = [str(x) for x in payload.get("stacking_seed_models", [])]
                merged = list(dict.fromkeys(selected_names + seed_names))
                selected_names = merged
        except Exception:
            selected_names = []

    if not selected_names:
        selected_names = [str(c["name"]) for c in sorted(candidates, key=lambda x: x["individual_auc"], reverse=True)[:40]]

    selected = [c for c in candidates if c["name"] in set(selected_names)]
    selected = [c for c in selected if c["pred"] is not None]
    if len(selected) < min_models:
        selected = [c for c in sorted(candidates, key=lambda x: x["individual_auc"], reverse=True)[:40] if c["pred"] is not None]
        selected = selected[: max(min_models, min(24, len(selected)))]
    if len(selected) < 2:
        print("[STACKING] Need at least 2 models with OOF and test predictions.")
        return

    n_classes = int(np.max(y_valid)) + 1
    x_valid, x_test, meta_feature_columns = _build_stacking_meta_features(selected, n_classes=n_classes)

    scores: dict[str, float] = {}
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    oof_logit, test_logit, _ = _fit_level4_logistic_cv(
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        n_splits=5,
        random_state=42,
        c_value=1.0,
    )
    logit_score = float(_score_predictions(y_valid, oof_logit))
    scores["sklearn_logistic_cv"] = logit_score
    preds["sklearn_logistic_cv"] = (oof_logit, test_logit)

    if XGBClassifier is not None and cp is not None:
        try:
            oof_xgb, test_xgb, _ = _fit_level4_xgb_meta_cv(
                x_valid=x_valid,
                y_valid=y_valid,
                x_test=x_test,
                n_splits=5,
                random_state=42,
            )
            xgb_score = float(_score_predictions(y_valid, oof_xgb))
            scores["xgboost_gpu_meta_cv"] = xgb_score
            preds["xgboost_gpu_meta_cv"] = (oof_xgb, test_xgb)
        except Exception as exc:
            print(f"[STACKING][WARN] XGBoost meta-CV unavailable, using logistic only: {_short_exc(exc)}")

    if CatBoostClassifier is not None:
        try:
            oof_cat, test_cat, _ = _fit_level4_catboost_meta_cv(
                x_valid=x_valid,
                y_valid=y_valid,
                x_test=x_test,
                n_splits=5,
                random_state=43,
            )
            cat_score = float(_score_predictions(y_valid, oof_cat))
            scores["catboost_gpu_meta_cv"] = cat_score
            preds["catboost_gpu_meta_cv"] = (oof_cat, test_cat)
        except Exception as exc:
            print(f"[STACKING][WARN] CatBoost meta-CV unavailable, skipping: {_short_exc(exc)}")

    backend = max(scores, key=scores.get)
    oof_pred, test_pred = preds[backend]
    valid_auc = float(scores[backend])
    calibration = {
        "enabled": False,
        "applied": False,
        "base_valid_auc": float(valid_auc),
    }
    if n_classes > 2 and _resolve_class_weight_calibration_enabled(paths, default=True):
        grid = _resolve_class_weight_calibration_grid(paths)
        random_trials = _resolve_class_weight_calibration_trials(paths, default=120)
        calibration["enabled"] = True
        cal_oof, cal_weights, cal_score, n_evaluated = _optimize_class_probability_weights(
            y_true=y_valid,
            pred=oof_pred,
            grid_values=grid,
            random_trials=random_trials,
            random_state=42,
        )
        calibration["weights"] = [float(v) for v in cal_weights.tolist()]
        calibration["n_weight_candidates_evaluated"] = int(n_evaluated)
        calibration["calibrated_valid_auc"] = float(cal_score)
        if cal_score > valid_auc + 1e-8:
            oof_pred = cal_oof
            test_pred = _apply_class_probability_weights(test_pred, cal_weights)
            valid_auc = float(cal_score)
            backend = f"{backend}+class_weight_calibration"
            calibration["applied"] = True

    out_dir = paths.level2_results_dir / "stacking"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "stacked_valid_pred.npy", oof_pred.astype(np.float32))
    np.save(out_dir / "stacked_test_pred.npy", test_pred.astype(np.float32))
    stacked_df = pd.DataFrame({"y_true": y_valid.astype(np.int32)})
    if np.asarray(oof_pred).ndim == 1:
        stacked_df["y_pred"] = np.asarray(oof_pred).astype(np.float32)
    else:
        for c in range(np.asarray(oof_pred).shape[1]):
            stacked_df[f"y_pred_c{c}"] = np.asarray(oof_pred)[:, c].astype(np.float32)
    _save_parquet(stacked_df, out_dir / "stacked_valid_predictions.parquet")
    metrics = {
        "backend": backend,
        "candidate_backend_scores": {k: float(v) for k, v in scores.items()},
        "target_col": target_col,
        "n_models": int(len(selected)),
        "selected_models": [str(c["name"]) for c in selected],
        "n_meta_features": int(x_valid.shape[1]),
        "meta_feature_columns": meta_feature_columns,
        "valid_auc": valid_auc,
        "class_weight_calibration": calibration,
    }
    (out_dir / "stacking_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[STACKING] valid_auc={valid_auc:.6f} | models={len(selected)}")
    print(f"[STACKING] Saved artifacts in: {out_dir}")


def _final_blend() -> None:
    paths = get_paths()
    valid_path = paths.processed_data / "valid.parquet"
    if not valid_path.exists():
        print("[FINAL_BLEND] Missing valid split. Run prepare_data first.")
        return

    valid_df = pd.read_parquet(valid_path)
    target_col = _infer_target_column(valid_df)
    class_labels, class_to_index = _load_target_mapping(paths)
    y_valid = _as_binary_target(valid_df[target_col], class_to_index=class_to_index or None)
    n_classes = int(np.max(y_valid)) + 1

    candidates = _collect_oof_candidates(paths, y_valid, min_auc=0.0)
    if not candidates:
        print("[FINAL_BLEND] No OOF candidates found.")
        return
    for cand in candidates:
        cand["oof"] = _as_proba_matrix(np.asarray(cand["oof"]), n_classes=n_classes)
        if cand["pred"] is not None:
            cand["pred"] = _as_proba_matrix(np.asarray(cand["pred"]), n_classes=n_classes)

    candidates = [c for c in candidates if c.get("pred") is not None]
    if not candidates:
        print("[FINAL_BLEND] No candidates have both OOF and test predictions.")
        return
    candidates = sorted(candidates, key=lambda x: float(x["individual_auc"]), reverse=True)
    best_single = candidates[0]
    top_single_pool = max(12, min(36, len(candidates)))
    pool: list[dict[str, object]] = []
    seen: set[str] = set()
    for cand in candidates[:top_single_pool]:
        name = str(cand["name"])
        if name in seen:
            continue
        pool.append(cand)
        seen.add(name)

    hc_valid = paths.level2_results_dir / "hill_climb" / "hill_climb_valid_pred.npy"
    hc_test = paths.level2_results_dir / "hill_climb" / "hill_climb_test_pred.npy"
    if hc_valid.exists() and hc_test.exists():
        try:
            oof = _as_proba_matrix(np.load(hc_valid), n_classes=n_classes)
            pred = _as_proba_matrix(np.load(hc_test), n_classes=n_classes)
            if oof.shape[0] == y_valid.shape[0]:
                auc = float(_score_predictions(y_valid, oof))
                name = "hill_climb"
                if name not in seen:
                    pool.append({"name": name, "oof": oof, "pred": pred, "individual_auc": auc})
                    seen.add(name)
        except Exception:
            pass

    stack_valid = paths.level2_results_dir / "stacking" / "stacked_valid_pred.npy"
    stack_test = paths.level2_results_dir / "stacking" / "stacked_test_pred.npy"
    if stack_valid.exists() and stack_test.exists():
        try:
            oof = _as_proba_matrix(np.load(stack_valid), n_classes=n_classes)
            pred = _as_proba_matrix(np.load(stack_test), n_classes=n_classes)
            if oof.shape[0] == y_valid.shape[0]:
                auc = float(_score_predictions(y_valid, oof))
                name = "stacking"
                if name not in seen:
                    pool.append({"name": name, "oof": oof, "pred": pred, "individual_auc": auc})
                    seen.add(name)
        except Exception:
            pass

    prev_final_valid = paths.level2_results_dir / "final_blend" / "final_blend_valid_pred.npy"
    prev_final_test = paths.level2_results_dir / "final_blend" / "final_blend_test_pred.npy"
    if prev_final_valid.exists() and prev_final_test.exists():
        try:
            oof = _as_proba_matrix(np.load(prev_final_valid), n_classes=n_classes)
            pred = _as_proba_matrix(np.load(prev_final_test), n_classes=n_classes)
            if oof.shape[0] == y_valid.shape[0]:
                auc = float(_score_predictions(y_valid, oof))
                name = "previous_final_blend"
                if name not in seen:
                    pool.append({"name": name, "oof": oof, "pred": pred, "individual_auc": auc})
                    seen.add(name)
        except Exception:
            pass

    for hist in _collect_historical_final_blend_candidates(
        paths=paths,
        y_valid=y_valid,
        n_classes=n_classes,
        max_items=10,
    ):
        name = str(hist["name"])
        if name in seen:
            continue
        pool.append(hist)
        seen.add(name)

    # Add synthetic ensemble candidates from top-N models for diversity in the final search pool.
    for n in (3, 5, 8, 12, 16, 24):
        if len(candidates) < n:
            continue
        top_n = candidates[:n]
        top_oof = np.stack([np.asarray(c["oof"], dtype=np.float32) for c in top_n], axis=0)
        top_pred = np.stack([np.asarray(c["pred"], dtype=np.float32) for c in top_n], axis=0)

        mean_name = f"top{n}_mean"
        if mean_name not in seen:
            mean_oof = np.mean(top_oof, axis=0).astype(np.float32)
            mean_pred = np.mean(top_pred, axis=0).astype(np.float32)
            mean_auc = float(_score_predictions(y_valid, mean_oof))
            pool.append({"name": mean_name, "oof": mean_oof, "pred": mean_pred, "individual_auc": mean_auc})
            seen.add(mean_name)

        geo_name = f"top{n}_geo"
        if geo_name not in seen:
            geo_oof = np.exp(np.mean(np.log(np.clip(top_oof, 1e-9, 1.0)), axis=0)).astype(np.float32)
            geo_oof = geo_oof / np.clip(geo_oof.sum(axis=1, keepdims=True), 1e-9, None)
            geo_pred = np.exp(np.mean(np.log(np.clip(top_pred, 1e-9, 1.0)), axis=0)).astype(np.float32)
            geo_pred = geo_pred / np.clip(geo_pred.sum(axis=1, keepdims=True), 1e-9, None)
            geo_auc = float(_score_predictions(y_valid, geo_oof))
            pool.append({"name": geo_name, "oof": geo_oof, "pred": geo_pred, "individual_auc": geo_auc})
            seen.add(geo_name)

        vote_name = f"top{n}_vote"
        if vote_name not in seen:
            vote_oof = _hard_vote_proba_from_stack(top_oof)
            vote_pred = _hard_vote_proba_from_stack(top_pred)
            vote_auc = float(_score_predictions(y_valid, vote_oof))
            pool.append({"name": vote_name, "oof": vote_oof, "pred": vote_pred, "individual_auc": vote_auc})
            seen.add(vote_name)

    pool = sorted(pool, key=lambda x: float(x["individual_auc"]), reverse=True)
    if not pool:
        print("[FINAL_BLEND] No blendable candidates found.")
        return

    best_score = -1.0
    best_weights: np.ndarray | None = None
    best_combo: tuple[int, ...] | None = None
    best_oof: np.ndarray | None = None
    best_test: np.ndarray | None = None
    step = 0.025
    coarse_pool = pool[: min(8, len(pool))]
    random_pool = pool[: min(64, len(pool))]
    random_trials = 5000
    random_meta = {
        "enabled": len(random_pool) >= 2,
        "trials": int(random_trials),
        "best_trial": None,
        "best_mode": None,
        "best_strategy": None,
    }

    for r in [1, 2, 3]:
        if len(coarse_pool) < r:
            continue
        for combo in itertools.combinations(range(len(coarse_pool)), r):
            grids = _weight_grid(r, step=step)
            oof_stack = [coarse_pool[i]["oof"] for i in combo]
            test_stack = [coarse_pool[i]["pred"] for i in combo]
            for weights in grids:
                blend_oof = np.zeros_like(oof_stack[0], dtype=np.float32)
                blend_test = np.zeros_like(test_stack[0], dtype=np.float32)
                for w, o, t in zip(weights, oof_stack, test_stack):
                    blend_oof += float(w) * np.asarray(o, dtype=np.float32)
                    blend_test += float(w) * np.asarray(t, dtype=np.float32)
                score = float(_score_predictions(y_valid, blend_oof))
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                    best_combo = combo
                    best_oof = blend_oof
                    best_test = blend_test

    if len(random_pool) >= 2:
        rng = np.random.default_rng(2026)
        oof_tensor = np.stack([np.asarray(c["oof"], dtype=np.float32) for c in random_pool], axis=0)
        test_tensor = np.stack([np.asarray(c["pred"], dtype=np.float32) for c in random_pool], axis=0)
        rank = np.arange(len(random_pool), dtype=np.float32)
        random_meta["pool_size"] = int(len(random_pool))

        strategy_specs = [
            {
                "name": "focused",
                "trials": int(random_trials * 0.45),
                "decay": 0.35,
                "min_k": 2,
                "max_k": min(12, len(random_pool)),
                "concentrations": np.array([0.25, 0.5, 0.8, 1.2], dtype=np.float32),
                "power_low": 0.85,
                "power_high": 1.35,
                "geo_prob": 0.55,
            },
            {
                "name": "broad",
                "trials": int(random_trials * 0.35),
                "decay": 0.16,
                "min_k": 3,
                "max_k": min(20, len(random_pool)),
                "concentrations": np.array([0.20, 0.35, 0.5, 0.8, 1.2], dtype=np.float32),
                "power_low": 0.80,
                "power_high": 1.80,
                "geo_prob": 0.40,
            },
            {
                "name": "dense",
                "trials": random_trials - int(random_trials * 0.45) - int(random_trials * 0.35),
                "decay": 0.10,
                "min_k": 6,
                "max_k": min(24, len(random_pool)),
                "concentrations": np.array([0.15, 0.25, 0.4, 0.6, 0.9, 1.2], dtype=np.float32),
                "power_low": 0.95,
                "power_high": 2.60,
                "geo_prob": 0.25,
            },
        ]

        global_trial_idx = 0
        for spec in strategy_specs:
            min_k = int(spec["min_k"])
            max_k = int(spec["max_k"])
            n_trials = int(spec["trials"])
            if n_trials <= 0 or max_k < min_k or max_k < 2:
                continue
            pick_prob = np.exp(-float(spec["decay"]) * rank)
            pick_prob = pick_prob / np.clip(pick_prob.sum(), 1e-9, None)

            for _ in range(n_trials):
                k = int(rng.integers(min_k, max_k + 1))
                idx = np.sort(rng.choice(len(random_pool), size=k, replace=False, p=pick_prob)).astype(np.int32)
                concentration = float(rng.choice(spec["concentrations"]))
                alpha = np.full(k, concentration, dtype=np.float32)
                weights = rng.dirichlet(alpha).astype(np.float32)
                power = float(rng.uniform(float(spec["power_low"]), float(spec["power_high"])))
                weights = np.power(weights, power).astype(np.float32)
                weights = (weights / np.clip(np.sum(weights), 1e-9, None)).astype(np.float32)

                blend_oof = np.tensordot(weights, oof_tensor[idx], axes=(0, 0)).astype(np.float32)
                blend_test = np.tensordot(weights, test_tensor[idx], axes=(0, 0)).astype(np.float32)
                score = float(_score_predictions(y_valid, blend_oof))
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                    best_combo = tuple(int(i) for i in idx.tolist())
                    best_oof = blend_oof
                    best_test = blend_test
                    random_meta["best_trial"] = int(global_trial_idx)
                    random_meta["best_mode"] = "linear"
                    random_meta["best_strategy"] = str(spec["name"])

                # Geometric mean in probability simplex often improves robustness for class imbalance.
                if float(spec["geo_prob"]) > 0.0 and float(rng.random()) <= float(spec["geo_prob"]):
                    log_oof = np.exp(
                        np.tensordot(weights, np.log(np.clip(oof_tensor[idx], 1e-9, 1.0)), axes=(0, 0))
                    ).astype(np.float32)
                    log_oof = log_oof / np.clip(log_oof.sum(axis=1, keepdims=True), 1e-9, None)
                    log_test = np.exp(
                        np.tensordot(weights, np.log(np.clip(test_tensor[idx], 1e-9, 1.0)), axes=(0, 0))
                    ).astype(np.float32)
                    log_test = log_test / np.clip(log_test.sum(axis=1, keepdims=True), 1e-9, None)
                    log_score = float(_score_predictions(y_valid, log_oof))
                    if log_score > best_score:
                        best_score = log_score
                        best_weights = weights.copy()
                        best_combo = tuple(int(i) for i in idx.tolist())
                        best_oof = log_oof
                        best_test = log_test
                        random_meta["best_trial"] = int(global_trial_idx)
                        random_meta["best_mode"] = "geometric"
                        random_meta["best_strategy"] = str(spec["name"])

                global_trial_idx += 1

    if best_weights is None or best_combo is None or best_oof is None or best_test is None:
        print("[FINAL_BLEND] Blend search failed to find a valid solution.")
        return

    power_refine = {"enabled": True, "applied": False, "best_power": 1.0, "base_valid_auc": float(best_score)}
    for power in np.arange(0.70, 1.31, 0.05, dtype=np.float32):
        p = float(power)
        if abs(p - 1.0) < 1e-9:
            continue
        candidate_oof = _power_normalize_proba(best_oof, p)
        candidate_score = float(_score_predictions(y_valid, candidate_oof))
        if candidate_score > best_score + 1e-8:
            best_score = candidate_score
            best_oof = candidate_oof
            best_test = _power_normalize_proba(best_test, p)
            power_refine["applied"] = True
            power_refine["best_power"] = float(p)

    class_weight_calibration = {
        "enabled": False,
        "applied": False,
        "base_valid_auc": float(best_score),
    }
    if n_classes > 2 and _resolve_class_weight_calibration_enabled(paths, default=True):
        grid = _resolve_class_weight_calibration_grid(paths)
        random_trials = _resolve_class_weight_calibration_trials(paths, default=120)
        class_weight_calibration["enabled"] = True
        cal_oof, cal_weights, cal_score, n_evaluated = _optimize_class_probability_weights(
            y_true=y_valid,
            pred=best_oof,
            grid_values=grid,
            random_trials=random_trials,
            random_state=42,
        )
        class_weight_calibration["weights"] = [float(v) for v in cal_weights.tolist()]
        class_weight_calibration["n_weight_candidates_evaluated"] = int(n_evaluated)
        class_weight_calibration["calibrated_valid_auc"] = float(cal_score)
        if cal_score > best_score + 1e-8:
            best_oof = cal_oof
            best_test = _apply_class_probability_weights(best_test, cal_weights)
            best_score = float(cal_score)
            class_weight_calibration["applied"] = True

    decision_calibration = {
        "enabled": bool(n_classes > 1),
        "applied": False,
        "base_valid_auc": float(best_score),
    }
    if n_classes > 1:
        cal_oof, cal_bias, cal_temp, cal_score, n_eval = _optimize_logit_bias_temperature(
            y_true=y_valid,
            pred=best_oof,
            random_trials=max(2000, _resolve_class_weight_calibration_trials(paths, default=120)),
            random_state=73,
        )
        decision_calibration["bias"] = [float(v) for v in cal_bias.tolist()]
        decision_calibration["temperature"] = float(cal_temp)
        decision_calibration["n_candidates_evaluated"] = int(n_eval)
        decision_calibration["calibrated_valid_auc"] = float(cal_score)
        if cal_score > best_score + 1e-8:
            best_oof = cal_oof
            best_test = _apply_logit_bias_temperature(best_test, bias=cal_bias, temperature=cal_temp)
            best_score = float(cal_score)
            decision_calibration["applied"] = True

    out_dir = paths.level2_results_dir / "final_blend"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "final_blend_valid_pred.npy", best_oof.astype(np.float32))
    np.save(out_dir / "final_blend_test_pred.npy", best_test.astype(np.float32))
    submission_path = _write_submission_from_proba(
        paths=paths,
        pred=best_test,
        class_labels=class_labels,
        target_col=target_col,
        filename="submission_final_blend.csv",
    )
    best_single_submission = _write_submission_from_proba(
        paths=paths,
        pred=np.asarray(best_single["pred"]),
        class_labels=class_labels,
        target_col=target_col,
        filename="submission_best_single_valid.csv",
    )

    blend_items: list[dict[str, object]] = []
    for weight, idx in zip(best_weights, best_combo):
        cand = random_pool[idx] if idx < len(random_pool) else pool[idx]
        blend_items.append(
            {
                "name": str(cand["name"]),
                "weight": float(weight),
                "candidate_valid_auc": float(cand["individual_auc"]),
            }
        )

    summary = {
        "target_col": target_col,
        "best_valid_auc": float(best_score),
        "best_single_name": str(best_single["name"]),
        "best_single_valid_auc": float(best_single["individual_auc"]),
        "blend_candidates": [
            {"name": str(c["name"]), "valid_auc": float(c["individual_auc"])}
            for c in random_pool
        ],
        "selected_blend": blend_items,
        "grid_step": float(step),
        "coarse_grid_pool_size": int(len(coarse_pool)),
        "random_pool_size": int(len(random_pool)),
        "random_search": random_meta,
        "power_refine": power_refine,
        "class_weight_calibration": class_weight_calibration,
        "decision_calibration": decision_calibration,
        "submission_path": str(submission_path),
        "best_single_submission_path": str(best_single_submission),
    }
    (out_dir / "final_blend_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        _archive_final_blend_artifacts(
            paths=paths,
            valid_pred=best_oof,
            test_pred=best_test,
            summary=summary,
            max_keep=24,
        )
    except Exception:
        pass
    print(f"[FINAL_BLEND] best_valid_auc={best_score:.6f}")
    print(f"[FINAL_BLEND] Saved artifacts in: {out_dir}")
    print(f"[FINAL_BLEND] Submission: {submission_path}")


def _pseudo_labeling() -> None:
    paths = get_paths()
    class_labels, _ = _load_target_mapping(paths)
    ranked_sources: list[tuple[float, str, Path]] = []

    stacking_pred = paths.level2_results_dir / "stacking" / "stacked_test_pred.npy"
    stacking_metrics = paths.level2_results_dir / "stacking" / "stacking_metrics.json"
    if stacking_pred.exists() and stacking_metrics.exists():
        try:
            m = json.loads(stacking_metrics.read_text(encoding="utf-8"))
            ranked_sources.append((float(m.get("valid_auc", 0.0)), "stacking", stacking_pred))
        except Exception:
            pass

    hc_pred = paths.level2_results_dir / "hill_climb" / "hill_climb_test_pred.npy"
    hc_metrics = paths.level2_results_dir / "hill_climb" / "hill_climb_selection.json"
    if hc_pred.exists() and hc_metrics.exists():
        try:
            m = json.loads(hc_metrics.read_text(encoding="utf-8"))
            ranked_sources.append((float(m.get("final_valid_auc", 0.0)), "hill_climb", hc_pred))
        except Exception:
            pass

    final_pred = paths.level2_results_dir / "final_blend" / "final_blend_test_pred.npy"
    final_summary = paths.level2_results_dir / "final_blend" / "final_blend_summary.json"
    if final_pred.exists() and final_summary.exists():
        try:
            m = json.loads(final_summary.read_text(encoding="utf-8"))
            ranked_sources.append((float(m.get("best_valid_auc", 0.0)), "final_blend", final_pred))
        except Exception:
            pass

    for level4_pred in sorted(paths.level2_results_dir.glob("*/level4_test_pred.npy"), key=lambda p: p.stat().st_mtime, reverse=True):
        metrics_path = level4_pred.with_name("level4_final_metrics.json")
        if not metrics_path.exists():
            continue
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            ranked_sources.append((float(m.get("valid_auc", 0.0)), f"level4::{level4_pred.parent.name}", level4_pred))
        except Exception:
            continue

    if not ranked_sources:
        print("[PSEUDO_LABELING] No test prediction source found.")
        return

    ranked_sources.sort(key=lambda t: t[0], reverse=True)
    best_auc, best_source, pred_path = ranked_sources[0]
    pred = np.asarray(np.load(pred_path), dtype=np.float32)
    n_classes = max(2, len(class_labels) or (pred.shape[1] if pred.ndim == 2 else 2))
    proba = _ensure_proba_2d(pred, n_classes=n_classes)
    conf_all = np.max(proba, axis=1).astype(np.float32)
    label_idx_all = np.argmax(proba, axis=1).astype(np.int32)

    thresholds = [0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96, 0.95]
    min_rows = max(250, int(0.005 * len(proba)))
    keep_thr = thresholds[-1]
    mask = conf_all >= keep_thr
    for thr in thresholds:
        candidate_mask = conf_all >= thr
        if int(np.sum(candidate_mask)) >= min_rows:
            keep_thr = thr
            mask = candidate_mask
            break

    idx = np.where(mask)[0]
    if idx.size == 0:
        keep_n = max(250, int(0.01 * len(pred)))
        top_idx = np.argsort(-conf_all)[:keep_n]
        idx = np.sort(top_idx.astype(np.int64))
    else:
        max_keep = max(250, int(0.15 * len(pred)))
        if idx.size > max_keep:
            conf_sel = conf_all[idx]
            keep_order = np.argsort(-conf_sel)[:max_keep]
            idx = np.sort(idx[keep_order].astype(np.int64))

    labels_idx = label_idx_all[idx].astype(np.int32)
    conf = conf_all[idx].astype(np.float32)
    labels = (
        np.array([class_labels[i] for i in labels_idx], dtype=object)
        if class_labels
        else labels_idx.astype(str)
    )

    out_dir = paths.level2_results_dir / "pseudo_labeling"
    out_dir.mkdir(parents=True, exist_ok=True)
    pseudo_df = pd.DataFrame(
        {
            "row_index": idx.astype(np.int64),
            "pseudo_label_idx": labels_idx,
            "pseudo_label": labels,
            "score_max": conf,
            "confidence": conf,
        }
    )
    _save_parquet(pseudo_df, out_dir / "pseudo_labels.parquet")
    summary = {
        "source_prediction_file": str(pred_path),
        "source_name": best_source,
        "source_valid_auc": float(best_auc),
        "confidence_threshold": float(keep_thr),
        "selected_rows": int(len(idx)),
        "selection_ratio": float(len(idx) / max(1, len(pred))),
    }
    (out_dir / "pseudo_label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[PSEUDO_LABELING] Selected {len(idx)} pseudo labels from {len(pred)} test rows.")
    print(f"[PSEUDO_LABELING] Saved artifacts in: {out_dir}")


def _extra_training() -> None:
    paths = get_paths()
    feature_sets = [p for p in paths.level1_features_dir.glob("*") if p.is_dir()]
    if not feature_sets:
        print("[EXTRA_TRAINING] No Level-1 feature set found. Run feature_engineering first.")
        return
    feature_root = sorted(feature_sets, key=lambda p: p.stat().st_mtime)[-1]

    train_path = feature_root / "train.parquet"
    valid_path = feature_root / "valid.parquet"
    test_path = feature_root / "test.parquet"
    if not train_path.exists() or not valid_path.exists() or not test_path.exists():
        print("[EXTRA_TRAINING] Missing Level-1 feature split files.")
        return

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    test_df = pd.read_parquet(test_path)
    target_col = _infer_target_column(train_df)
    class_labels, class_to_index = _load_target_mapping(paths)
    if not class_to_index:
        class_labels, class_to_index = _target_mapping_from_series(train_df[target_col])

    full_train = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    pseudo_path = paths.level2_results_dir / "pseudo_labeling" / "pseudo_labels.parquet"
    pseudo_rows_added = 0
    if pseudo_path.exists():
        pseudo_df = pd.read_parquet(pseudo_path)
        if not pseudo_df.empty and "row_index" in pseudo_df.columns:
            row_idx = pseudo_df["row_index"].astype(int).to_numpy()
            row_idx = row_idx[(row_idx >= 0) & (row_idx < len(test_df))]
            if len(row_idx) > 0:
                pseudo_rows = test_df.iloc[row_idx].copy()
                if "pseudo_label_idx" in pseudo_df.columns:
                    pseudo_rows[target_col] = pseudo_df.iloc[: len(row_idx)]["pseudo_label_idx"].astype(np.int32).to_numpy()
                else:
                    pseudo_rows[target_col] = (
                        pseudo_df.iloc[: len(row_idx)]["pseudo_label"]
                        .astype("string")
                        .map(class_to_index)
                        .fillna(0)
                        .astype(np.int32)
                        .to_numpy()
                    )
                full_train = pd.concat([full_train, pseudo_rows], axis=0, ignore_index=True)
                pseudo_rows_added = int(len(pseudo_rows))

    full_train_target = (
        full_train[target_col]
        if pd.api.types.is_numeric_dtype(full_train[target_col])
        else full_train[target_col].astype("string").map(class_to_index).fillna(0).astype(np.int32)
    )
    full_train[target_col] = np.asarray(full_train_target, dtype=np.int32)

    x_train_enc, _, x_test_enc, used_cols = _encode_level_features(
        train_df=full_train,
        valid_df=full_train,
        test_df=test_df,
        target_col=target_col,
    )
    y_train = _as_binary_target(full_train[target_col]).astype(np.int32)

    out_dir = paths.level2_results_dir / "extra_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    if XGBClassifier is None or cp is None:
        raise RuntimeError("[EXTRA_TRAINING] strict GPU mode requires xgboost + cupy.")

    n_classes = max(2, len(class_labels) or (int(np.max(y_train)) + 1))
    xgb_params: dict[str, object] = {
        "device": "cuda",
        "tree_method": "hist",
        "n_jobs": -1,
        "n_estimators": 360,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
    }
    if n_classes > 2:
        xgb_params.update({"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": int(n_classes)})
    else:
        xgb_params.update({"objective": "binary:logistic", "eval_metric": "auc"})
    model = XGBClassifier(**xgb_params)
    x_tr_gpu = _to_cupy_frame(x_train_enc)
    x_te_gpu = _to_cupy_frame(x_test_enc)
    model.fit(x_tr_gpu, y_train.astype(np.int32))
    test_pred = _predict_score(model, x_te_gpu, n_classes=n_classes)
    backend = "xgboost_gpu_cuda"

    np.save(out_dir / "extra_training_test_pred.npy", test_pred.astype(np.float32))
    np.save(paths.pred_dir / f"pred_extra_training_{feature_root.name}.npy", test_pred.astype(np.float32))
    submissions_dir = paths.outputs_root / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    split_summary_path = paths.processed_data / "split_summary.json"
    id_col = "id"
    if split_summary_path.exists():
        try:
            payload = json.loads(split_summary_path.read_text(encoding="utf-8"))
            id_col = str(payload.get("id_col", "id"))
        except Exception:
            id_col = "id"
    pred_labels = _proba_to_labels(test_pred, class_labels)
    if id_col in test_df.columns:
        submission_id = test_df[id_col].to_numpy()
    else:
        submission_id = np.arange(len(test_df), dtype=np.int64)
    submission = pd.DataFrame({id_col: submission_id, target_col: pred_labels})
    submission_path = submissions_dir / "submission_extra_training.csv"
    submission.to_csv(submission_path, index=False)

    metrics = {
        "feature_set_name": feature_root.name,
        "target_col": target_col,
        "backend": backend,
        "n_train_rows": int(len(full_train)),
        "pseudo_rows_added": int(pseudo_rows_added),
        "n_features_used": int(len(used_cols)),
        "n_classes": int(n_classes),
        "submission_path": str(submission_path),
    }
    (out_dir / "extra_training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[EXTRA_TRAINING] backend={backend} | train_rows={len(full_train)} | pseudo_rows_added={pseudo_rows_added}")
    print(f"[EXTRA_TRAINING] Saved artifacts in: {out_dir}")


def _kgmon_playbook() -> None:
    _prepare_data(PrepareDataConfig())
    _feature_engineering(FeatureEngineeringConfig())
    _baseline(BaselineConfig())
    _eda()
    _hill_climb()
    _stacking()
    _final_blend()
    _pseudo_labeling()
    _extra_training()


STAGES: dict[str, StageFn] = {
    "prepare_data": _prepare_data,
    "feature_engineering": _feature_engineering,
    "gpu_feature_engineering": _feature_engineering,
    "baseline": _baseline,
    "build_baselines": _baseline,
    "eda": _eda,
    "hill_climb": _hill_climb,
    "gpu_hill_climbing": _hill_climb,
    "stacking": _stacking,
    "final_blend": _final_blend,
    "pseudo_labeling": _pseudo_labeling,
    "extra_training": _extra_training,
    "gpu_extra_training": _extra_training,
    "kgmon_playbook": _kgmon_playbook,
}


def run_stage(stage: str) -> None:
    _ensure_base_dirs()
    if stage not in STAGES:
        valid = ", ".join(STAGES)
        raise ValueError(f"Unknown stage '{stage}'. Use one of: {valid}")
    STAGES[stage]()


def run_stage_with_config(
    stage: str,
    prepare_data_config: dict | None = None,
    feature_engineering_config: dict | None = None,
    baseline_config: dict | None = None,
) -> None:
    _ensure_base_dirs()
    if stage not in STAGES:
        valid = ", ".join(STAGES)
        raise ValueError(f"Unknown stage '{stage}'. Use one of: {valid}")

    if stage == "prepare_data":
        _prepare_data(PrepareDataConfig(**(prepare_data_config or {})))
        return

    if stage in {"feature_engineering", "gpu_feature_engineering"}:
        _feature_engineering(FeatureEngineeringConfig(**(feature_engineering_config or {})))
        return

    if stage in {"baseline", "build_baselines"}:
        _baseline(BaselineConfig(**(baseline_config or {})))
        return

    if stage == "kgmon_playbook":
        _prepare_data(PrepareDataConfig(**(prepare_data_config or {})))
        _feature_engineering(FeatureEngineeringConfig(**(feature_engineering_config or {})))
        _baseline(BaselineConfig(**(baseline_config or {})))
        _eda()
        _hill_climb()
        _stacking()
        _final_blend()
        _pseudo_labeling()
        _extra_training()
        return

    STAGES[stage]()
