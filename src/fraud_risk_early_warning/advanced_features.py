from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection


DEFAULT_TELCO_CATEGORICAL_COLUMNS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]

DEFAULT_TELCO_SERVICE_COLUMNS = [
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]

DEFAULT_TE_STATS = [
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
]

DEFAULT_FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "target_encoding": ("te__", "te_prior__"),
    "arithmetic": ("arith__",),
    "multi_scale_binning": ("bin__",),
    "cross_features": ("cross__",),
    "frequency_count": ("freq__",),
    "service_aggregations": ("svc__",),
    "original_lookup": ("lookup__",),
    "radix_interactions": ("radix__",),
    "artifact_detection": ("artifact__",),
    "projection_manifold": ("proj__",),
}


@dataclass
class AdvancedFeatureResult:
    features_by_split: dict[str, pd.DataFrame]
    family_feature_counts: dict[str, int]
    family_columns: dict[str, list[str]]
    metadata: dict[str, Any]


def _binary_target(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    s = series.astype("string").fillna("__NA__")
    codes = pd.factorize(s, sort=True)[0]
    return codes.astype(np.float32)


def _existing_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _as_category_key(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__NA__")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_numeric_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)


def _fit_quantile_bins(train_s: pd.Series, n_bins: int) -> np.ndarray | None:
    s = train_s.dropna()
    if s.empty:
        return None
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(s.to_numpy(dtype=np.float64), q))
    if len(edges) < 2:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _fit_fixed_bins(train_s: pd.Series, n_bins: int) -> np.ndarray | None:
    s = train_s.dropna()
    if s.empty:
        return None
    min_v = float(s.min())
    max_v = float(s.max())
    if not np.isfinite(min_v) or not np.isfinite(max_v) or min_v == max_v:
        return None
    edges = np.linspace(min_v, max_v, n_bins + 1, dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _fit_log_bins(train_s: pd.Series, n_bins: int) -> tuple[np.ndarray, float] | None:
    s = train_s.dropna()
    if s.empty:
        return None
    min_v = float(s.min())
    shift = 0.0
    if min_v <= 0:
        shift = abs(min_v) + 1.0
    transformed = np.log1p(s + shift)
    edges = _fit_fixed_bins(transformed, n_bins=n_bins)
    if edges is None:
        return None
    return edges, shift


def _digit_decimal_snap(df: pd.DataFrame, column: str, factor: int = 100) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    x = _to_numeric(df[column]).fillna(0.0)
    return (np.round(x * factor) / factor).astype(np.float32)


def _build_categorical_signal_frames(
    split_frames: dict[str, pd.DataFrame],
    categorical_columns: list[str],
    numeric_columns_for_bins: list[str],
    monthly_col: str,
    total_col: str,
    quantile_bins: list[int],
    fixed_bins: list[int],
    log_bins: list[int],
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], dict[str, Any]]:
    splits = ["train", "valid", "test", "train_folds"]
    out: dict[str, pd.DataFrame] = {
        s: pd.DataFrame(index=split_frames[s].index) for s in splits if s in split_frames
    }
    metadata: dict[str, Any] = {"binning": {}}

    train_df = split_frames["train"]
    present_cats = _existing_columns(train_df, categorical_columns)

    for col in present_cats:
        for split in out:
            out[split][f"cat__raw__{col}"] = _as_category_key(split_frames[split][col])

    # Bigrams/trigrams on requested high-signal pairs.
    pair_specs = [
        ("Soil_Type", "Crop_Type"),
        ("Season", "Region"),
        ("Irrigation_Type", "Water_Source"),
    ]
    trigram_specs = [
        ("Soil_Type", "Season", "Region"),
    ]

    for a, b in pair_specs:
        if a in present_cats and b in present_cats:
            name = f"cat__bi__{a}__{b}"
            for split in out:
                sa = _as_category_key(split_frames[split][a])
                sb = _as_category_key(split_frames[split][b])
                out[split][name] = sa + "__" + sb

    for a, b, c in trigram_specs:
        if a in present_cats and b in present_cats and c in present_cats:
            name = f"cat__tri__{a}__{b}__{c}"
            for split in out:
                sa = _as_category_key(split_frames[split][a])
                sb = _as_category_key(split_frames[split][b])
                sc = _as_category_key(split_frames[split][c])
                out[split][name] = sa + "__" + sb + "__" + sc

    # Numeric snapping helpers used in anchors / radix / interaction products.
    for split in out:
        out[split]["num__MC_snap"] = _digit_decimal_snap(split_frames[split], monthly_col)
        out[split]["num__TC_snap"] = _digit_decimal_snap(split_frames[split], total_col)

    # Multi-scale bins (as category-strings).
    present_nums = _existing_columns(train_df, numeric_columns_for_bins)
    for col in present_nums:
        train_series = _to_numeric(train_df[col])

        for n_bins in quantile_bins:
            if n_bins <= 1:
                continue
            edges = _fit_quantile_bins(train_series, n_bins)
            if edges is None:
                continue
            key = f"cat__bin_q{n_bins}__{col}"
            metadata["binning"][key] = {
                "type": "quantile",
                "n_bins": int(n_bins),
                "n_edges": int(len(edges)),
            }
            for split in out:
                s = _to_numeric(split_frames[split][col])
                b = pd.cut(s, bins=edges, labels=False, include_lowest=True)
                out[split][key] = b.fillna(-1).astype("int32").astype("string")

        for n_bins in fixed_bins:
            if n_bins <= 1:
                continue
            edges = _fit_fixed_bins(train_series, n_bins)
            if edges is None:
                continue
            key = f"cat__bin_w{n_bins}__{col}"
            metadata["binning"][key] = {
                "type": "fixed_width",
                "n_bins": int(n_bins),
                "n_edges": int(len(edges)),
            }
            for split in out:
                s = _to_numeric(split_frames[split][col])
                b = pd.cut(s, bins=edges, labels=False, include_lowest=True)
                out[split][key] = b.fillna(-1).astype("int32").astype("string")

        for n_bins in log_bins:
            if n_bins <= 1:
                continue
            fit = _fit_log_bins(train_series, n_bins)
            if fit is None:
                continue
            edges, shift = fit
            key = f"cat__bin_log{n_bins}__{col}"
            metadata["binning"][key] = {
                "type": "log_scale",
                "n_bins": int(n_bins),
                "n_edges": int(len(edges)),
                "shift": float(shift),
            }
            for split in out:
                s = _to_numeric(split_frames[split][col])
                log_s = np.log1p(s + shift)
                b = pd.cut(log_s, bins=edges, labels=False, include_lowest=True)
                out[split][key] = b.fillna(-1).astype("int32").astype("string")

        int_key = f"cat__bin_floor__{col}"
        for split in out:
            s = _to_numeric(split_frames[split][col]).fillna(0.0)
            out[split][int_key] = np.floor(s).astype("int64").astype("string")

    # Anchor keys and numeric x categorical snap products.
    anchor_cols = [c for c in ["Contract", "PaymentMethod", "InternetService"] if c in present_cats]
    if anchor_cols:
        for split in out:
            mc = out[split]["num__MC_snap"].fillna(-999.0).astype("float32")
            tc = out[split]["num__TC_snap"].fillna(-999.0).astype("float32")
            mc_bin = np.floor(mc * 10).astype("int32").astype("string")
            tc_bin = np.floor(tc * 10).astype("int32").astype("string")
            out[split]["cat__anchor__mc_tc"] = mc_bin + "__" + tc_bin

            for c in anchor_cols:
                cat = _as_category_key(split_frames[split][c])
                out[split][f"cat__anchor__{c}__mc"] = cat + "__" + mc_bin
                out[split][f"cat__anchor__{c}__tc"] = cat + "__" + tc_bin
                # numeric x categorical snap product represented as category string
                out[split][f"cat__snapprod__{c}__mc"] = cat + "__" + mc.round(2).astype("string")
                out[split][f"cat__snapprod__{c}__tc"] = cat + "__" + tc.round(2).astype("string")

    # Categorical cross features as integer codes (for direct tree usage).
    for col in [c for c in out["train"].columns if c.startswith("cat__bi__") or c.startswith("cat__tri__")]:
        all_values = pd.concat([out[s][col] for s in out], axis=0).astype("string")
        categories = pd.Index(all_values.dropna().unique())
        cat_dtype = pd.CategoricalDtype(categories=categories)
        for split in out:
            out[split][f"cross__{col[5:]}__code"] = (
                out[split][col].astype(cat_dtype).cat.codes.astype("int32")
            )

    # Build family column map from category frame names.
    family_cats = {
        "raw_categorical": [c for c in out["train"].columns if c.startswith("cat__raw__")],
        "cross_categorical": [
            c
            for c in out["train"].columns
            if c.startswith("cat__bi__") or c.startswith("cat__tri__")
        ],
        "binned_categorical": [c for c in out["train"].columns if c.startswith("cat__bin_")],
        "anchor_categorical": [c for c in out["train"].columns if c.startswith("cat__anchor__")],
        "snapprod_categorical": [c for c in out["train"].columns if c.startswith("cat__snapprod__")],
    }

    return out, family_cats, metadata


def _compute_te_stats(keys: pd.Series, target: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"key": keys, "target": target})
    grouped = d.groupby("key", dropna=False)["target"]

    stats = pd.DataFrame(index=grouped.mean().index)
    stats["mean"] = grouped.mean().astype(np.float32)
    stats["std"] = grouped.std().fillna(0.0).astype(np.float32)
    stats["min"] = grouped.min().astype(np.float32)
    stats["max"] = grouped.max().astype(np.float32)
    stats["median"] = grouped.median().astype(np.float32)
    stats["q05"] = grouped.quantile(0.05).astype(np.float32)
    stats["q10"] = grouped.quantile(0.10).astype(np.float32)
    stats["q45"] = grouped.quantile(0.45).astype(np.float32)
    stats["q55"] = grouped.quantile(0.55).astype(np.float32)
    stats["q90"] = grouped.quantile(0.90).astype(np.float32)
    stats["q95"] = grouped.quantile(0.95).astype(np.float32)
    return stats


def _fold_assignments(train_df: pd.DataFrame, target_col: str, fold_col: str, n_splits: int) -> np.ndarray:
    if fold_col in train_df.columns:
        fold_values = pd.to_numeric(train_df[fold_col], errors="coerce").fillna(-1).astype(int).to_numpy()
        if np.all(fold_values >= 0):
            return fold_values

    y = _binary_target(train_df[target_col])
    out = np.full(len(train_df), -1, dtype=np.int32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        out[val_idx] = fold
    return out


def _nested_target_encoding(
    split_frames: dict[str, pd.DataFrame],
    cat_frames: dict[str, pd.DataFrame],
    target_col: str,
    te_stats: list[str],
    inner_folds: int,
    fold_col: str,
    original_df: pd.DataFrame | None,
    original_target_col: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[str]]:
    out_dict: dict[str, dict[str, pd.Series]] = {k: {} for k in cat_frames}
    train_target = pd.Series(_binary_target(split_frames["train"][target_col]), index=cat_frames["train"].index)

    if "train_folds" in split_frames and len(split_frames["train_folds"]) == len(split_frames["train"]):
        fold_source = split_frames["train_folds"]
    else:
        fold_source = split_frames["train"]

    folds = _fold_assignments(fold_source, target_col=target_col, fold_col=fold_col, n_splits=inner_folds)
    unique_folds = [int(f) for f in sorted(set(folds.tolist())) if f >= 0]
    if not unique_folds:
        unique_folds = list(range(inner_folds))

    te_columns: list[str] = []
    te_meta: dict[str, Any] = {"encoded_keys": 0, "stats": te_stats, "inner_folds": unique_folds}

    for cat_col in [c for c in cat_frames["train"].columns if c.startswith("cat__")]:
        train_keys = _as_category_key(cat_frames["train"][cat_col])
        valid_keys = _as_category_key(cat_frames["valid"][cat_col]) if "valid" in cat_frames else None
        test_keys = _as_category_key(cat_frames["test"][cat_col]) if "test" in cat_frames else None
        train_folds_keys = (
            _as_category_key(cat_frames["train_folds"][cat_col]) if "train_folds" in cat_frames else None
        )

        full_stats = _compute_te_stats(train_keys, train_target)
        full_global = {
            k: float(train_target.mean()) if k in {"mean", "median", "q45", "q55"} else 0.0
            for k in te_stats
        }
        full_global["min"] = float(train_target.min())
        full_global["max"] = float(train_target.max())
        full_global["q05"] = float(np.quantile(train_target, 0.05))
        full_global["q10"] = float(np.quantile(train_target, 0.10))
        full_global["q90"] = float(np.quantile(train_target, 0.90))
        full_global["q95"] = float(np.quantile(train_target, 0.95))

        # OOF train encoding with nested leak-free folds.
        oof_block = {stat: np.full(len(train_keys), np.nan, dtype=np.float32) for stat in te_stats}
        for fold in unique_folds:
            val_mask = folds == fold
            tr_mask = folds != fold
            if val_mask.sum() == 0 or tr_mask.sum() == 0:
                continue
            fold_stats = _compute_te_stats(train_keys[tr_mask], train_target[tr_mask])
            val_keys = train_keys[val_mask]
            for stat in te_stats:
                mapped = val_keys.map(fold_stats[stat]).astype(np.float32)
                oof_block[stat][val_mask] = mapped.fillna(np.float32(full_global[stat])).to_numpy()

        for stat in te_stats:
            fname = f"te__{cat_col[5:]}__{stat}"
            out_dict["train"][fname] = pd.Series(
                oof_block[stat], index=cat_frames["train"].index
            ).fillna(np.float32(full_global[stat]))
            te_columns.append(fname)

            if valid_keys is not None:
                out_dict["valid"][fname] = valid_keys.map(full_stats[stat]).astype(np.float32).fillna(
                    np.float32(full_global[stat])
                )
            if test_keys is not None:
                out_dict["test"][fname] = test_keys.map(full_stats[stat]).astype(np.float32).fillna(
                    np.float32(full_global[stat])
                )
            if train_folds_keys is not None:
                out_dict["train_folds"][fname] = train_folds_keys.map(full_stats[stat]).astype(
                    np.float32
                ).fillna(np.float32(full_global[stat]))

        # Original-data churn prior per raw categorical value (no synthetic label leakage).
        if original_df is not None and cat_col.startswith("cat__raw__"):
            raw_col = cat_col.removeprefix("cat__raw__")
            if raw_col in original_df.columns and original_target_col in original_df.columns:
                prior_target = pd.Series(_binary_target(original_df[original_target_col]), index=original_df.index)
                prior_keys = _as_category_key(original_df[raw_col])
                prior_map = pd.DataFrame({"key": prior_keys, "target": prior_target}).groupby("key")[
                    "target"
                ].mean()
                prior_global = float(prior_target.mean())
                prior_name = f"te_prior__{raw_col}"
                for split in out_dict:
                    split_keys = _as_category_key(cat_frames[split][cat_col])
                    out_dict[split][prior_name] = split_keys.map(prior_map).astype(np.float32).fillna(
                        np.float32(prior_global)
                    )
                te_columns.append(prior_name)

    te_meta["encoded_keys"] = int(len([c for c in cat_frames["train"].columns if c.startswith("cat__")]))
    out = {
        split: pd.DataFrame(data=cols, index=cat_frames[split].index)
        for split, cols in out_dict.items()
    }
    return out, te_meta, sorted(set(te_columns))


def _add_arithmetic_features(
    split_frames: dict[str, pd.DataFrame],
    monthly_col: str,
    total_col: str,
    tenure_col: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    out = {k: pd.DataFrame(index=v.index) for k, v in split_frames.items()}
    cols: list[str] = []

    for split, df in split_frames.items():
        mc = _to_numeric(df[monthly_col]) if monthly_col in df.columns else pd.Series(0.0, index=df.index)
        tc = _to_numeric(df[total_col]) if total_col in df.columns else pd.Series(0.0, index=df.index)
        tenure = _to_numeric(df[tenure_col]) if tenure_col in df.columns else pd.Series(0.0, index=df.index)

        mc_snap = (np.round(mc.fillna(0.0) * 100) / 100).astype(np.float32)
        tc_snap = (np.round(tc.fillna(0.0) * 100) / 100).astype(np.float32)
        tenure_f = tenure.fillna(0.0).astype(np.float32)

        out[split]["arith__tc_deviation"] = (tc.fillna(0.0) - tenure_f * mc.fillna(0.0)).astype(np.float32)
        out[split]["arith__tc_snap_exp_dev"] = (tc_snap - tenure_f * mc_snap).astype(np.float32)
        out[split]["arith__tc_per_month"] = (tc.fillna(0.0) / (tenure_f + 1.0)).astype(np.float32)
        out[split]["arith__mc_to_tc_ratio"] = (mc.fillna(0.0) / (tc.fillna(0.0) + 1e-9)).astype(np.float32)
        out[split]["arith__mc_x_tenure"] = (mc.fillna(0.0) * (tenure_f + 1.0)).astype(np.float32)

    cols = list(out["train"].columns) if "train" in out else []
    return out, cols


def _add_multi_scale_numeric_bins(
    split_frames: dict[str, pd.DataFrame],
    numeric_columns: list[str],
    quantile_bins: list[int],
    fixed_bins: list[int],
    log_bins: list[int],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    out = {k: pd.DataFrame(index=v.index) for k, v in split_frames.items()}
    per_split_features: dict[str, dict[str, pd.Series]] = {k: {} for k in split_frames}
    train_df = split_frames["train"]

    present_nums = _existing_columns(train_df, numeric_columns)
    for col in present_nums:
        train_s = _to_numeric(train_df[col])

        for n_bins in quantile_bins:
            edges = _fit_quantile_bins(train_s, n_bins)
            if edges is None:
                continue
            name = f"bin__{col}__q{n_bins}"
            for split in out:
                s = _to_numeric(split_frames[split][col])
                b = pd.cut(s, bins=edges, labels=False, include_lowest=True)
                per_split_features[split][name] = b.fillna(-1).astype("int32")

        for n_bins in fixed_bins:
            edges = _fit_fixed_bins(train_s, n_bins)
            if edges is None:
                continue
            name = f"bin__{col}__w{n_bins}"
            for split in out:
                s = _to_numeric(split_frames[split][col])
                b = pd.cut(s, bins=edges, labels=False, include_lowest=True)
                per_split_features[split][name] = b.fillna(-1).astype("int32")

        for n_bins in log_bins:
            fit = _fit_log_bins(train_s, n_bins)
            if fit is None:
                continue
            edges, shift = fit
            name = f"bin__{col}__log{n_bins}"
            for split in out:
                s = _to_numeric(split_frames[split][col])
                log_s = np.log1p(s + shift)
                b = pd.cut(log_s, bins=edges, labels=False, include_lowest=True)
                per_split_features[split][name] = b.fillna(-1).astype("int32")

        floor_name = f"bin__{col}__floor"
        for split in out:
            s = _to_numeric(split_frames[split][col]).fillna(0.0)
            per_split_features[split][floor_name] = np.floor(s).astype("int32")

    for split in out:
        if per_split_features[split]:
            out[split] = pd.DataFrame(per_split_features[split], index=split_frames[split].index)
    return out, list(out["train"].columns)


def _add_frequency_features(
    split_frames: dict[str, pd.DataFrame],
    columns: list[str],
    original_df: pd.DataFrame | None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    out = {k: pd.DataFrame(index=v.index) for k, v in split_frames.items()}
    if not columns:
        return out, []
    per_split_features: dict[str, dict[str, pd.Series]] = {k: {} for k in split_frames}

    joined = pd.concat([
        pd.DataFrame({c: _as_category_key(split_frames[s][c]) for c in columns if c in split_frames[s].columns})
        for s in ["train", "valid", "test", "train_folds"]
        if s in split_frames
    ], axis=0, ignore_index=True)

    for col in [c for c in columns if c in joined.columns]:
        freq_map = joined[col].value_counts(normalize=True)
        count_map = joined[col].value_counts()
        orig_count_map = None
        if original_df is not None and col in original_df.columns:
            orig_count_map = _as_category_key(original_df[col]).value_counts()

        for split in out:
            if col not in split_frames[split].columns:
                continue
            key = _as_category_key(split_frames[split][col])
            freq = key.map(freq_map).astype(np.float32).fillna(0.0)
            count = key.map(count_map).astype(np.float32).fillna(0.0)
            per_split_features[split][f"freq__{col}__freq"] = freq
            per_split_features[split][f"freq__{col}__count"] = count
            if orig_count_map is not None:
                orig_count = key.map(orig_count_map).astype(np.float32).fillna(0.0)
                ratio = np.log1p((count + 1.0) / (orig_count + 1.0)).astype(np.float32)
                per_split_features[split][f"freq__{col}__drift_ratio"] = ratio

    for split in out:
        if per_split_features[split]:
            out[split] = pd.DataFrame(per_split_features[split], index=split_frames[split].index)

    return out, list(out["train"].columns)


def _add_service_aggregations(
    split_frames: dict[str, pd.DataFrame],
    service_columns: list[str],
    internet_col: str,
    phone_col: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    out = {k: pd.DataFrame(index=v.index) for k, v in split_frames.items()}

    for split, df in split_frames.items():
        svc_cols = [c for c in service_columns if c in df.columns]
        if svc_cols:
            yes_matrix = np.column_stack([
                _as_category_key(df[c]).str.lower().eq("yes").astype(np.int8).to_numpy() for c in svc_cols
            ])
            no_matrix = np.column_stack([
                _as_category_key(df[c]).str.lower().eq("no").astype(np.int8).to_numpy() for c in svc_cols
            ])
            yes_count = yes_matrix.sum(axis=1)
            no_count = no_matrix.sum(axis=1)
            out[split]["svc__yes_count"] = yes_count.astype(np.int16)
            out[split]["svc__no_count"] = no_count.astype(np.int16)
            out[split]["svc__other_count"] = (len(svc_cols) - yes_count - no_count).astype(np.int16)

            for c in svc_cols:
                s = _as_category_key(df[c]).str.lower()
                out[split][f"svc__{c}__is_yes"] = s.eq("yes").astype(np.int8)
                out[split][f"svc__{c}__is_no"] = s.eq("no").astype(np.int8)
                out[split][f"svc__{c}__is_other"] = (~s.isin(["yes", "no"])).astype(np.int8)

        if internet_col in df.columns:
            internet = _as_category_key(df[internet_col]).str.lower()
            out[split]["svc__has_internet"] = (~internet.eq("no")).astype(np.int8)
        if phone_col in df.columns:
            phone = _as_category_key(df[phone_col]).str.lower()
            out[split]["svc__has_phone"] = phone.eq("yes").astype(np.int8)

    return out, list(out["train"].columns)


def _add_original_lookup_features(
    split_frames: dict[str, pd.DataFrame],
    original_df: pd.DataFrame | None,
    monthly_col: str,
    total_col: str,
    tenure_col: str,
    original_target_col: str,
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, Any]]:
    out = {k: pd.DataFrame(index=v.index) for k, v in split_frames.items()}
    if original_df is None:
        return out, [], {"enabled": False, "reason": "original_reference_missing"}

    ref_cols = [c for c in [monthly_col, total_col, tenure_col] if c in original_df.columns]
    if len(ref_cols) < 2 or original_target_col not in original_df.columns:
        return out, [], {"enabled": False, "reason": "reference_columns_missing"}

    ref_num = original_df[ref_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mean = ref_num.mean(axis=0)
    std = ref_num.std(axis=0).replace(0.0, 1.0)
    ref_scaled = (ref_num - mean) / std

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(ref_scaled.to_numpy(dtype=np.float32))

    ref_target = _binary_target(original_df[original_target_col])

    for split, df in split_frames.items():
        curr = pd.DataFrame(index=df.index)
        q = df.reindex(columns=ref_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        q_scaled = (q - mean) / std

        dists, idx = nn.kneighbors(q_scaled.to_numpy(dtype=np.float32), return_distance=True)
        idx_1d = idx[:, 0]
        dist_1d = dists[:, 0]

        curr["lookup__nn_distance"] = dist_1d.astype(np.float32)
        curr["lookup__nn_label"] = ref_target[idx_1d].astype(np.float32)

        for c in ref_cols:
            nn_vals = ref_num[c].to_numpy(dtype=np.float32)[idx_1d]
            q_vals = q[c].to_numpy(dtype=np.float32)
            curr[f"lookup__nn_{c}"] = nn_vals
            curr[f"lookup__delta_{c}"] = (q_vals - nn_vals).astype(np.float32)

        out[split] = curr

    return out, list(out["train"].columns), {"enabled": True, "reference_rows": int(len(original_df))}


def _add_radix_features(
    split_frames: dict[str, pd.DataFrame],
    categorical_columns: list[str],
    monthly_col: str,
    total_col: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    out = {k: pd.DataFrame(index=v.index) for k, v in split_frames.items()}
    train_df = split_frames["train"]
    cats = _existing_columns(train_df, categorical_columns)
    if not cats:
        return out, []

    # Keep top few to control cardinality while still capturing strong interactions.
    cats = cats[:5]

    # Fit shared category codes on train values only.
    cat_maps: dict[str, dict[str, int]] = {}
    for c in cats:
        keys = _as_category_key(train_df[c]).value_counts().index.tolist()
        cat_maps[c] = {k: i + 1 for i, k in enumerate(keys)}

    for split, df in split_frames.items():
        mc_snap = _digit_decimal_snap(df, monthly_col, factor=100).fillna(0.0)
        tc_snap = _digit_decimal_snap(df, total_col, factor=100).fillna(0.0)
        mc_int = (mc_snap * 100).round().astype(np.int64)
        tc_int = (tc_snap * 100).round().astype(np.int64)

        for c in cats:
            codes = _as_category_key(df[c]).map(cat_maps[c]).fillna(0).astype(np.int64)
            out[split][f"radix__mc__{c}"] = (mc_int + (codes * 100_000)).astype(np.int64)
            out[split][f"radix__tc__{c}"] = (tc_int + (codes * 100_000)).astype(np.int64)

    return out, list(out["train"].columns)


def _leading_digit_benford_distance(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    expected = np.array([np.log10(1.0 + 1.0 / d) for d in range(1, 10)], dtype=np.float64)
    rows = len(df)
    probs = np.zeros((rows, 9), dtype=np.float64)

    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return pd.DataFrame(
            {
                "artifact__benford_l1": np.zeros(rows, dtype=np.float32),
                "artifact__benford_l2": np.zeros(rows, dtype=np.float32),
                "artifact__leading_digit_mean": np.zeros(rows, dtype=np.float32),
            },
            index=df.index,
        )

    digit_matrix: list[np.ndarray] = []
    for c in valid_cols:
        x = _to_numeric(df[c]).fillna(0.0).to_numpy(dtype=np.float64)
        abs_x = np.abs(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            powers = np.floor(np.log10(abs_x), where=abs_x > 0, out=np.zeros_like(abs_x))
            leading = np.where(abs_x > 0, np.floor(abs_x / (10 ** powers)).astype(np.int16), 0)
        leading = np.where((leading >= 1) & (leading <= 9), leading, 0)
        digit_matrix.append(leading)

    stacked = np.column_stack(digit_matrix)
    for d in range(1, 10):
        probs[:, d - 1] = (stacked == d).mean(axis=1)

    l1 = np.abs(probs - expected[None, :]).sum(axis=1)
    l2 = np.sqrt(np.square(probs - expected[None, :]).sum(axis=1))
    mean_digit = np.where((stacked > 0).sum(axis=1) > 0, np.where(stacked > 0, stacked, np.nan).mean(axis=1), 0)

    return pd.DataFrame(
        {
            "artifact__benford_l1": l1.astype(np.float32),
            "artifact__benford_l2": l2.astype(np.float32),
            "artifact__leading_digit_mean": np.nan_to_num(mean_digit, nan=0.0).astype(np.float32),
        },
        index=df.index,
    )


def _add_artifact_features(
    split_frames: dict[str, pd.DataFrame],
    numeric_columns: list[str],
    monthly_col: str,
    total_col: str,
    tfidf_max_features: int,
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, Any]]:
    out = {k: pd.DataFrame(index=v.index) for k, v in split_frames.items()}
    train_df = split_frames["train"]

    used_numeric = _existing_columns(train_df, numeric_columns)
    if not used_numeric:
        used_numeric = _existing_columns(train_df, [monthly_col, total_col])

    tfidf_text_by_split: dict[str, pd.Series] = {}

    for split, df in split_frames.items():
        numeric = _safe_numeric_matrix(df, used_numeric)
        if not numeric.empty:
            frac = np.abs(numeric.to_numpy(dtype=np.float64) - np.floor(numeric.to_numpy(dtype=np.float64)))
            intlike = (frac < 0.001).sum(axis=1)
            quarterlike = (
                (np.abs(frac - 0.25) < 0.001)
                | (np.abs(frac - 0.50) < 0.001)
                | (np.abs(frac - 0.75) < 0.001)
            ).sum(axis=1)
            halflike = (np.abs(frac - 0.5) < 0.001).sum(axis=1)

            denom = max(1, numeric.shape[1])
            out[split]["artifact__intlike_count"] = intlike.astype(np.int16)
            out[split]["artifact__quarterlike_count"] = quarterlike.astype(np.int16)
            out[split]["artifact__halflike_count"] = halflike.astype(np.int16)
            out[split]["artifact__intlike_ratio"] = (intlike / denom).astype(np.float32)
            out[split]["artifact__quarterlike_ratio"] = (quarterlike / denom).astype(np.float32)
            out[split]["artifact__halflike_ratio"] = (halflike / denom).astype(np.float32)

        benford_df = _leading_digit_benford_distance(df, used_numeric)
        out[split] = pd.concat([out[split], benford_df], axis=1)

        if monthly_col in df.columns and total_col in df.columns:
            mc = _to_numeric(df[monthly_col]).fillna(0.0).round(6).astype("string")
            tc = _to_numeric(df[total_col]).fillna(0.0).round(6).astype("string")
            tfidf_text_by_split[split] = (mc + "|" + tc).fillna("0|0")

    tfidf_meta = {"enabled": False, "n_features": 0}
    if "train" in tfidf_text_by_split:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=tfidf_max_features)
        train_matrix = vectorizer.fit_transform(tfidf_text_by_split["train"])
        tfidf_names = [f"artifact__tfidf_{i:02d}" for i in range(train_matrix.shape[1])]
        out["train"][tfidf_names] = train_matrix.toarray().astype(np.float32)

        for split in ["valid", "test", "train_folds"]:
            if split not in tfidf_text_by_split:
                continue
            mat = vectorizer.transform(tfidf_text_by_split[split])
            out[split][tfidf_names] = mat.toarray().astype(np.float32)

        tfidf_meta = {"enabled": True, "n_features": int(train_matrix.shape[1])}

    return out, list(out["train"].columns), {"tfidf": tfidf_meta, "numeric_cols": used_numeric}


def _add_projection_features(
    split_frames: dict[str, pd.DataFrame],
    original_df: pd.DataFrame | None,
    monthly_col: str,
    total_col: str,
    tenure_col: str,
    projection_components: int,
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, Any]]:
    out = {k: pd.DataFrame(index=v.index) for k, v in split_frames.items()}

    seed_df = split_frames["train"] if original_df is None else original_df
    proj_cols = [c for c in [monthly_col, total_col, tenure_col] if c in seed_df.columns]
    if not proj_cols:
        return out, [], {"enabled": False, "reason": "projection_columns_missing"}

    fit_df = seed_df[proj_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mean = fit_df.mean(axis=0)
    std = fit_df.std(axis=0).replace(0.0, 1.0)
    fit_x = ((fit_df - mean) / std).to_numpy(dtype=np.float32)

    n_comp = max(1, min(projection_components, fit_x.shape[1], fit_x.shape[0]))
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(fit_x)

    grp = GaussianRandomProjection(n_components=n_comp, random_state=42)
    grp.fit(fit_x)

    for split, df in split_frames.items():
        q = df.reindex(columns=proj_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        q_x = ((q - mean) / std).to_numpy(dtype=np.float32)

        pca_arr = pca.transform(q_x).astype(np.float32)
        grp_arr = grp.transform(q_x).astype(np.float32)

        pca_cols = [f"proj__pca_{i:02d}" for i in range(pca_arr.shape[1])]
        grp_cols = [f"proj__grp_{i:02d}" for i in range(grp_arr.shape[1])]

        out[split][pca_cols] = pca_arr
        out[split][grp_cols] = grp_arr

        if tenure_col in df.columns:
            tenure = _to_numeric(df[tenure_col]).fillna(0.0).astype(np.float32)
            out[split]["proj__tenure_sin_12"] = np.sin(tenure * 2.0 * np.pi / 12.0).astype(np.float32)
            out[split]["proj__tenure_cos_12"] = np.cos(tenure * 2.0 * np.pi / 12.0).astype(np.float32)
            out[split]["proj__tenure_sin_24"] = np.sin(tenure * 2.0 * np.pi / 24.0).astype(np.float32)
            out[split]["proj__tenure_cos_24"] = np.cos(tenure * 2.0 * np.pi / 24.0).astype(np.float32)

    return (
        out,
        list(out["train"].columns),
        {
            "enabled": True,
            "n_components": int(n_comp),
            "fit_rows": int(len(fit_df)),
            "fit_source": "original_reference" if original_df is not None else "train_split",
            "projection_columns": proj_cols,
        },
    )


def _load_original_reference(path_str: str | None) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if not path_str:
        return None, {"loaded": False, "reason": "path_missing"}

    path = Path(path_str)
    if not path.exists():
        return None, {"loaded": False, "reason": "file_missing", "path": str(path)}

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive error detail
        return None, {"loaded": False, "reason": f"read_failed:{exc}", "path": str(path)}

    return df, {"loaded": True, "path": str(path), "rows": int(len(df)), "cols": int(len(df.columns))}


def build_advanced_feature_set(split_frames: dict[str, pd.DataFrame], cfg: Any) -> AdvancedFeatureResult:
    splits = ["train", "valid", "test", "train_folds"]
    frames = {s: split_frames[s] for s in splits if s in split_frames}

    features_by_split = {s: pd.DataFrame(index=frames[s].index) for s in frames}
    family_columns: dict[str, list[str]] = {k: [] for k in DEFAULT_FAMILY_PREFIXES}
    metadata: dict[str, Any] = {}

    original_df, original_meta = _load_original_reference(getattr(cfg, "original_reference_csv", None))
    metadata["original_reference"] = original_meta

    categorical_columns = getattr(cfg, "categorical_columns", DEFAULT_TELCO_CATEGORICAL_COLUMNS)
    service_columns = getattr(cfg, "service_columns", DEFAULT_TELCO_SERVICE_COLUMNS)
    numeric_columns = getattr(
        cfg,
        "numeric_columns_for_bins",
        ["Soil_Moisture", "Rainfall_mm", "Previous_Irrigation_mm"],
    )
    te_stats = getattr(cfg, "te_stats", DEFAULT_TE_STATS)
    enabled_families = set(getattr(cfg, "enabled_feature_families", list(DEFAULT_FAMILY_PREFIXES.keys())))

    monthly_col = getattr(cfg, "monthly_charge_col", "Soil_Moisture")
    total_col = getattr(cfg, "total_charge_col", "Rainfall_mm")
    tenure_col = getattr(cfg, "tenure_col", "Previous_Irrigation_mm")
    phone_col = getattr(cfg, "phone_service_col", "PhoneService")
    internet_col = getattr(cfg, "internet_service_col", "InternetService")

    # Shared categorical signal frame used by TE + frequency + categorical cross model columns.
    cat_frames, cat_family_meta, cat_meta = _build_categorical_signal_frames(
        split_frames=frames,
        categorical_columns=categorical_columns,
        numeric_columns_for_bins=numeric_columns,
        monthly_col=monthly_col,
        total_col=total_col,
        quantile_bins=getattr(cfg, "quantile_bins", [50, 200, 1000, 5000]),
        fixed_bins=getattr(cfg, "fixed_width_bins", [20, 50, 100]),
        log_bins=getattr(cfg, "log_bins", [20, 50]),
    )
    metadata["categorical_signals"] = {"counts": {k: len(v) for k, v in cat_family_meta.items()}, **cat_meta}

    # Keep direct cross-code columns in output.
    cross_code_cols = [c for c in cat_frames["train"].columns if c.startswith("cross__")]
    if cross_code_cols and "cross_features" in enabled_families:
        for s in features_by_split:
            features_by_split[s][cross_code_cols] = cat_frames[s][cross_code_cols]
        family_columns["cross_features"].extend(cross_code_cols)

    if "target_encoding" in enabled_families:
        te_frames, te_meta, te_cols = _nested_target_encoding(
            split_frames=frames,
            cat_frames=cat_frames,
            target_col=getattr(cfg, "target_col", "Irrigation_Need"),
            te_stats=te_stats,
            inner_folds=getattr(cfg, "nested_te_folds", 5),
            fold_col=getattr(cfg, "fold_col", "cv_fold"),
            original_df=original_df,
            original_target_col=getattr(cfg, "original_target_col", "Irrigation_Need"),
        )
        metadata["target_encoding"] = te_meta
        for s in features_by_split:
            features_by_split[s] = pd.concat([features_by_split[s], te_frames[s]], axis=1)
        family_columns["target_encoding"].extend(te_cols)

    if "arithmetic" in enabled_families:
        arith_frames, arith_cols = _add_arithmetic_features(
            split_frames=frames,
            monthly_col=monthly_col,
            total_col=total_col,
            tenure_col=tenure_col,
        )
        for s in features_by_split:
            features_by_split[s] = pd.concat([features_by_split[s], arith_frames[s]], axis=1)
        family_columns["arithmetic"].extend(arith_cols)

    if "multi_scale_binning" in enabled_families:
        bin_frames, bin_cols = _add_multi_scale_numeric_bins(
            split_frames=frames,
            numeric_columns=numeric_columns,
            quantile_bins=getattr(cfg, "quantile_bins", [50, 200, 1000, 5000]),
            fixed_bins=getattr(cfg, "fixed_width_bins", [20, 50, 100]),
            log_bins=getattr(cfg, "log_bins", [20, 50]),
        )
        for s in features_by_split:
            features_by_split[s] = pd.concat([features_by_split[s], bin_frames[s]], axis=1)
        family_columns["multi_scale_binning"].extend(bin_cols)

    if "frequency_count" in enabled_families:
        freq_candidate_cols = [
            c
            for c in cat_frames["train"].columns
            if c.startswith("cat__raw__")
            or c.startswith("cat__bin_")
            or c.startswith("cat__anchor__")
            or c.startswith("cat__snapprod__")
        ]
        # Frequency features use the string-key columns from cat_frames.
        freq_source_frames = {s: pd.DataFrame(index=cat_frames[s].index) for s in cat_frames}
        for s in freq_source_frames:
            freq_source_frames[s][freq_candidate_cols] = cat_frames[s][freq_candidate_cols]
        freq_frames, freq_cols = _add_frequency_features(
            split_frames=freq_source_frames,
            columns=freq_candidate_cols,
            original_df=original_df,
        )
        for s in features_by_split:
            features_by_split[s] = pd.concat([features_by_split[s], freq_frames[s]], axis=1)
        family_columns["frequency_count"].extend(freq_cols)

    if "service_aggregations" in enabled_families:
        svc_frames, svc_cols = _add_service_aggregations(
            split_frames=frames,
            service_columns=service_columns,
            internet_col=internet_col,
            phone_col=phone_col,
        )
        for s in features_by_split:
            features_by_split[s] = pd.concat([features_by_split[s], svc_frames[s]], axis=1)
        family_columns["service_aggregations"].extend(svc_cols)

    if "original_lookup" in enabled_families:
        lookup_frames, lookup_cols, lookup_meta = _add_original_lookup_features(
            split_frames=frames,
            original_df=original_df,
            monthly_col=monthly_col,
            total_col=total_col,
            tenure_col=tenure_col,
            original_target_col=getattr(cfg, "original_target_col", "Irrigation_Need"),
        )
        metadata["original_lookup"] = lookup_meta
        for s in features_by_split:
            features_by_split[s] = pd.concat([features_by_split[s], lookup_frames[s]], axis=1)
        family_columns["original_lookup"].extend(lookup_cols)

    if "radix_interactions" in enabled_families:
        radix_frames, radix_cols = _add_radix_features(
            split_frames=frames,
            categorical_columns=categorical_columns,
            monthly_col=monthly_col,
            total_col=total_col,
        )
        for s in features_by_split:
            features_by_split[s] = pd.concat([features_by_split[s], radix_frames[s]], axis=1)
        family_columns["radix_interactions"].extend(radix_cols)

    if "artifact_detection" in enabled_families:
        artifact_frames, artifact_cols, artifact_meta = _add_artifact_features(
            split_frames=frames,
            numeric_columns=numeric_columns,
            monthly_col=monthly_col,
            total_col=total_col,
            tfidf_max_features=getattr(cfg, "tfidf_max_features", 24),
        )
        metadata["artifact_detection"] = artifact_meta
        for s in features_by_split:
            features_by_split[s] = pd.concat([features_by_split[s], artifact_frames[s]], axis=1)
        family_columns["artifact_detection"].extend(artifact_cols)

    if "projection_manifold" in enabled_families:
        proj_frames, proj_cols, proj_meta = _add_projection_features(
            split_frames=frames,
            original_df=original_df,
            monthly_col=monthly_col,
            total_col=total_col,
            tenure_col=tenure_col,
            projection_components=getattr(cfg, "projection_components", 12),
        )
        metadata["projection_manifold"] = proj_meta
        for s in features_by_split:
            features_by_split[s] = pd.concat([features_by_split[s], proj_frames[s]], axis=1)
        family_columns["projection_manifold"].extend(proj_cols)

    # Remove duplicated columns caused by overlaps and ensure numeric compatibility for model stage.
    for s in features_by_split:
        features_by_split[s] = features_by_split[s].loc[:, ~features_by_split[s].columns.duplicated()].copy()

    family_feature_counts = {k: int(len(v)) for k, v in family_columns.items()}

    return AdvancedFeatureResult(
        features_by_split=features_by_split,
        family_feature_counts=family_feature_counts,
        family_columns=family_columns,
        metadata=metadata,
    )
