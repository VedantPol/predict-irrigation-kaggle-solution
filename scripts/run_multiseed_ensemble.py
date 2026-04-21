#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_early_warning.pipeline import run_stage_with_config
from scripts.run_pipeline import BASELINE_CONFIG


SEEDS = [42, 314159, 271828, 161803]
MODELS_PER_LIBRARY = 8
OPTUNA_TRIALS_PER_LIBRARY = 10
OPTUNA_MAX_TUNED = 4


def _read_final_blend_score() -> float | None:
    path = ROOT / "outputs" / "level2_results" / "final_blend" / "final_blend_summary.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        return float(payload.get("best_valid_auc"))
    except Exception:
        return None


def main() -> None:
    score_log: list[dict[str, object]] = []
    for i, seed in enumerate(SEEDS):
        cfg = deepcopy(BASELINE_CONFIG)
        cfg.update(
            {
                "random_state": int(seed),
                "models_per_library": int(MODELS_PER_LIBRARY),
                "tree_optuna_trials_per_library": int(OPTUNA_TRIALS_PER_LIBRARY),
                "tree_optuna_max_tuned_models_per_library": int(OPTUNA_MAX_TUNED),
                # Keep historical OOF/pred across seeds for richer blending.
                "cleanup_oof_pred_before_baseline": False,
                "cleanup_level2_outputs_before_baseline": False,
            }
        )

        print(f"[MULTISEED] ({i + 1}/{len(SEEDS)}) baseline seed={seed}")
        run_stage_with_config("baseline", baseline_config=cfg)

        print("[MULTISEED] Rebuilding downstream blends")
        run_stage_with_config("hill_climb")
        run_stage_with_config("stacking")
        run_stage_with_config("final_blend")

        score = _read_final_blend_score()
        print(f"[MULTISEED] seed={seed} | final_blend_valid={score}")
        score_log.append({"seed": int(seed), "final_blend_valid": score})

    out = ROOT / "outputs" / "level2_results" / "multiseed_score_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score_log, indent=2), encoding="utf-8")
    print(f"[MULTISEED] Saved log: {out}")


if __name__ == "__main__":
    main()

