from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_data: Path
    processed_data: Path
    level1_features_dir: Path
    outputs_root: Path
    level2_results_dir: Path
    oof_dir: Path
    pred_dir: Path



def get_paths() -> Paths:
    root = Path(__file__).resolve().parents[2]
    processed_data = root / "data" / "processed"
    outputs_root = root / "outputs"
    return Paths(
        root=root,
        raw_data=root / "data" / "raw",
        processed_data=processed_data,
        level1_features_dir=processed_data / "level1_features",
        outputs_root=outputs_root,
        level2_results_dir=outputs_root / "level2_results",
        oof_dir=outputs_root / "oof",
        pred_dir=outputs_root / "pred",
    )
