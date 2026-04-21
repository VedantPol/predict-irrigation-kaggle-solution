#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "outputs" / "external_preds" / "nina2025"
OUT = ROOT / "outputs" / "submissions"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    required = {"id", "Irrigation_Need"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    return df[["id", "Irrigation_Need"]].copy()


def _merge_predictions(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="id", how="inner")
    return merged


def _voting_schema(row: pd.Series, cols: list[str]) -> str:
    base_cols = cols[:-1]
    fallback = cols[-1]
    if all(row[base_cols[0]] == row[c] for c in base_cols):
        return row[base_cols[0]]
    return row[fallback]


def _build_notebook_style_submission() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_07 = EXT / "ps-s6e4-07"
    data_74 = EXT / "ps-s6e4-74"
    data_85 = EXT / "ps-s6e4-85"

    df_a = _read_csv(data_07 / "0.97971.a.csv").rename(columns={"Irrigation_Need": "A"})
    df_b = _read_csv(data_07 / "0.97971.b.csv").rename(columns={"Irrigation_Need": "B"})
    df_c = _read_csv(data_07 / "0.97971.c.csv").rename(columns={"Irrigation_Need": "C"})
    df_d = _read_csv(data_07 / "0.97971.d.csv").rename(columns={"Irrigation_Need": "D"})
    df_x = _read_csv(data_07 / "0.97971.x.csv").rename(columns={"Irrigation_Need": "X"})
    df_8010 = _read_csv(data_07 / "0.98010.csv").rename(columns={"Irrigation_Need": "s8010"})

    dfs = _merge_predictions([df_a, df_b, df_c, df_d, df_x, df_8010])
    cols = ["A", "B", "C", "D", "X"]

    voted = dfs.copy()
    voted["Irrigation_Need"] = voted.apply(lambda x: _voting_schema(x, cols), axis=1)

    transfer = dfs.copy()
    transfer["Irrigation_Need"] = transfer.apply(
        lambda row: row["X"] if len({row["A"], row["B"], row["C"], row["D"]}) != 1 else row["s8010"],
        axis=1,
    )

    df74 = _read_csv(data_74 / "5(4) - 0.98074.csv").rename(columns={"Irrigation_Need": "Irrigation_Need_74"})
    df72 = _read_csv(data_85 / "5(9) - 0.98072.csv").rename(columns={"Irrigation_Need": "Irrigation_Need_72"})
    df_aux = _read_csv(data_85 / "Aux - 0.97254.csv").rename(columns={"Irrigation_Need": "Irrigation_Need_aux"})

    aux_merge = _merge_predictions([df74, df72, df_aux])
    aux_merge["Irrigation_Need"] = aux_merge.apply(
        lambda row: row["Irrigation_Need_74"]
        if row["Irrigation_Need_74"] == row["Irrigation_Need_72"]
        else row["Irrigation_Need_aux"],
        axis=1,
    )

    notebook_final = aux_merge[["id", "Irrigation_Need"]].copy()
    transfer_only = transfer[["id", "Irrigation_Need"]].copy()
    return notebook_final, transfer_only


def _majority_or_fallback(a: str, b: str, c: str, fallback: str) -> str:
    vals = [a, b, c]
    if vals.count(a) >= 2:
        return a
    if vals.count(b) >= 2:
        return b
    if vals.count(c) >= 2:
        return c
    return fallback


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebook_final, transfer_only = _build_notebook_style_submission()
    top_88 = _read_csv(EXT / "ps-s6e4-85" / "Top - 0.98088.csv").rename(
        columns={"Irrigation_Need": "top88"}
    )

    ours_path = OUT / "submission_final_blend.csv"
    ours = _read_csv(ours_path).rename(columns={"Irrigation_Need": "ours"}) if ours_path.exists() else None

    nb = notebook_final.rename(columns={"Irrigation_Need": "nb"})
    transfer = transfer_only.rename(columns={"Irrigation_Need": "transfer"})

    merged = _merge_predictions([nb, transfer, top_88])
    if ours is not None:
        merged = _merge_predictions([merged, ours])

    # Candidate 1: exact notebook transfer strategy endpoint.
    sub_nb = nb.rename(columns={"nb": "Irrigation_Need"})
    sub_nb.to_csv(OUT / "submission_external_notebook981.csv", index=False)

    # Candidate 2: strongest single external file in dataset.
    sub_top = merged[["id", "top88"]].rename(columns={"top88": "Irrigation_Need"})
    sub_top.to_csv(OUT / "submission_external_top088.csv", index=False)

    # Candidate 3: hybrid vote with our current final blend if available.
    if ours is not None:
        hybrid = merged[["id", "nb", "top88", "ours"]].copy()
        hybrid["Irrigation_Need"] = hybrid.apply(
            lambda row: _majority_or_fallback(
                row["nb"],
                row["top88"],
                row["ours"],
                fallback=row["top88"],
            ),
            axis=1,
        )
        hybrid[["id", "Irrigation_Need"]].to_csv(
            OUT / "submission_external_hybrid_vote.csv",
            index=False,
        )
        disagree_nb_ours = int((hybrid["nb"] != hybrid["ours"]).sum())
        disagree_top_ours = int((hybrid["top88"] != hybrid["ours"]).sum())
        print(f"[EXTERNAL_BLEND] Disagreements nb vs ours: {disagree_nb_ours}")
        print(f"[EXTERNAL_BLEND] Disagreements top88 vs ours: {disagree_top_ours}")

    print(f"[EXTERNAL_BLEND] Saved: {OUT / 'submission_external_notebook981.csv'}")
    print(f"[EXTERNAL_BLEND] Saved: {OUT / 'submission_external_top088.csv'}")
    if ours is not None:
        print(f"[EXTERNAL_BLEND] Saved: {OUT / 'submission_external_hybrid_vote.csv'}")


if __name__ == "__main__":
    main()

