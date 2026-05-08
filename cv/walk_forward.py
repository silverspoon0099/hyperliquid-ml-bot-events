"""Walk-forward fold construction (spec §9.1 + DR v3.0.9 §2-§4).

Calendar-anchored expanding-window folds; bar-count purge/embargo.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: date
    val_start: date   # also = train_end (exclusive)
    val_end: date     # also = oot_start
    oot_end: date     # exclusive


def _add_months(d: date, months: int) -> date:
    new_month = d.month - 1 + months
    new_year = d.year + new_month // 12
    new_month = new_month % 12 + 1
    return date(new_year, new_month, 1)


def generate_folds(
    data_start: date,
    data_end: date,
    initial_train_months: int = 24,
    val_months: int = 3,
    oot_months: int = 3,
    step_months: int = 3,
) -> list[Fold]:
    """Walk-forward expanding-window folds. Stops when oot_end > data_end."""
    folds: list[Fold] = []
    fold_id = 0
    val_start = _add_months(data_start, initial_train_months)
    while True:
        val_end = _add_months(val_start, val_months)
        oot_end = _add_months(val_end, oot_months)
        if oot_end > data_end:
            break
        fold_id += 1
        folds.append(Fold(
            fold_id=fold_id,
            train_start=data_start,
            val_start=val_start,
            val_end=val_end,
            oot_end=oot_end,
        ))
        val_start = _add_months(val_start, step_months)
    return folds


def split_fold(
    df: pd.DataFrame,
    fold: Fold,
    purge_bars: int = 24,
    embargo_bars: int = 24,
    ts_col: str = "bar_close_ts",
) -> dict[str, pd.DataFrame]:
    """Split sorted DataFrame into train/val/oot subsets with purge+embargo.

    - Purge: drop last `purge_bars` of train (their labels could exit in val).
    - Embargo: drop first `embargo_bars` of OOT (val-fitted Platt could
      have informed those bars indirectly).
    """
    ts = pd.to_datetime(df[ts_col], utc=True)
    train_mask = (ts >= pd.Timestamp(fold.train_start, tz="UTC")) & (
        ts < pd.Timestamp(fold.val_start, tz="UTC")
    )
    val_mask = (ts >= pd.Timestamp(fold.val_start, tz="UTC")) & (
        ts < pd.Timestamp(fold.val_end, tz="UTC")
    )
    oot_mask = (ts >= pd.Timestamp(fold.val_end, tz="UTC")) & (
        ts < pd.Timestamp(fold.oot_end, tz="UTC")
    )

    train = df[train_mask]
    if purge_bars > 0 and len(train) > purge_bars:
        train = train.iloc[:-purge_bars]

    val = df[val_mask]

    oot = df[oot_mask]
    if embargo_bars > 0 and len(oot) > embargo_bars:
        oot = oot.iloc[embargo_bars:]

    return {"train": train, "val": val, "oot": oot}
