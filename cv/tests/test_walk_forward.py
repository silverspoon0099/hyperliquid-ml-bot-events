"""Tests for cv.walk_forward (spec §9.1 + DR v3.0.9 §2-§4)."""
from datetime import date, timedelta, timezone, datetime

import pandas as pd

from cv.walk_forward import Fold, generate_folds, split_fold


def test_fold_count_for_phase_a_range():
    """2019-01-01 to 2026-05-01 with default config → ~20 folds."""
    folds = generate_folds(
        data_start=date(2019, 1, 1), data_end=date(2026, 5, 1),
        initial_train_months=24, val_months=3, oot_months=3, step_months=3,
    )
    # First val_start = 2021-01-01. oot_end advances 3 mo per step.
    # Last oot_end ≤ 2026-05-01: oot_end can be 2026-04-01 (val_end=2026-01-01,
    # val_start=2025-10-01). Count from 2021-Q1 val_start through 2025-Q3:
    # quarters = 19. (Inclusive count.)
    assert 18 <= len(folds) <= 20
    # First and last fold sanity
    assert folds[0].train_start == date(2019, 1, 1)
    assert folds[0].val_start == date(2021, 1, 1)
    assert folds[0].val_end == date(2021, 4, 1)
    assert folds[0].oot_end == date(2021, 7, 1)


def test_fold_boundaries_first_three():
    folds = generate_folds(
        data_start=date(2019, 1, 1), data_end=date(2026, 5, 1),
        initial_train_months=24, val_months=3, oot_months=3, step_months=3,
    )
    assert folds[0].val_start == date(2021, 1, 1)
    assert folds[1].val_start == date(2021, 4, 1)
    assert folds[2].val_start == date(2021, 7, 1)
    # Fold spans exactly 6 months end-to-end (val + oot)
    for f in folds[:3]:
        assert (f.oot_end.year - f.val_start.year) * 12 + (f.oot_end.month - f.val_start.month) == 6


def test_split_purge_and_embargo_applied():
    """Hourly bars over 11 days. Each section has > 24 bars so purge and
    embargo apply. Verify exact bar-count gap = 24."""
    n = 11 * 24  # 264 hourly bars over 11 days
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "bar_id": range(1, n + 1),
        "bar_close_ts": [base + timedelta(hours=i) for i in range(n)],
        "feat": list(range(n)),
    })
    # train: [01-01, 01-05) = 96 bars; val: [01-05, 01-08) = 72; oot: [01-08, 01-12) = 96
    fold = Fold(
        fold_id=1,
        train_start=date(2020, 1, 1),
        val_start=date(2020, 1, 5),
        val_end=date(2020, 1, 8),
        oot_end=date(2020, 1, 12),
    )
    parts = split_fold(df, fold, purge_bars=24, embargo_bars=24)
    assert len(parts["train"]) == 96 - 24  # 72
    assert len(parts["val"]) == 72
    assert len(parts["oot"]) == 96 - 24    # 72

    # Bar-id gap from train-end to val-start = 24 (purge) + 1 (boundary) = 25
    assert int(parts["val"]["bar_id"].iloc[0]) - int(parts["train"]["bar_id"].iloc[-1]) == 25

    # Bar-id gap from val-end to OOT-start = 24 (embargo) + 1 (boundary) = 25
    assert int(parts["oot"]["bar_id"].iloc[0]) - int(parts["val"]["bar_id"].iloc[-1]) == 25


def test_split_handles_short_train_no_purge():
    """If train has fewer bars than purge_bars, return train as-is (no negative slicing)."""
    n = 30
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "bar_id": range(1, n + 1),
        "bar_close_ts": [base + timedelta(hours=i) for i in range(n)],
        "feat": list(range(n)),
    })
    fold = Fold(
        fold_id=1,
        train_start=date(2020, 1, 1),
        val_start=date(2020, 1, 1),  # no train range
        val_end=date(2020, 1, 2),
        oot_end=date(2020, 1, 3),
    )
    parts = split_fold(df, fold, purge_bars=100, embargo_bars=100)
    # Train empty (val starts at train_start) — slicing should not fail.
    assert len(parts["train"]) == 0
