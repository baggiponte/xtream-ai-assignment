"""Training and forecasting pipeline."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from powerload.dataloader import DataLoader
from powerload.pipeline import ForecastingPipeline

if TYPE_CHECKING:
    from argparse import Namespace


def main(args: Namespace) -> None:
    """Pipeline entry point."""
    window = args.training_window
    fh = args.forecasting_horizon
    strat = args.validation_strategy

    dataloader = DataLoader(cutoff=2019, ignore=2022)

    (
        data_train,
        data_test,
        target_train,
        target_test,
    ) = dataloader.get_train_test_splits()

    scores = ["neg_mean_absolute_percentage_error", "neg_mean_absolute_error"]

    pipe = ForecastingPipeline(data_train, target_train, scores=scores)

    pipe.fit()

    pipe.validate(window, fh, strat)

    predictions = pipe.predict(data_test)

    mape = mean_absolute_percentage_error(target_test, predictions)
    mae = mean_absolute_error(target_test, predictions)

    print(f"MAE: {mae:,.0f} GHW", f"MAPE: {mape:.1%}", sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Powerload forecasting pipeline",
    )

    parser.add_argument(
        "--training-window",
        type=int,
        default=365 * 10,
        help="Cross-validation training window, in days (default: 365*10)",
    )
    parser.add_argument(
        "--forecasting-horizon",
        type=int,
        default=365,
        help="Cross-validation forecasting horizon, in days (default: 365)",
    )
    parser.add_argument(
        "--validation-strategy",
        type=str,
        default="rolling",
        help="Cross-validation strategy: 'rolling' or 'expanding' (default: 'rolling')",
    )

    args: Namespace = parser.parse_args()
    main(args)
