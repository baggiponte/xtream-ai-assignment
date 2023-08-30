"""Dataloader utilities."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

import polars as pl

from powerload.datasets import fetch_powerload
from powerload.preprocessing import add_holidays, extract_datetime_features

if TYPE_CHECKING:
    from powerload.datasets import Dataset


class DataLoader:
    """Load the data and prepare it for training."""

    def __init__(self, *, cutoff: int, ignore: int = 2022) -> None:
        self.cutoff = cutoff
        self.ignore = ignore

    @staticmethod
    def load_data() -> tuple[pl.DataFrame, list[str], list[str]]:
        """Load the training data."""
        dset: Dataset = fetch_powerload(parser="polars")  # type: ignore[call-overload]

        return dset.data, dset.feature_names, dset.target_names  # type: ignore[return-value]

    @staticmethod
    def prepare_training_data(
        data: pl.DataFrame,
        date_col: str,
    ) -> pl.DataFrame:
        """Process the data for training."""
        return (
            data.pipe(extract_datetime_features, date_col, ["year", "month", "weekday"])
            .pipe(add_holidays, date_col, "IT")
            .select(pl.all().exclude("load"))
        )

    def get_train_test_splits(
        self,
    ) -> tuple[pl.DataFrame, ...]:
        """Split the data into train and test sets."""
        rawdata, date_cols, _ = self.load_data()

        data = self.prepare_training_data(rawdata, *date_cols)

        arrays = [
            (
                df.filter(
                    pl.col(date_cols).dt.year().lt(self.cutoff),
                ).select(
                    pl.all().exclude(date_cols),
                ),
                df.filter(
                    pl.col(date_cols)
                    .dt.year()
                    .is_between(self.cutoff, self.ignore, closed="none"),
                ).select(
                    pl.all().exclude(date_cols),
                ),
            )
            for df in (data, rawdata)
        ]

        return tuple(chain.from_iterable(arrays))
